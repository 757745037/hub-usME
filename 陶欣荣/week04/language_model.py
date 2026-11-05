"""
字符级因果 Transformer 语言模型，支持读取原来的 RNN / LSTM 权重。
用法:
    python language_model.py --baseline best_model_rnn.pt
    python language_model.py --evaluate best_model_transformer.pt --baseline best_model_rnn.pt
"""

import math
import argparse
import glob
import hashlib
import json
from pathlib import Path
from time import perf_counter
import torch.nn.functional as F
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────── 数据 ───────────────────────────

def load_corpus(pattern="*.txt"):
    texts = []
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)


def build_vocab(text, max_vocab=None):
    """
    构建字表：按出现频率取前 max_vocab 个字（None = 全部字符）。
    语料里的生僻字超出字表后会被 CharDataset 过滤掉 —— 字表越小模型越小，
    但覆盖不足会丢信息，这是词表大小的取舍。
    """
    from collections import Counter
    counter = Counter(text)
    chars = [c for c, _ in counter.most_common(max_vocab)] if max_vocab else sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    """
    字符级语言模型的训练样本：输入 x = 连续 seq_len 个字符，目标 y = 往后错一位。

    stride 控制切块间隔：
        stride=1  —— 每个字符位置都当一个样本（样本大量重叠，一期语料能切出上百万个样本）
        stride=seq_len —— 不重叠切块（标准做法，样本数 = 语料长度 / seq_len）
    语料一大，stride=1 会让一轮训练变得极慢，这不是模型慢，而是样本数被放大了 seq_len 倍。
    """

    def __init__(self, text, char2idx, seq_len, stride=1):
        self.seq_len = seq_len
        self.stride = max(1, stride)
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, (len(self.data) - self.seq_len - 1) // self.stride + 1)

    def __getitem__(self, idx):
        start = idx * self.stride
        x = self.data[start: start + self.seq_len]
        y = self.data[start + 1: start + self.seq_len + 1]
        return x, y


# ─────────────────────────── 模型 ───────────────────────────

class LM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, model_type, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        rnn_cls = nn.LSTM if model_type == "lstm" else nn.RNN
        self.rnn = rnn_cls(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        e = self.drop(self.embed(x))
        out, _ = self.rnn(e)
        logits = self.fc(self.drop(out))   # (B, T, V)
        return logits


class CausalBlock(nn.Module):
    """多头因果自注意力 + 前馈网络，两个子层均使用预归一化和残差连接。"""
    def __init__(self, width, heads, ff_dim, dropout):
        super().__init__()
        self.heads = heads
        self.attn_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, ff_dim), nn.GELU(),
            nn.Linear(ff_dim, width), nn.Dropout(dropout),
        )

    def forward(self, x):
        batch, length, width = x.shape
        qkv = self.qkv(self.attn_norm(x)).reshape(
            batch, length, 3, self.heads, width // self.heads
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        x = x + self.dropout(self.proj(attended))
        return x + self.ff(x)


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()
        width, heads = cfg["embed_dim"], cfg["heads"]
        if heads < 1 or width % heads:
            raise ValueError("embed_dim 必须是 heads 的整数倍")
        self.max_len = cfg["seq_len"]
        self.embed = nn.Embedding(vocab_size, width)
        self.position = nn.Embedding(self.max_len, width)
        self.drop = nn.Dropout(cfg["dropout"])
        self.blocks = nn.Sequential(*[
            CausalBlock(width, heads, cfg["hidden_dim"], cfg["dropout"])
            for _ in range(cfg["num_layers"])
        ])
        self.norm = nn.LayerNorm(width)
        # 字表按频次排序。自适应 softmax 对常见字直接分类，对低频字分组，
        # 可在 CPU 上计算真实的归一化概率，而无需在训练时展开全部词表。
        cutoffs = [n for n in (256, 1024) if n < vocab_size]
        if cutoffs:
            self.output = nn.AdaptiveLogSoftmaxWithLoss(
                width, vocab_size, cutoffs, div_value=2, head_bias=True
            )
        else:
            self.output = nn.Linear(width, vocab_size)

    def features(self, x):
        if x.size(1) > self.max_len:
            raise ValueError("输入长度超过模型上下文窗口")
        pos = torch.arange(x.size(1), device=x.device)
        return self.norm(self.blocks(self.drop(self.embed(x) + self.position(pos))))

    def token_loss(self, x, y):
        h = self.features(x).flatten(0, 1)
        if isinstance(self.output, nn.AdaptiveLogSoftmaxWithLoss):
            return self.output(h, y.flatten()).loss
        return F.cross_entropy(self.output(h), y.flatten())

    def forward(self, x):
        h = self.features(x)
        if isinstance(self.output, nn.AdaptiveLogSoftmaxWithLoss):
            return self.output.log_prob(h.flatten(0, 1)).reshape(*x.shape, -1)
        return self.output(h)


def make_model(vocab_size, cfg):
    if cfg["model"] == "transformer":
        return TransformerLM(vocab_size, cfg)
    return LM(vocab_size, cfg["embed_dim"], cfg["hidden_dim"],
              cfg["num_layers"], cfg["model"], cfg["dropout"])


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = make_model(len(ckpt["char2idx"]), ckpt["args"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model.eval(), ckpt


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    total_loss = 0.0
    total_tokens = 0
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if isinstance(model, TransformerLM):
                loss = model.token_loss(x, y)
            else:
                loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

    if not total_tokens:
        raise ValueError("没有可计分的字符，请检查语料和 seq_len")
    avg_loss = total_loss / total_tokens
    return avg_loss, math.exp(avg_loss)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="transformer", choices=["transformer", "rnn", "lstm"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--stride", type=int, help="训练切块步长，默认等于 seq_len")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=192, help="前馈层宽度或循环网络隐藏维度")
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--max_vocab", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4, help="CPU 计算线程数")
    parser.add_argument("--corpus", default="corpus_general.txt")
    parser.add_argument("--save", help="模型路径，默认 best_model_<模型>.pt")
    parser.add_argument("--baseline", help="在相同验证字符上评估一个旧权重")
    parser.add_argument("--evaluate", help="只复评指定权重，沿用其划分和词表")
    args = parser.parse_args()
    if args.epochs < 1 or args.seq_len < 1 or args.batch_size < 1:
        parser.error("epochs、seq_len 和 batch_size 必须为正整数")
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = vars(args).copy()
    if args.evaluate:
        model, ckpt = load_checkpoint(args.evaluate, device)
        cfg = ckpt["args"]
    save = Path(args.save or f"best_model_{args.model}.pt")
    if args.baseline and not args.evaluate and save.resolve() == Path(args.baseline).resolve():
        parser.error("保存路径必须与对照权重不同")
    torch.manual_seed(cfg["seed"])
    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError(f"没有读到语料: {args.corpus}")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if args.baseline:
        baseline, old = load_checkpoint(args.baseline, device)
    if args.evaluate:
        char2idx, idx2char = ckpt["char2idx"], ckpt["idx2char"]
        if ckpt.get("corpus_sha256", digest) != digest:
            raise ValueError("复评语料与训练语料的指纹不一致")
    elif args.baseline:
        char2idx, idx2char = old["char2idx"], old["idx2char"]
    else:
        char2idx, idx2char = build_vocab(text, cfg["max_vocab"] or None)
    lines = text.splitlines()
    random.Random(cfg["seed"]).shuffle(lines)
    split = int(len(lines) * (1 - cfg["val_ratio"]))
    train_lines, val_lines = lines[:split], lines[split:]
    seen = set(train_lines)
    overlap = sum(line in seen for line in val_lines)
    seq_len = cfg["seq_len"]
    val_ds = CharDataset("\n".join(val_lines), char2idx, seq_len, seq_len)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    print(f"设备 {device}，词表 {len(char2idx)}，训练/验证行 {split}/{len(val_lines)}，"
          f"验证目标 {len(val_ds) * seq_len:,}，与训练重复的验证行 {overlap}")
    report = dict(corpus_sha256=digest, config=cfg, train_lines=split, val_lines=len(val_lines),
                  val_lines_also_in_train=overlap, val_tokens=len(val_ds) * seq_len, history=[])
    if args.baseline:
        if (old["char2idx"] != char2idx or
                any(old["args"][key] != cfg[key] for key in ("seed", "val_ratio", "seq_len"))):
            raise ValueError("对照模型的字表、划分或上下文长度不一致")
        loss, ppl = run_epoch(baseline, val_loader, device)
        print(f"{args.baseline}  验证 loss {loss:.5f}  PPL {ppl:.2f}", flush=True)
        report["baseline"] = dict(path=args.baseline, model=old["args"]["model"], loss=loss, ppl=ppl)
        del baseline, old
    if args.evaluate:
        loss, ppl = run_epoch(model, val_loader, device)
        print(f"{args.evaluate}  验证 loss {loss:.5f}  PPL {ppl:.2f}")
        return
    train_ds = CharDataset("\n".join(train_lines), char2idx, seq_len, args.stride or seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    model = make_model(len(char2idx), cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    report["parameters"] = sum(p.numel() for p in model.parameters())
    report["train_tokens_per_epoch"] = len(train_ds) * seq_len
    print(f"{args.model} 参数量 {report['parameters']:,}，每轮训练目标 {len(train_ds) * seq_len:,}",
          flush=True)
    best_ppl = float("inf")
    for epoch in range(1, args.epochs + 1):
        start = perf_counter()
        tr_loss, tr_ppl = run_epoch(model, train_loader, device, optimizer)
        va_loss, va_ppl = run_epoch(model, val_loader, device)
        report["history"].append(dict(epoch=epoch, train_loss=tr_loss, train_ppl=tr_ppl,
                                      val_loss=va_loss, val_ppl=va_ppl, seconds=perf_counter()-start))
        if va_ppl < best_ppl:
            best_ppl = va_ppl
            report.update(best_epoch=epoch, best_val_ppl=va_ppl)
            torch.save(dict(model_state=model.state_dict(), char2idx=char2idx, idx2char=idx2char,
                            args=cfg, epoch=epoch, val_ppl=va_ppl, corpus_sha256=digest), save)
        save.with_suffix(".metrics.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"轮次 {epoch}/{args.epochs}  训练 PPL {tr_ppl:.2f}  验证 PPL {va_ppl:.2f}  "
              f"耗时 {perf_counter()-start:.0f}s", flush=True)
    print(f"最佳验证 PPL {best_ppl:.2f}，权重已保存到 {save}")


if __name__ == "__main__":
    main()
