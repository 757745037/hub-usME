# -*- coding: utf-8 -*-
"""
train_transformer_lm.py — 作业：训练基于 Transformer 的语言模型

任务与课堂字符级 RNN/LSTM 语言模型相同：给定前文，预测下一个字。
序列模型换成解码器式 Transformer（自回归，而不是 BERT 那种双向填空）：

    Embedding + 可学习位置编码
        → 因果掩码的多头自注意力（只能看当前位置及左侧）
        → 前馈网络
        → 残差 + Pre-LN，堆叠若干层
        → 与词嵌入共享权重，输出下一个字的分布

损失是下一字交叉熵，困惑度 PPL = exp(loss)，越低越好。
语料 data/corpus.txt 取自课件 corpus_general.txt 的随机子集（约 80 万字），
方便在 CPU 上完整跑完。

用法：
    python train_transformer_lm.py
    python train_transformer_lm.py --epochs 8 --seq-len 32
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
import sys
import time
import random
import argparse
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ─── 超参数 ───────────────────────────────────────────────────────────
SEED = 42
SEQ_LEN = 32
STRIDE = 32
BATCH_SIZE = 64
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 2
D_FF = 512
DROPOUT = 0.1
LR = 1e-3
EPOCHS = 8
MAX_VOCAB = 2000
VAL_RATIO = 0.05
THREADS = 8

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(HERE, "data", "corpus.txt")


# ─── 1. 数据 ──────────────────────────────────────────────────────────
def load_lines(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        return [ln.strip() for ln in f if ln.strip()]


def build_vocab(text, max_vocab):
    """按出现次数取前 max_vocab 个字。字表外的字在切样本时丢掉。"""
    chars = [c for c, _ in Counter(text).most_common(max_vocab)]
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    """x 是连续 seq_len 个字，y 是向后错一位，即每个位置预测下一个字。"""

    def __init__(self, text, char2idx, seq_len, stride):
        self.seq_len = seq_len
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)
        self.stride = max(1, stride)
        self.n = max(0, (len(self.data) - seq_len - 1) // self.stride + 1)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        start = idx * self.stride
        x = self.data[start: start + self.seq_len]
        y = self.data[start + 1: start + self.seq_len + 1]
        return x, y


def ids_of(text, char2idx):
    return [char2idx[c] for c in text if c in char2idx]


# ─── 2. 模型 ──────────────────────────────────────────────────────────
class TransformerLM(nn.Module):
    """
    解码器式字符语言模型。
    nn.TransformerEncoderLayer 本身不带掩码，这里传入上三角因果掩码，
    让每个位置只能注意到自己和左边的字，从而可以逐字生成。
    Pre-LN（norm_first=True）比 BERT 的 Post-LN 更容易从零训稳。
    """

    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_len, dropout):
        super().__init__()
        self.max_len = max_len
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        b, t = x.shape
        if t > self.max_len:
            raise ValueError(f"序列长度 {t} 超过位置编码上限 {self.max_len}")
        pos = torch.arange(t, device=x.device)
        h = self.drop(self.embed(x) + self.pos(pos)[None, :, :])
        # True 表示禁止注意。对角线以上是「未来」，全部挡住。
        causal = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        h = self.blocks(h, mask=causal)
        h = self.ln(h)
        # 输出层与词嵌入共享权重，少一组 V×d 参数
        return F.linear(h, self.embed.weight)


@torch.no_grad()
def check_causal(model, vocab_size):
    """改未来的字，过去位置的 logits 应不变；改过去的字，后面应变化。"""
    was_training = model.training
    model.eval()
    x = torch.randint(0, vocab_size, (2, 16))
    y1 = model(x)
    x_future = x.clone()
    x_future[:, 8:] = torch.randint(0, vocab_size, (2, 8))
    y_future = model(x_future)
    x_past = x.clone()
    x_past[:, 0] = (x_past[:, 0] + 1) % vocab_size
    y_past = model(x_past)
    model.train(was_training)
    if not torch.allclose(y1[:, :8], y_future[:, :8], atol=1e-4):
        raise RuntimeError("因果掩码失效：未来的字影响了过去位置的预测")
    if torch.allclose(y1[:, 5], y_past[:, 5], atol=1e-4):
        raise RuntimeError("注意力未使用上文：修改过去的字后，后续预测没有变化")


# ─── 3. 训练 / 评估 ───────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, train):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0
    for x, y in loader:
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
    avg = total_loss / max(1, total_tokens)
    return avg, math.exp(avg)


def ngram_ppl(train_ids, val_ids, n, vocab_size):
    """加一平滑的 n-gram 困惑度，用来和 Transformer 对照。"""
    ctx_n = n - 1
    counts = Counter()
    ctx_counts = Counter()
    for i in range(ctx_n, len(train_ids)):
        ctx = tuple(train_ids[i - ctx_n:i])
        counts[(ctx, train_ids[i])] += 1
        ctx_counts[ctx] += 1
    logp = 0.0
    tokens = 0
    for i in range(ctx_n, len(val_ids)):
        ctx = tuple(val_ids[i - ctx_n:i])
        p = (counts[(ctx, val_ids[i])] + 1.0) / (ctx_counts[ctx] + vocab_size)
        logp += math.log(p)
        tokens += 1
    return math.exp(-logp / tokens)


@torch.no_grad()
def generate(model, char2idx, idx2char, prompt, steps, seq_len, temperature=0.9, top_k=10):
    model.eval()
    ids = [char2idx[c] for c in prompt if c in char2idx]
    if not ids:
        ids = [0]
    for _ in range(steps):
        x = torch.tensor(ids[-seq_len:], dtype=torch.long).unsqueeze(0)
        logits = model(x)[0, -1] / max(temperature, 1e-6)
        if top_k is not None and top_k < logits.numel():
            vals, keep = torch.topk(logits, top_k)
            masked = torch.full_like(logits, float("-inf"))
            masked[keep] = vals
            logits = masked
        prob = torch.softmax(logits, dim=-1)
        nxt = int(torch.multinomial(prob, 1))
        ids.append(nxt)
    return "".join(idx2char[i] for i in ids)


@torch.no_grad()
def next_topk(model, char2idx, idx2char, prompt, seq_len, k=8):
    model.eval()
    ids = [char2idx[c] for c in prompt if c in char2idx]
    if not ids:
        ids = [0]
    x = torch.tensor(ids[-seq_len:], dtype=torch.long).unsqueeze(0)
    prob = torch.softmax(model(x)[0, -1], dim=-1)
    val, idx = torch.topk(prob, k)
    return [(idx2char[int(i)], float(v)) for v, i in zip(val, idx)]


def plot_curve(history, path, bigram_ppl=None):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, history["train_loss"], "b-o", label="train loss")
    axes[0].plot(epochs, history["val_loss"], "r-o", label="val loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross entropy")
    axes[0].set_title("损失")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_ppl"], "b-o", label="train PPL")
    axes[1].plot(epochs, history["val_ppl"], "r-o", label="val PPL")
    if bigram_ppl is not None:
        axes[1].axhline(bigram_ppl, color="gray", ls="--",
                        label=f"bigram {bigram_ppl:.0f}")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("perplexity")
    axes[1].set_title("困惑度（对数坐标）")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, which="both")

    fig.suptitle("Transformer 字符级语言模型")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ─── 4. 主流程 ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=CORPUS_PATH)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--d-model", type=int, default=D_MODEL)
    parser.add_argument("--n-heads", type=int, default=N_HEADS)
    parser.add_argument("--n-layers", type=int, default=N_LAYERS)
    parser.add_argument("--d-ff", type=int, default=D_FF)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--max-vocab", type=int, default=MAX_VOCAB)
    parser.add_argument("--threads", type=int, default=THREADS)
    parser.add_argument("--save", default=os.path.join(HERE, "transformer_lm.pt"))
    parser.add_argument("--curve", default=os.path.join(HERE, "train_curve.png"))
    parser.add_argument("--report", default=os.path.join(HERE, "result.txt"))
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    random.seed(SEED)
    torch.manual_seed(SEED)

    if not os.path.exists(args.corpus):
        raise FileNotFoundError(f"找不到语料：{args.corpus}")

    lines = load_lines(args.corpus)
    random.shuffle(lines)
    split = int(len(lines) * (1 - VAL_RATIO))
    train_text = "\n".join(lines[:split])
    val_text = "\n".join(lines[split:])

    char2idx, idx2char = build_vocab(train_text, args.max_vocab)
    vocab_size = len(char2idx)
    train_ids = ids_of(train_text, char2idx)
    val_ids = ids_of(val_text, char2idx)

    train_ds = CharDataset(train_text, char2idx, args.seq_len, args.stride)
    val_ds = CharDataset(val_text, char2idx, args.seq_len, args.stride)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        drop_last=len(train_ds) >= args.batch_size,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        drop_last=False,
    )

    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_len=args.seq_len,
        dropout=args.dropout,
    )
    check_causal(model, vocab_size)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device: cpu   threads: {args.threads}")
    print(f"语料行数: {len(lines):,}   训练字数: {len(train_ids):,}   验证字数: {len(val_ids):,}")
    print(f"词表大小: {vocab_size}   样本: train {len(train_ds):,} / val {len(val_ds):,}")
    print(f"模型: Transformer decoder  d={args.d_model}  heads={args.n_heads}  "
          f"layers={args.n_layers}  ff={args.d_ff}")
    print(f"参数量: {n_params:,}")
    print("因果掩码自检通过")

    uni_ppl = ngram_ppl(train_ids, val_ids, 1, vocab_size)
    bi_ppl = ngram_ppl(train_ids, val_ids, 2, vocab_size)
    print(f"对照  加一平滑 unigram PPL: {uni_ppl:.2f}   bigram PPL: {bi_ppl:.2f}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {"train_loss": [], "val_loss": [], "train_ppl": [], "val_ppl": []}
    best_ppl = float("inf")
    best_epoch = 0

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, train=True)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, train=False)
        scheduler.step()
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_ppl"].append(tr_ppl)
        history["val_ppl"].append(va_ppl)

        mark = ""
        if va_ppl < best_ppl:
            best_ppl = va_ppl
            best_epoch = epoch
            mark = "  *"
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "vocab_size": vocab_size,
                "d_model": args.d_model,
                "n_heads": args.n_heads,
                "n_layers": args.n_layers,
                "d_ff": args.d_ff,
                "dropout": args.dropout,
                "seq_len": args.seq_len,
                "best_val_ppl": best_ppl,
                "best_epoch": best_epoch,
            }, args.save)
        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}{mark}")

    elapsed = time.time() - t0
    print(f"\n训练完成。耗时 {elapsed:.0f} 秒。最佳验证 PPL: {best_ppl:.2f}（epoch {best_epoch}）")
    print(f"模型已保存：{args.save}")

    # 用最佳权重做生成，避免最后一轮比最佳轮更差
    ckpt = torch.load(args.save, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    prompts = ["为什么", "怎么才能", "北京", "我喜欢"]
    samples = []
    print("\n--- 续写示例（temperature=0.9, top-k=10）---")
    for prompt in prompts:
        text = generate(model, char2idx, idx2char, prompt, steps=24,
                        seq_len=args.seq_len, temperature=0.9, top_k=10)
        samples.append(text)
        print(f"  {text}")

    top_prompt = "我"
    tops = next_topk(model, char2idx, idx2char, top_prompt, args.seq_len, k=8)
    print(f"\n--- 「{top_prompt}」的下一个字 ---")
    for ch, p in tops:
        print(f"  {ch}  {p:.3f}")

    plot_curve(history, args.curve, bigram_ppl=bi_ppl)
    print(f"训练曲线已保存：{args.curve}")

    lines_out = [
        "作业：训练基于 Transformer 的语言模型",
        "",
        "模型：解码器式 Transformer（因果自注意力 + 前馈 + Pre-LN）",
        f"结构：d_model={args.d_model}  heads={args.n_heads}  layers={args.n_layers}  "
        f"ff={args.d_ff}  seq_len={args.seq_len}",
        f"参数量：{n_params:,}",
        f"词表：训练集高频字 {vocab_size} 个",
        f"训练字数：{len(train_ids):,}    验证字数：{len(val_ids):,}",
        f"训练耗时：{elapsed:.0f} 秒（CPU，{args.epochs} 个 epoch）",
        "",
        f"加一平滑 unigram 验证 PPL：{uni_ppl:.2f}",
        f"加一平滑 bigram  验证 PPL：{bi_ppl:.2f}",
        f"Transformer 最佳验证 PPL：{best_ppl:.2f}（epoch {best_epoch}）",
        "",
        "各轮：",
        f"{'epoch':>5}  {'train_loss':>10}  {'train_ppl':>10}  {'val_loss':>10}  {'val_ppl':>10}",
    ]
    for i in range(args.epochs):
        lines_out.append(
            f"{i + 1:>5}  {history['train_loss'][i]:>10.4f}  {history['train_ppl'][i]:>10.2f}  "
            f"{history['val_loss'][i]:>10.4f}  {history['val_ppl'][i]:>10.2f}"
        )
    lines_out.append("")
    lines_out.append("续写示例：")
    lines_out.extend(f"  {s}" for s in samples)
    lines_out.append("")
    lines_out.append(f"「{top_prompt}」的下一个字：")
    lines_out.extend(f"  {ch}  {p:.3f}" for ch, p in tops)
    lines_out.append("")
    with open(args.report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out))
    print(f"结果已写入：{args.report}")


if __name__ == "__main__":
    main()
