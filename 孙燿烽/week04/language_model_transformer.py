"""
字符级因果 Transformer 语言模型，从头训练并计算 PPL。
用法:
    python language_model_transformer.py --corpus "data/*.txt" --epochs 20
    python language_model_transformer.py --corpus "data/*.txt" --embed_dim 256 --nhead 8 --ffn_dim 1024

默认 seq_len=64、stride=seq_len、batch_size=32、lr=3e-4。
语料按行随机划分训练集/验证集，请使用有足够有效行数的 UTF-8 文本。
权重、字表及训练参数保存到 best_transformer.pt。
"""

import math
import argparse
import glob
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────── 数据 ───────────────────────────

def load_corpus(pattern="*.txt"):
    texts = []
    for path in glob.glob(pattern):
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

class TransformerLM(nn.Module):
    """字符嵌入 + 可学习位置嵌入 + 多层因果自注意力 + 字表分类。

    使用 TransformerEncoderLayer 的自注意力和前馈网络，再加因果掩码，
    实现只依赖当前及之前字符的自回归语言模型；不需要交叉注意力。
    """

    def __init__(self, vocab_size, embed_dim, num_layers, nhead, ffn_dim,
                 dropout, max_seq_len):
        super().__init__()
        if nhead < 1 or embed_dim < 1 or embed_dim % nhead != 0:
            raise ValueError("embed_dim 必须为正数，且能被正整数 nhead 整除。")
        if num_layers < 1 or ffn_dim < 1 or max_seq_len < 1:
            raise ValueError("num_layers、ffn_dim 和 max_seq_len 必须为正整数。")
        self.max_seq_len = max_seq_len
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.position_embed = nn.Embedding(max_seq_len, embed_dim)
        self.drop = nn.Dropout(dropout)
        # 分别创建每层，使各层独立随机初始化。
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, vocab_size)
        # True 表示禁止关注；屏蔽严格上三角，避免看到未来字符（包括标签）。
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1),
            persistent=False,
        )

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.max_seq_len:
            raise ValueError(f"输入长度 {seq_len} 超过最大上下文长度 {self.max_seq_len}。")
        positions = torch.arange(seq_len, device=x.device)
        out = self.drop(self.embed(x) + self.position_embed(positions).unsqueeze(0))
        mask = self.causal_mask[:seq_len, :seq_len]
        for layer in self.layers:
            out = layer(out, src_mask=mask)
        return self.fc(self.norm(out))  # (batch_size, seq_len, vocab_size)


# ─────────────────────────── 训练 / 评估 ───────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if train:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser(description="字符级因果 Transformer 语言模型训练")
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--seq_len",    type=int,   default=64)
    parser.add_argument("--stride",     type=int,   default=None, help="切块间隔，默认等于 seq_len；1=密集重叠")
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--embed_dim",  type=int,   default=128)
    parser.add_argument("--nhead",      type=int,   default=4, help="Transformer 注意力头数，须整除 embed_dim")
    parser.add_argument("--ffn_dim",    type=int,   default=512, help="Transformer 前馈网络中间维度")
    parser.add_argument("--num_layers", type=int,   default=2)
    parser.add_argument("--dropout",    type=float, default=0.1)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--val_ratio",  type=float, default=0.05)
    parser.add_argument("--max_vocab",  type=int,   default=4000, help="字表最大字数，0=不限")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--corpus",     default="*.txt")
    parser.add_argument("--save",       default="best_transformer.pt")
    args = parser.parse_args()

    if args.stride is None:
        args.stride = args.seq_len
    if not 0 < args.val_ratio < 1:
        parser.error("val_ratio 必须在 0 和 1 之间。")
    if min(args.seq_len, args.stride, args.batch_size, args.epochs) < 1:
        parser.error("seq_len、stride、batch_size 和 epochs 必须为正整数。")
    if args.nhead < 1 or args.embed_dim < 1 or args.embed_dim % args.nhead != 0:
        parser.error("embed_dim 必须为正数，且能被正整数 nhead 整除。")
    if args.num_layers < 1 or args.ffn_dim < 1:
        parser.error("num_layers 和 ffn_dim 必须为正整数。")
    if not 0 <= args.dropout < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("dropout 必须在 [0, 1) 之间，lr 必须为有限正数。")
    if args.max_vocab < 0:
        parser.error("max_vocab 必须为非负整数，0 表示不限。")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  model: TRANSFORMER")

    # 数据准备
    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError("未找到任何 .txt 文件，请确认路径正确。")
    print(f"语料字符数: {len(text):,}")

    char2idx, idx2char = build_vocab(text, args.max_vocab or None)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    train_text = "\n".join(lines[:split])
    val_text   = "\n".join(lines[split:])

    train_ds = CharDataset(train_text, char2idx, args.seq_len, args.stride)
    val_ds   = CharDataset(val_text,   char2idx, args.seq_len, args.stride)

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(
            f"训练/验证样本不足（train={len(train_ds)}, val={len(val_ds)}）。"
            f"按行划分并过滤字表外字符后，两部分各需至少 {args.seq_len + 1} 个字符；"
            "请增加语料和有效行数、调整 val_ratio 或减小 seq_len。"
        )
    print(f"训练样本数: {len(train_ds):,}  验证样本数: {len(val_ds):,}")
    # 保留不足一个 batch 的样本，尤其避免小验证集被全部丢弃。
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, drop_last=False)

    # 模型
    model = TransformerLM(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        nhead=args.nhead,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        max_seq_len=args.seq_len,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    best_val_ppl = float("inf")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        marker = "  *" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            torch.save({
                "model_type": "transformer",
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": vars(args),
            }, args.save)

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}{marker}")

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")


if __name__ == "__main__":
    main()
