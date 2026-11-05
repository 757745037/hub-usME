# -*- coding: utf-8 -*-
"""

模型结构：
    Token Embedding + 可学习 Position Embedding
      → N × Transformer Decoder Block（因果自注意力 + FFN，残差 + LayerNorm）
      → Linear 输出词表 logits

【与课堂 RNN / LSTM 版的两处核心区别】
  1. 循环层 nn.LSTM / nn.RNN  →  因果自注意力（Causal Self-Attention）
     - 预测第 i 个字时只能看到 0..i-1（因果掩码把上三角屏蔽掉），不能偷看右边；
     - 序列内所有位置一步并行算完，不像 RNN 那样按时间步串行传递 hidden state。
  2. 位置信息不再靠 hidden state 串行携带，而是显式加上位置向量（课件 P21：
     注意力负责「看谁」，位置编码负责「在哪」）。

数据切分、词表构建、CrossEntropyLoss、Adam、PPL 评估、checkpoint 格式均与课堂版保持一致。

运行（CPU 即可）：
    python train_transformer_lm.py --epochs 10
    python train_transformer_lm.py --save transformer_lm.pt
    # 语料大、CPU 慢时，可用 --stride 64 做不重叠切块，减少样本数：
    python train_transformer_lm.py --stride 64 --epochs 10
"""

import math
import argparse
import glob
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────── 数据（与课堂版一致）───────────────────────────

def load_corpus(pattern="*.txt"):
    """读入匹配到的全部 .txt 语料，拼成一整段文本。"""
    texts = []
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)


def build_vocab(text, max_vocab=None):
    """
    构建字表：按出现频率取前 max_vocab 个字（None = 全部字符）。
    语料里的生僻字超出字表后会被 CharDataset 过滤掉。
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
        stride=1        —— 每个字符位置都当一个样本（样本大量重叠）
        stride=seq_len  —— 不重叠切块（标准做法，样本数 = 语料长度 / seq_len）
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


# ─────────────────────────── 模型：Transformer Decoder ──────────────────────

class TransformerBlock(nn.Module):
    """
    单个 Decoder Block（课件 P22 / P27 结构）：
        Multi-Head Causal Self-Attention → Add & LayerNorm
        → Position-wise FFN → Add & LayerNorm
    """

    def __init__(self, d_model, nhead, dim_ff, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, causal_mask):
        # 因果掩码：True 位置被屏蔽；位置 i 只能看到自己及左边
        attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask, need_weights=False)
        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.drop(self.ffn(x)))
        return x


class TransformerLM(nn.Module):
    """
    字符级语言模型：
        Embedding(Token + Position) → 堆叠 Decoder Block → LayerNorm → Linear 输出。
    """

    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_ff, max_len, dropout):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_ff, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)
        self.max_len = max_len

    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(T, device=x.device)
        h = self.tok_emb(x) + self.pos_emb(positions)
        # 因果掩码 (T, T)：上三角为 True（屏蔽），对角线及以下保留
        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
        )
        for block in self.blocks:
            h = block(h, causal_mask)
        h = self.norm(h)
        return self.fc(h)   # (B, T, V)


# ─────────────────────────── 训练 / 评估（与课堂版一致）─────────────────────

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
            optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)   # 困惑度：越低越好
    return avg_loss, ppl


def main():
    parser = argparse.ArgumentParser(description="字符级 Transformer Decoder 语言模型")
    parser.add_argument("--corpus", default="data/*.txt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=1,
                        help="切块间隔：1=密集重叠，seq_len=不重叠")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--max_vocab", type=int, default=4000, help="字表最大字数，0=不限")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", default="transformer_lm.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  model: Transformer Decoder")

    # 数据准备
    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError("未找到任何 .txt 语料，请确认 --corpus 路径。")
    print(f"语料字符数: {len(text):,}")

    char2idx, idx2char = build_vocab(text, args.max_vocab or None)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    train_text = "\n".join(lines[:split])
    val_text = "\n".join(lines[split:])

    train_ds = CharDataset(train_text, char2idx, args.seq_len, args.stride)
    val_ds = CharDataset(val_text, char2idx, args.seq_len, args.stride)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # 模型
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_ff=args.dim_ff,
        max_len=args.seq_len,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    print(f"  d_model={args.d_model}  nhead={args.nhead}  layers={args.num_layers}  "
          f"dim_ff={args.dim_ff}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_ppl = float("inf")
    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        marker = "  *" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": vars(args),
            }, args.save)

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  "
              f"{va_loss:>10.4f}  {va_ppl:>10.2f}{marker}")

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")


if __name__ == "__main__":
    main()
