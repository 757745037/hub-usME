"""
字符级语言模型训练脚本，使用Transformer，含 PPL 计算。
Transformer 2层时运行结果：
device: cuda  model: TRANSFORMERS
语料字符数: 6,021,645
词表大小: 4000
模型参数量: 1,688,352

 Epoch  Train Loss   Train PPL    Val Loss     Val PPL      tok/s
--------------------------------------------------------
     1      6.7268      834.43      4.5864       98.14    939,482  *
     2      4.5726       96.79      4.3638       78.56    956,345  *
     3      4.4038       81.76      4.2442       69.70    925,546  *
     4      4.3037       73.97      4.1589       64.00    967,307  *
     5      4.2265       68.48      4.0768       58.96    960,429  *
     6      4.1542       63.70      3.9933       54.23    957,010  *
     7      4.0861       59.51      3.9218       50.49    912,917  *
     8      4.0288       56.19      3.8690       47.90    931,042  *
     9      3.9820       53.62      3.8252       45.84    935,861  *
    10      3.9425       51.54      3.7853       44.05    905,746  *
    11      3.9085       49.83      3.7556       42.76    868,873  *
    12      3.8793       48.39      3.7289       41.63    949,152  *
    13      3.8528       47.13      3.7022       40.54    925,469  *
    14      3.8300       46.06      3.6796       39.63    891,673  *
    15      3.8090       45.10      3.6603       38.87    964,915  *
    16      3.7901       44.26      3.6443       38.26    958,516  *
    17      3.7720       43.47      3.6270       37.60    969,459  *
    18      3.7564       42.80      3.6128       37.07    967,041  *
    19      3.7420       42.18      3.6018       36.66    971,098  *
    20      3.7273       41.57      3.5858       36.08    966,113  *

训练完成。最佳验证 PPL: 36.08  已保存至 Transformer_best_model.pt

Transformer 4层时运行结果：
device: cuda  model: TRANSFORMERS
语料字符数: 6,021,645
词表大小: 4000
模型参数量: 2,084,896

 Epoch  Train Loss   Train PPL    Val Loss     Val PPL      tok/s
--------------------------------------------------------
     1      6.4403      626.61      4.5704       96.58    623,822  *
     2      4.5207       91.90      4.2939       73.25    684,394  *
     3      4.3210       75.26      4.1276       62.03    684,498  *
     4      4.1898       66.01      4.0076       55.01    662,611  *
     5      4.0876       59.60      3.9148       50.14    652,997  *
     6      4.0030       54.76      3.8335       46.23    666,523  *
     7      3.9356       51.19      3.7733       43.52    650,201  *
     8      3.8800       48.43      3.7228       41.38    651,232  *
     9      3.8349       46.29      3.6855       39.86    661,353  *
    10      3.7960       44.52      3.6475       38.38    693,065  *
    11      3.7643       43.13      3.6202       37.34    683,361  *
    12      3.7353       41.90      3.5969       36.49    637,665  *
    13      3.7107       40.88      3.5728       35.62    679,725  *
    14      3.6881       39.97      3.5530       34.92    661,891  *
    15      3.6680       39.17      3.5340       34.26    676,798  *
    16      3.6501       38.48      3.5205       33.80    680,973  *
    17      3.6329       37.82      3.5030       33.21    681,660  *
    18      3.6177       37.25      3.4899       32.78    674,870  *
    19      3.6033       36.72      3.4825       32.54    661,584  *
    20      3.5898       36.23      3.4652       31.98    687,274  *

训练完成。最佳验证 PPL: 31.98  已保存至 Transformer_best_model.pt

Transformer 6层时运行结果：
device: cuda  model: TRANSFORMERS
语料字符数: 6,021,645
词表大小: 4000
模型参数量: 2,481,440

 Epoch  Train Loss   Train PPL    Val Loss     Val PPL      tok/s
--------------------------------------------------------
     1      6.2559      521.09      4.5494       94.57    504,812  *
     2      4.4706       87.41      4.2305       68.75    496,436  *
     3      4.2459       69.82      4.0444       57.08    512,396  *
     4      4.0972       60.17      3.9139       50.10    523,229  *
     5      3.9856       53.82      3.8175       45.49    522,429  *
     6      3.9002       49.41      3.7440       42.27    526,908  *
     7      3.8331       46.21      3.6836       39.79    527,172  *
     8      3.7792       43.78      3.6357       37.93    521,955  *
     9      3.7349       41.88      3.5980       36.52    521,381  *
    10      3.6968       40.32      3.5620       35.23    505,555  *
    11      3.6653       39.07      3.5357       34.32    513,882  *
    12      3.6370       37.98      3.5121       33.52    517,299  *
    13      3.6120       37.04      3.4885       32.74    518,150  *
    14      3.5902       36.24      3.4691       32.11    518,486  *
    15      3.5697       35.50      3.4517       31.56    515,994  *
    16      3.5514       34.86      3.4366       31.08    515,636  *
    17      3.5342       34.27      3.4212       30.61    507,032  *
    18      3.5189       33.75      3.4068       30.17    519,324  *
    19      3.5040       33.25      3.3999       29.96    484,832  *
    20      3.4906       32.81      3.3819       29.43    524,517  *

训练完成。最佳验证 PPL: 29.43  已保存至 Transformer_best_model.pt


生成语句效果：
============================================================
模型      : Transformer_best_model.pt（6 层, embed 128）
输入      : '今天天气'
续写 40 字 : '怎么样\n天天酷跑破解版\n什么东西是最便宜的\n男人可以送什么礼物？\n在淘宝上买东西'
------------------------------------------------------------
今天天气怎么样
天天酷跑破解版
什么东西是最便宜的
男人可以送什么礼物？
在淘宝上买东西
============================================================
"""

import os
import time
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

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout, nhead, dim_feedforward, max_len=512):
        super().__init__()
        assert embed_dim % nhead == 0, "d_model 必须能被 nhead 整除"
        self.embed = nn.Embedding(vocab_size, embed_dim)

        # ★ 零件1：位置编码
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.d_model = embed_dim
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,        # ★ 默认是 False，必须显式设 True
            norm_first=True,         # Pre-LN，训练更稳
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)        # (1, T)
        h = self.embed(x) * math.sqrt(self.d_model) + self.pos_embed(pos)
        h = self.drop(h)

        # ★ 零件2：因果掩码
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.encoder(h, mask=mask)
        return self.fc(h)    

# ─────────────────────────── 训练 / 评估 ───────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0

    # ★ GPU 异步执行，计时前必须同步，否则测到的是「下发命令的时间」
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()

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

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t_start

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl, total_tokens, elapsed


# ─────────────────────────── 生成后40个字 ───────────────────────────
def makeSentence(prompt,
                 ckpt_path="Transformer_best_model.pt",
                 gen_len=40,
                 ctx_len=64,
                 temperature=0.7,
                 top_k=20,
                 seed=None,
                 device=None):
    """
    输入一小段文字，让训练好的 Transformer 语言模型续写 gen_len 个字符并打印。

        prompt      : 开头文字，建议长度 < 20
        ckpt_path   : 训练时 --save 保存的权重文件
        gen_len     : 续写多少个字符
        ctx_len     : 每次喂给模型的上下文长度上限，不要超过训练时的 seq_len(64)
        temperature : 0.1 接近贪心（保守），0.7 适中，1.0+ 更随机
        top_k       : 每步只在概率最高的 k 个字符里采样（0 = 不限制）
        seed        : 随机种子，便于复现
        device      : None 则自动选 cuda
    """
    # ── 0. 设备与权重 ──
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    if not os.path.isfile(ckpt_path):
        raise SystemExit(
            "找不到模型权重：%s\n请先运行 main() 完成训练（会自动保存该文件）。"
            % os.path.abspath(ckpt_path)
        )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args     = ckpt["args"]
    char2idx = ckpt["char2idx"]
    idx2char = ckpt["idx2char"]

    # ── 1. 按 checkpoint 里的超参重建模型（不能用当前命令行参数，可能不一致）──
    model = TransformerLM(
        vocab_size=len(char2idx),
        embed_dim=args["embed_dim"],
        hidden_dim=args.get("hidden_dim", args["embed_dim"]),
        num_layers=args["num_layers"],
        dropout=args["dropout"],
        nhead=args["nhead"],
        dim_feedforward=args["dim_feedforward"],
    ).to(device)

    # strict=False：兼容旧权重里多余的 layer.* 死参数
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    # ── 2. 编码 prompt（本模型词表没有 <UNK>，词表外字符只能丢弃）──
    prompt_ids = [char2idx[c] for c in prompt if c in char2idx]
    dropped    = [c for c in prompt if c not in char2idx]
    if not prompt_ids:
        raise SystemExit("输入的字都不在词表里，无法生成：%r" % (dropped,))

    # ── 3. 自回归生成：只看最后一位的输出，采样一个字符，追加到末尾 ──
    if seed is not None:
        torch.manual_seed(seed)

    ids = list(prompt_ids)
    V = model.fc.out_features
    with torch.no_grad():
        for _ in range(gen_len):
            x = torch.tensor([ids[-ctx_len:]], dtype=torch.long, device=device)
            logits = model(x)[0, -1] / max(temperature, 1e-6)      # ★ 只取最后位置
            if top_k and top_k < V:
                kth = torch.topk(logits, top_k).values[-1]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            nxt = int(torch.multinomial(torch.softmax(logits, dim=-1), 1))
            ids.append(nxt)

    # ── 4. 解码 + 打印 ──
    generated = "".join(idx2char.get(i, "?") for i in ids[len(prompt_ids):])
    full = prompt + generated

    print("=" * 60)
    print("模型      : %s（%d 层, embed %d）"
          % (os.path.basename(ckpt_path), args["num_layers"], args["embed_dim"]))
    print("输入      : %r%s" % (prompt, ("   [忽略词表外: %s]" % (dropped,)) if dropped else ""))
    print("续写 %d 字 : %r" % (gen_len, generated))
    print("-" * 60)
    print(full)
    print("=" * 60)
    return generated, full


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="Transformers")
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--seq_len",    type=int,   default=64)
    parser.add_argument("--stride",     type=int,   default=64, help="切块间隔：1=密集重叠，seq_len=不重叠")
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--embed_dim",  type=int,   default=128)
    parser.add_argument("--hidden_dim", type=int,   default=256)
    parser.add_argument("--num_layers", type=int,   default=6)
    parser.add_argument("--dropout",    type=float, default=0.1)
    parser.add_argument("--lr",         type=float, default=5e-4)
    parser.add_argument("--val_ratio",  type=float, default=0.05)
    parser.add_argument("--max_vocab",  type=int,   default=4000, help="字表最大字数，0=不限")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--nhead",          type=int,   default=4, help="多头数")
    parser.add_argument("--dim_feedforward", type=int,  default=512, help="FFN层拓宽的宽度，输出仍是embed_dim的宽度")
    parser.add_argument("--corpus",     default="*.txt")
    parser.add_argument("--save",       default="Transformer_best_model.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  model: {args.model.upper()}")

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

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=True, drop_last=True)

    # 模型
    model = TransformerLM(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        nhead=args.nhead,
        dim_feedforward=args.dim_feedforward, 
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_ppl = float("inf")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}  {'tok/s':>9}")
    print("-" * 56)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl, tr_tok, tr_sec = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        with torch.no_grad():
            va_loss, va_ppl, va_tok, va_sec = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        tok_per_s = tr_tok / tr_sec
        marker = "  *" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": vars(args),
            }, args.save)

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}  {tok_per_s:>9,.0f}{marker}")

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")


if __name__ == "__main__":
    #训练
    #main()

    #测试生成效果
    makeSentence("今天天气")
