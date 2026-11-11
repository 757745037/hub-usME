"""
字符级语言模型训练脚本，使用Transformer+SFT，含 PPL 计算。

SFT训练输出：
device: cuda
词表 4000 -> 4109（补入 107 个 OOV 字符）
参数量 1,914,637
SFT 训练 3000 条 / 验证 200 条, seq_len=128

 Epoch      Train  TrainPPL        Val    ValPPL
--------------------------------------------------
     1     4.6937    109.26     4.0091     55.10  *
     2     4.0671     58.38     3.7736     43.54  *
     3     3.8559     47.27     3.6382     38.02  *
     4     3.7133     40.99     3.5503     34.82  *
     5     3.6114     37.02     3.4793     32.44  *
     6     3.5229     33.88     3.4261     30.76  *
     7     3.4478     31.43     3.3792     29.35  *
     8     3.3782     29.32     3.3395     28.20  *
     9     3.3129     27.46     3.3067     27.29  *
    10     3.2523     25.85     3.2795     26.56  *
    11     3.2049     24.65     3.2531     25.87  *
    12     3.1532     23.41     3.2283     25.24  *
    13     3.1073     22.36     3.2132     24.86  *
    14     3.0776     21.71     3.1998     24.53  *
    15     3.0299     20.69     3.1811     24.07  *
    16     2.9951     19.99     3.1739     23.90  *
    17     2.9593     19.28     3.1574     23.51  *
    18     2.9291     18.71     3.1421     23.15  *
    19     2.8989     18.15     3.1363     23.02  *
    20     2.8669     17.58     3.1314     22.91  *
    21     2.8375     17.07     3.1189     22.62  *
    22     2.8094     16.60     3.1173     22.59  *
    23     2.7870     16.23     3.1076     22.37  *
    24     2.7593     15.79     3.1084     22.38
    25     2.7373     15.45     3.1006     22.21  *
    26     2.7104     15.04     3.0967     22.13  *
    27     2.6881     14.70     3.0963     22.11  *
    28     2.6656     14.38     3.0948     22.08  *
    29     2.6475     14.12     3.0887     21.95  *
    30     2.6307     13.88     3.0899     21.97
    31     2.6096     13.59     3.0775     21.71  *
    32     2.5881     13.30     3.0823     21.81
    33     2.5703     13.07     3.0811     21.78
    34     2.5540     12.86     3.0806     21.77
    35     2.5343     12.61     3.0794     21.75
    36     2.5175     12.40     3.0767     21.69  *
    37     2.5051     12.25     3.0795     21.75
    38     2.4846     12.00     3.0815     21.79
    39     2.4720     11.85     3.0806     21.77
    40     2.4589     11.69     3.0789     21.73

SFT 完成。最佳 val PPL: 21.69  已保存至 Transformer_SFT_best_model.pt

问答输出：
问：今天天气怎么样？
答：今天气晴天气晴天气晴朗，气温20℃，气温2℃，湿度20℃，最低温度较低气温度，降雨概率20度：今天气温度，风，气温度2°2℃摄氏温度，天降雨。
问：1+1=?
答：这两个数的整数是5。
问：这附近哪有便利店？
答：1. 北京的驾车辆车辆车司，2. 车辆车辆车辆车
"""

import os
import time
import math
import argparse
import glob
import random
import torch
import torch.nn as nn
import json
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


# ═════════════════════  SFT 环节  ═════════════════════

SFT_TMPL = "问：{instruction}\n答："     # 模板字符都在词表里（问/答/：/\n 都有）
IGNORE   = -100                          # CrossEntropyLoss 忽略用的值


class SFTDataset(Dataset):
    """
    把 (instruction, output) 套成模板，**只对回答部分算 loss**。

    为什么只算回答：
      指令是"已经给定的条件"，模型不需要学怎么生成它；
      把梯度集中在回答上，同样的训练量效果更好。这是 SFT 的标准做法。

    为什么右侧 padding 不需要额外的 padding mask：
      Transformer 用的是【因果掩码】，位置 i 只能看 0..i。
      padding 都排在内容右侧，所以任何真实内容位置都看不到 padding。
      padding 位置自己的输出被 label=-100 忽略掉，完全无害。
    """
    def __init__(self, path, char2idx, seq_len, pad_id, eos_id):
        self.seq_len = seq_len
        self.pad_id  = pad_id
        self.eos_id = eos_id
        self.c2i     = char2idx
        with open(path, encoding="utf-8") as f:
            self.rows = [json.loads(l) for l in f if l.strip()]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        p_ids = [self.c2i[c] for c in SFT_TMPL.format(instruction=r["instruction"])
                 if c in self.c2i]
        a_ids = [self.c2i[c] for c in r["output"] if c in self.c2i]

        ids = (p_ids + a_ids + [self.eos_id])[: self.seq_len + 1]     # ★ 末尾加 EOS； +1 因为要错开一位
        x, y = ids[:-1], ids[1:]                        # 输入 / 目标

        # 只保留"预测回答"的那些位置：把前 len(p_ids)-1 个目标屏蔽掉
        n_mask = min(max(len(p_ids) - 1, 0), len(y))
        y = [IGNORE] * n_mask + y[n_mask:]

        pad = self.seq_len - len(x)                     # 右侧补齐
        return (torch.tensor(x + [self.pad_id] * pad, dtype=torch.long),
                torch.tensor(y + [IGNORE]   * pad, dtype=torch.long))


def load_sft_texts(paths):
    """把 jsonl 拼成纯文本，用于补词表。"""
    out = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    o = json.loads(line)
                    out.append(SFT_TMPL.format(instruction=o["instruction"]) + o["output"])
    return out


def expand_vocab_with_sft(char2idx, idx2char, texts):
    """
    把 SFT 里出现、但预训练词表没有的字符【追加到末尾】。

    关键：只往末尾追加，原有 id 一个都不动 —— 这样预训练的权重可以直接复用。
    （你的 SFT 数据里 ' ' 空格就不在词表里，出现 8126 次，不能丢。）
    """
    from collections import Counter
    cnt = Counter()
    for t in texts:
        cnt.update(t)
    added = [c for c, _ in cnt.most_common() if c not in char2idx]
    for c in added:
        char2idx[c] = len(char2idx)
        idx2char[char2idx[c]] = c
    return char2idx, idx2char, added


def load_state_expanded(model, old_sd, old_vocab, new_vocab, std=0.02, seed=0):
    """
    vocab 变大后，embedding 和 fc 要多出若干行。
    做法：老权重复制到前 old_vocab 行，新增行随机初始化（SFT 会训练它们）。
    """
    g = torch.Generator().manual_seed(seed)
    new_sd = {}
    for k, v in old_sd.items():
        if k in ("embed.weight", "fc.weight"):
            w = torch.empty(new_vocab, v.size(1))       # shape: (V, D)
            w[:old_vocab] = v
            nn.init.normal_(w[old_vocab:], std=std, generator=g)
            new_sd[k] = w
        elif k == "fc.bias":
            b = torch.empty(new_vocab)
            b[:old_vocab] = v
            b[old_vocab:] = 0.0
            new_sd[k] = b
        else:
            new_sd[k] = v
    model.load_state_dict(new_sd, strict=False)
    return model


def sft_main():
    """在预训练权重基础上做 SFT。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--init",        default="Transformer_best_model.pt",
                        help="预训练权重，SFT 从它继续")
    parser.add_argument("--train_jsonl", default="sft_train.jsonl")
    parser.add_argument("--val_jsonl",   default="sft_val.jsonl")
    parser.add_argument("--save",        default="Transformer_SFT_best_model.pt")
    parser.add_argument("--epochs",      type=int,   default=40)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=1e-4,
                        help="SFT 学习率要比预训练小（预训练 5e-4）")
    parser.add_argument("--sft_seq_len", type=int,   default=128,
                        help="模板拼接后 p90≈90, max≈116，需要 >=128")
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # ── 1. 读预训练权重 ──
    ck = torch.load(args.init, map_location="cpu", weights_only=False)
    char2idx, idx2char = dict(ck["char2idx"]), dict(ck["idx2char"])
    pa = ck["args"]
    old_vocab = len(char2idx)

    # ── 2. 用 SFT 语料补词表（空格等 OOV）──
    texts = load_sft_texts([args.train_jsonl, args.val_jsonl])
    char2idx, idx2char, added = expand_vocab_with_sft(char2idx, idx2char, texts)
    PAD, EOS = "<PAD>", "<EOS>"
    PAD_ID = len(char2idx); char2idx[PAD] = PAD_ID; idx2char[PAD_ID] = PAD
    EOS_ID = len(char2idx); char2idx[EOS] = EOS_ID; idx2char[EOS_ID] = EOS

    new_vocab = len(char2idx)
    print(f"词表 {old_vocab} -> {new_vocab}（补入 {len(added)} 个 OOV 字符）")

    # ── 3. 建模型 + 迁移权重 ──
    model = TransformerLM(
        vocab_size=new_vocab,
        embed_dim=pa["embed_dim"],
        hidden_dim=pa.get("hidden_dim", pa["embed_dim"]),
        num_layers=pa["num_layers"],
        dropout=pa["dropout"],
        nhead=pa["nhead"],
        dim_feedforward=pa["dim_feedforward"],
    ).to(device)
    model = load_state_expanded(model, ck["model_state"], old_vocab, new_vocab)
    print(f"参数量 {sum(p.numel() for p in model.parameters()):,}")

    # ── 4. 数据 ──
    tr_ds = SFTDataset(args.train_jsonl, char2idx, args.sft_seq_len, PAD_ID, EOS_ID)
    va_ds = SFTDataset(args.val_jsonl,   char2idx, args.sft_seq_len, PAD_ID, EOS_ID)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size)
    print(f"SFT 训练 {len(tr_ds)} 条 / 验证 {len(va_ds)} 条, seq_len={args.sft_seq_len}")

    # ── 5. loss 必须忽略 -100 ──
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best = float("inf")
    print(f"\n{'Epoch':>6}  {'Train':>9} {'TrainPPL':>9}  {'Val':>9} {'ValPPL':>9}")
    print("-" * 50)
    for ep in range(1, args.epochs + 1):
        model.train()
        tl, tc = 0.0, 0
        for x, y in tr_dl:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x).reshape(-1, new_vocab), y.reshape(-1))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            n = int((y != IGNORE).sum()); tl += loss.item() * n; tc += n
        tr = tl / max(tc, 1)

        model.eval()
        vl, vc = 0.0, 0
        with torch.no_grad():
            for x, y in va_dl:
                x, y = x.to(device), y.to(device)
                loss = criterion(model(x).reshape(-1, new_vocab), y.reshape(-1))
                n = int((y != IGNORE).sum()); vl += loss.item() * n; vc += n
        va = vl / max(vc, 1)

        mark = ""
        if va < best:
            best = va; mark = "  *"
            torch.save({"model_state": model.state_dict(), "char2idx": char2idx,
                        "idx2char": idx2char,
                        "args": {**pa, "seq_len": args.sft_seq_len, "sft": True}}, args.save)
        print(f"{ep:>6}  {tr:>9.4f} {math.exp(tr):>9.2f}  {va:>9.4f} {math.exp(va):>9.2f}{mark}")

    print(f"\nSFT 完成。最佳 val PPL: {math.exp(best):.2f}  已保存至 {args.save}")


@torch.no_grad()
def ask(question, ckpt_path="Transformer_SFT_best_model.pt",
        gen_len=128, temperature=0.5, top_k=10, seed=0, device=None):
    """按 SFT 模板提问，只返回回答部分。"""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if not os.path.isfile(ckpt_path):
        raise SystemExit(f"找不到 SFT 权重：{os.path.abspath(ckpt_path)}\n请先运行 sft_main()")

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    char2idx, idx2char, pa = ck["char2idx"], ck["idx2char"], ck["args"]
    SEQ = pa["seq_len"]
    model = TransformerLM(len(char2idx), pa["embed_dim"],
                          pa.get("hidden_dim", pa["embed_dim"]), pa["num_layers"],
                          pa["dropout"], pa["nhead"], pa["dim_feedforward"]).to(device)
    model.load_state_dict(ck["model_state"], strict=False)
    model.eval()
    torch.manual_seed(seed)

    ids = [char2idx[c] for c in SFT_TMPL.format(instruction=question) if c in char2idx]
    start = len(ids)
    for _ in range(gen_len):
        logits = model(torch.tensor([ids[-SEQ:]], dtype=torch.long, device=device))[0, -1]
        logits = logits / max(temperature, 1e-6)
        if top_k and top_k < logits.numel():
            logits = logits.masked_fill(logits < torch.topk(logits, top_k).values[-1],
                                        float("-inf"))
            
        nxt = int(torch.multinomial(torch.softmax(logits, -1), 1))
        if nxt == char2idx["<EOS>"]:                          # ★★ 自己说完了
            break
        ids.append(nxt)

    ans = "".join(idx2char.get(i, "?") for i in ids[start:])
    print(f"问：{question}")
    print(f"答：{ans}")
    return ans

# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="Transformers")
    parser.add_argument("--epochs",     type=int,   default=15)
    parser.add_argument("--seq_len",    type=int,   default=64)
    parser.add_argument("--stride",     type=int,   default=64, help="切块间隔：1=密集重叠，seq_len=不重叠")
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--embed_dim",  type=int,   default=128)
    parser.add_argument("--hidden_dim", type=int,   default=256)
    parser.add_argument("--num_layers", type=int,   default=4)
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
    # main()                            # ① 预训练（已做过，权重在 Transformer_best_model.pt）
    # sft_main()                        # ② SFT 微调
    # makeSentence("今天天气")           # ③ 预训练模型的续写
    ask("今天天气怎么样？")   # ④ SFT 模型的问答
    ask("1+1=?")
    ask("这附近哪有便利店？")
