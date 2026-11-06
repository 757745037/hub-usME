# -*- coding: utf-8 -*-
"""
train_langid_lstm.py —— 多语种「语种识别」分类器（作业版：12 语种 · LSTM）

任务：一句话 → 它是什么语言（12 分类）
模型：Embedding → pack_padded_sequence → BiLSTM → 掩码 MaxPool → LayerNorm → Dropout → Linear
损失：CrossEntropyLoss      优化器：Adam
数据：语种识别实战/data/langid/v12（12 语种：课堂 8 个 + ko / th / tr / vi）
运行：CPU 单线程，约 1 分钟。

    python train_langid_lstm.py                          # 默认 12 类 + LSTM
    python train_langid_lstm.py --save langid_lstm_v12.pt # 存档给 eval_langid.py
    python train_langid_lstm.py --cell rnn               # 换成课堂的 RNN 做对照
    python train_langid_lstm.py --rebuild-vocab          # 从 train.csv 重建字符表

【和课堂 8 类 RNN 版相比，这份作业改了三个地方】
  1. 数据：v8 → v12，LABELS 从 8 个变 12 个，模型输出维度 = len(LABELS)
  2. 算法：nn.RNN → nn.LSTM（带输入门/遗忘门/输出门 + 细胞状态），
     加双向，并且用 pack_padded_sequence 让补齐位真正"不参与计算"
  3. 字符表：课堂那张 vocab_v8.txt 只认识 8 个语种里的字符，
     新语种（ko / th / vi / tr）的字符会全部落进 <UNK>，必须先重建词表

【为什么 LSTM 那句 pack_padded_sequence 是非改不可的】
  · 单向 RNN 时，补齐位只是"多读了一串 0"，掩码池化还能救回来；
  · 双向 LSTM 的**反向**分支是从序列末尾往前读的，
    如果不打包，它第一步读到的就是 [PAD]，真正有信息的字符被排到最后，
    "从右往左"这个归纳偏置直接废掉。pack 之后反向分支从真实末字符起步。
"""

import os
import csv
import time
import random
import argparse
import unicodedata
from collections import Counter, OrderedDict

import torch
import torch.nn as nn
from torch.nn.utils.rnn import (pack_padded_sequence,
                               pad_packed_sequence)
from torch.utils.data import Dataset, DataLoader

# ─── 超参数 ────────────────────────────────────────────────────────────
SEED = 42
MAX_LEN = 40          # 数据集本身长度窗口 8~40，统一截/补到 40
EMBED_DIM = 96
HIDDEN_DIM = 96       # LSTM 每个方向 96 维，双向拼接后 192
NUM_LAYERS = 1
DROPOUT = 0.3
BATCH_SIZE = 64
LR = 2e-3
EPOCHS = 10
NUM_THREADS = 1       # 小模型单线程比多线程快，别改成 16

HERE = os.path.dirname(os.path.abspath(__file__))
# 数据放在兄弟目录「语种识别实战」里，不复制一份，避免两份数据走样
DATA_DIR = os.path.join(os.path.dirname(HERE), "语种识别实战", "data", "langid", "v12")
VOCAB_PATH = os.path.join(HERE, "vocab_v12.txt")

# 12 个语种：模型的输出就是这 12 个格子（顺序固定，存档/评测都按它）
LABELS = ["ar", "en", "es", "hi", "ja", "ko",
          "pt", "ru", "th", "tr", "vi", "zh"]


# ─── 1. 文本归一化（必须和建数据时一致）────────────────────────────────
def normalize(text):
    """
    NFKC 归一化 + 转小写 + 只保留「字母(L)」与「组合符号(M)」。

    必须保留 M：印地语、泰语的元音是组合标记（कि 里的 ि、促成了 Thai 的 ่），
    只留字母会把印地语、泰语砍成裸辅音，这两个语种就再也分不开了。
    """
    t = unicodedata.normalize("NFKC", text).lower()
    return "".join(c for c in t if unicodedata.category(c)[0] in ("L", "M"))


# ─── 2. 字符表构建 ─────────────────────────────────────────────────────
def collect_char_freq(sentences):
    freq = Counter()
    for s in sentences:
        freq.update(normalize(s))
    return freq


def build_char_vocab_with_freq(sentences, min_freq=1):
    """
    从训练集统计字符表，返回 (vocab, freq)。

    第 0 号 <PAD>：补齐用（Embedding 里 padding_idx=0，向量恒为 0、不参与训练）
    第 1 号 <UNK>：没见过的字符（字符级下几乎用不到）
    其余按出现次数从高到低编号，次数相同按字符编码排序（保证可复现）

    ★ 作业的关键点 ★
    12 类里多了韩语谚文、泰文、越南语声调符号、土耳其语带点字母。
    课堂的 vocab_v8.txt 里根本没有这些字符 → 必须用本函数从 v12/train.csv
    重新统计，否则新语种整句都是 <UNK>，模型只能瞎猜。
    """
    freq = collect_char_freq(sentences)
    chars = sorted(((c, n) for c, n in freq.items() if n >= min_freq),
                   key=lambda kv: (-kv[1], kv[0]))
    vocab = OrderedDict([("<PAD>", 0), ("<UNK>", 1)])
    for ch, _ in chars:
        vocab[ch] = len(vocab)
    return vocab, freq


def build_char_vocab(sentences, min_freq=1):
    """只要表、不要频次时用这个（和课堂代码同名同义，方便对照）。"""
    return build_char_vocab_with_freq(sentences, min_freq)[0]


def write_vocab_file(path, vocab, freq):
    """写成和课堂 vocab_v8.txt 完全相同的格式：字符 \\t id \\t 出现次数。"""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("# 字符\tid\t出现次数\n")
        for ch, idx in vocab.items():
            f.write(f"{ch}\t{idx}\t{freq.get(ch, 0)}\n")


def load_vocab_file(path):
    """读字符表（每行「字符 \\t id \\t 出现次数」，# 开头是注释）。"""
    vocab = OrderedDict()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            ch, idx, _freq = line.rstrip("\n").split("\t")
            vocab[ch] = int(idx)
    return vocab


def encode(text, vocab, maxlen=MAX_LEN):
    ids = [vocab.get(ch, 1) for ch in normalize(text)][:maxlen]
    ids += [0] * (maxlen - len(ids))
    return ids


# ─── 3. 读数据 ─────────────────────────────────────────────────────────
def load_split(data_dir, name):
    path = os.path.join(data_dir, f"{name}.csv")
    if not os.path.exists(path):
        raise SystemExit(f"找不到数据文件：{path}\n  用 --data-dir 指到 v12 目录")
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["text"], r["label"]))
    return rows


class TextDataset(Dataset):
    def __init__(self, rows, vocab, label2id):
        self.X = [encode(t, vocab) for t, _ in rows]
        self.y = [label2id[l] for _, l in rows]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.tensor(self.X[i], dtype=torch.long), self.y[i]


# ─── 4. 模型 ───────────────────────────────────────────────────────────
class LangIDLSTM(nn.Module):
    """
    Embedding → (pack) → BiLSTM → 掩码 MaxPool → LayerNorm → Dropout → Linear

    三个和 RNN 版不同的地方：

    ① 门控结构
       RNN:  h_t = tanh(W_ih x_t + W_hh h_{t-1} + b)
       LSTM: i_t = σ(W_i x_t + U_i h_{t-1})      输入门 —— 这条新信息要不要写进去
             f_t = σ(W_f x_t + U_f h_{t-1})      遗忘门 —— 上一格细胞状态留多少
             g_t = tanh(W_g x_t + U_g h_{t-1})   候选值
             o_t = σ(W_o x_t + U_o h_{t-1})      输出门 —— 细胞状态暴露多少给 h
             c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t     细胞状态（一条几乎线性的传送带）
             h_t = o_t ⊙ tanh(c_t)
       细胞状态那条加法通路让梯度能跨很多步回传，所以 40 步的句子不怕梯度消失；
       RNN 每一步都乘一次 W_hh 再过 tanh，连乘几次就没了。

    ② 双向
       正向读一遍、反向读一遍，每步拼成 2*hidden 维。
       "这个字符后面跟着什么" 对语种判断很有用（比如 汉字的搭配 vs 拉丁字母的词尾）。

    ③ 打包（pack_padded_sequence）
       只把每条句子的真实长度送进 LSTM，补齐位不产生任何计算，
       反向分支才能从真实末字符开始读。见文件头的说明。
    """

    def __init__(self, vocab_size, num_classes,
                 embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM,
                 dropout=DROPOUT, num_layers=NUM_LAYERS,
                 bidirectional=True, cell="lstm", use_pack=True):
        super().__init__()
        self.cell = cell
        self.use_pack = use_pack
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        rnn_cls = nn.LSTM if cell == "lstm" else nn.RNN
        self.rnn = rnn_cls(embed_dim, hidden_dim,
                           num_layers=num_layers, batch_first=True,
                           bidirectional=bidirectional,
                           dropout=dropout if num_layers > 1 else 0.0)
        feat_dim = hidden_dim * (2 if bidirectional else 1)
        self.norm = nn.LayerNorm(feat_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        mask = x != 0                                  # (B, L)  补齐位为 False
        emb = self.embedding(x)                        # (B, L, E)

        if self.use_pack:
            lengths = mask.sum(dim=1).clamp(min=1).cpu()   # 每条句子的真实长度
            packed = pack_padded_sequence(emb, lengths,
                                          batch_first=True, enforce_sorted=False)
            out, _ = self.rnn(packed)                  # LSTM 返回 (out, (h, c))，RNN 返回 (out, h)
            h, _ = pad_packed_sequence(out, batch_first=True,
                                       total_length=x.size(1))   # (B, L, H*dirs)
        else:
            h, _ = self.rnn(emb)                       # (B, L, H*dirs)

        # 补齐位在池化里必须排除：LSTM 读过 [PAD] 也会积出非零状态
        h = h.masked_fill(~mask.unsqueeze(-1), -1e9)
        pooled = h.max(dim=1)[0]                       # (B, H*dirs)
        pooled = self.dropout(self.norm(pooled))
        return self.fc(pooled)                         # (B, C)  未过 softmax 的 logits


# ─── 5. 评测 ───────────────────────────────────────────────────────────
def predict(model, loader):
    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for X, y in loader:
            preds += model(X).argmax(dim=1).tolist()
            golds += y.tolist()
    return preds, golds


def macro_f1(preds, golds, num_classes):
    """每一类各自算 F1，再对类别取平均（少数类不会被大类淹掉）。"""
    f1s = []
    for c in range(num_classes):
        tp = sum(p == c and g == c for p, g in zip(preds, golds))
        fp = sum(p == c and g != c for p, g in zip(preds, golds))
        fn = sum(p != c and g == c for p, g in zip(preds, golds))
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s)


def confusion(preds, golds, labels):
    n = len(labels)
    mat = [[0] * n for _ in range(n)]
    for p, g in zip(preds, golds):
        mat[g][p] += 1
    return mat


def print_confusion(mat, labels, top=5):
    """只打印错的格子，按错误数排序 —— 12 类时看前十行就够了。"""
    pairs = [(mat[g][p], labels[g], labels[p])
             for g in range(len(labels)) for p in range(len(labels)) if g != p]
    pairs.sort(reverse=True)
    print("  最常混淆的组合（真实 → 预测）：")
    for cnt, g, p in pairs[:top]:
        if cnt == 0:
            break
        print(f"    {g} → {p}: {cnt} 条")


# ─── 6. 训练 ───────────────────────────────────────────────────────────
def prepare_vocab(args, train_rows):
    """字符表：优先读文件；文件不在或要求重建时，从训练集现算并落盘。"""
    if args.rebuild_vocab:
        sentences = [t for t, _ in train_rows]
        vocab, freq = build_char_vocab_with_freq(sentences)
        write_vocab_file(args.vocab, vocab, freq)
        print(f"  字符表已重建：{len(vocab)} 个字符 → {args.vocab}")
        return vocab
    if not os.path.exists(args.vocab):
        print(f"  字符表不存在（{args.vocab}），从训练集自动重建 ...")
        return prepare_vocab(argparse.Namespace(vocab=args.vocab, rebuild_vocab=True),
                             train_rows)
    return load_vocab_file(args.vocab)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--vocab", default=VOCAB_PATH, help="字符表文件（12 类专用）")
    ap.add_argument("--rebuild-vocab", action="store_true",
                    help="从 train.csv 重新统计字符表并覆盖 --vocab")
    ap.add_argument("--cell", default="lstm", choices=["lstm", "rnn"],
                    help="lstm=本作业；rnn=课堂算法，用来做对照实验")
    ap.add_argument("--no-pack", action="store_true",
                    help="关掉 pack_padded_sequence，现场看补齐位对双向 LSTM 的伤害")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--embed-dim", type=int, default=EMBED_DIM)
    ap.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--dropout", type=float, default=DROPOUT)
    ap.add_argument("--per-class", type=int, default=0,
                    help="只用每类前 N 条训练，用来演示「数据少了会过拟合」")
    ap.add_argument("--threads", type=int, default=NUM_THREADS)
    ap.add_argument("--save", default="", help="保存模型到该路径，供 eval_langid.py 使用")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    random.seed(SEED)
    torch.manual_seed(SEED)

    train_rows = load_split(args.data_dir, "train")
    dev_rows = load_split(args.data_dir, "dev")
    test_rows = load_split(args.data_dir, "test")

    # 数据里的标签必须和 LABELS 对得上，否则模型的输出格子根本不够用
    seen = sorted({l for _, l in train_rows})
    extra = [l for l in seen if l not in LABELS]
    missing = [l for l in LABELS if l not in seen]
    if extra or missing:
        raise SystemExit(
            f"标签和数据不匹配。\n"
            f"  数据里的标签：{' '.join(seen)}\n"
            f"  当前 LABELS（{len(LABELS)} 类）：{' '.join(LABELS)}\n"
            f"  多出来的：{' '.join(extra) or '无'}；缺的：{' '.join(missing) or '无'}\n"
            f"  模型输出维度 = len(LABELS)，改数据就要同步改 LABELS。"
        )
    label2id = {l: i for i, l in enumerate(LABELS)}
    num_classes = len(LABELS)

    if args.per_class:
        by_label = {}
        for t, l in train_rows:
            by_label.setdefault(l, []).append(t)
        rng = random.Random(SEED)
        train_rows = [(t, l) for l, ts in by_label.items()
                      for t in rng.sample(ts, min(args.per_class, len(ts)))]
        rng.shuffle(train_rows)

    vocab = prepare_vocab(args, train_rows)
    unknown = sum(1 for (t, _) in train_rows for ch in normalize(t) if ch not in vocab)
    print(f"数据目录：{args.data_dir}")
    print(f"  训练 {len(train_rows)} 条 / 验证 {len(dev_rows)} 条 / 测试 {len(test_rows)} 条")
    print(f"  类别 {num_classes} 个：{' '.join(LABELS)}")
    print(f"  字符表 {len(vocab)} 个字符（{os.path.basename(args.vocab)}，"
          f"训练集未登录字符 {unknown} 个）")
    print(f"  算法：{'BiLSTM' if args.cell == 'lstm' else 'BiRNN'}"
          f"{' + pack' if not args.no_pack else '（未打包）'}")
    print(f"  随机猜的准确率基准：{1/num_classes:.1%}")

    train_loader = DataLoader(TextDataset(train_rows, vocab, label2id),
                              batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(TextDataset(dev_rows, vocab, label2id), batch_size=128)
    test_loader = DataLoader(TextDataset(test_rows, vocab, label2id), batch_size=128)

    model = LangIDLSTM(len(vocab), num_classes,
                       embed_dim=args.embed_dim, hidden_dim=args.hidden_dim,
                       dropout=args.dropout, cell=args.cell,
                       use_pack=not args.no_pack)
    print(f"  模型参数量：{sum(p.numel() for p in model.parameters()):,}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    best_acc, best_state = -1.0, None
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for X, y in train_loader:
            logits = model(X)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # LSTM 也要防梯度爆炸
            optimizer.step()
            total += loss.item()
        scheduler.step()

        preds, golds = predict(model, dev_loader)
        acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
        f1 = macro_f1(preds, golds, num_classes)
        flag = ""
        if acc > best_acc:
            best_acc, flag = acc, "  ← best"
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"Epoch {epoch:2d}/{args.epochs}  loss={total/len(train_loader):.4f}  "
              f"dev_acc={acc:.4f}  dev_f1={f1:.4f}{flag}")
    elapsed = time.time() - t0
    print(f"\n训练耗时：{elapsed:.0f} 秒（{args.epochs} 个 epoch，{args.threads} 线程）")

    # 用验证集上最好的那一版做测试（不要拿测试集挑模型）
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"已回滚到验证集最优权重（dev_acc={best_acc:.4f}）")

    preds, golds = predict(model, test_loader)
    acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
    f1 = macro_f1(preds, golds, num_classes)
    print(f"\n测试集：accuracy = {acc:.4f}   macro-F1 = {f1:.4f}")
    print_confusion(confusion(preds, golds, LABELS), LABELS)

    # 逐类准确率 + 最容易被认错的去向
    print("\n  各语种准确率：")
    for c, lab in enumerate(LABELS):
        idx = [i for i, g in enumerate(golds) if g == c]
        hit = sum(preds[i] == c for i in idx)
        wrong = Counter(LABELS[preds[i]] for i in idx if preds[i] != c)
        hint = ("  ← 被认成：" + "×".join(f"{k}" for k, _ in wrong.most_common(2))
                if wrong else "")
        print(f"    {lab}: {hit}/{len(idx)} = {hit/max(1,len(idx)):.3f}{hint}")

    # 推理示例
    print("\n--- 推理示例 ---")
    model.eval()
    demos = [t for t, _ in test_rows[:5]]
    with torch.no_grad():
        for text in demos:
            logits = model(torch.tensor([encode(text, vocab)]))
            prob = torch.softmax(logits, dim=1)[0]
            pid = int(prob.argmax())
            print(f"  [{LABELS[pid]} {prob[pid]:.2f}]  {text[:50]}")

    if args.save:
        torch.save({"state_dict": model.state_dict(),
                    "vocab": dict(vocab),
                    "labels": LABELS,
                    "maxlen": MAX_LEN,
                    "embed_dim": args.embed_dim,
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                    "cell": args.cell,
                    "use_pack": not args.no_pack,
                    "bidirectional": True,
                    "test_acc": acc},
                   args.save)
        print(f"\n模型已保存：{args.save}")


if __name__ == "__main__":
    main()
