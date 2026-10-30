# -*- coding: utf-8 -*-
"""
train_langid_v12.py — 作业：12 语种语种识别分类器

在课堂 8 语种脚本基础上扩展：
  1. LABELS 增加 ko / th / vi / tr（共 12 类）
  2. 从 v12/train.csv 用 build_char_vocab() 生成 vocab_v12.txt
  3. 数据目录改为 data/v12

模型结构与课堂一致：
  Embedding → RNN → 掩码 MaxPool → LayerNorm → Dropout → Linear

用法：
    python train_langid_v12.py
    python train_langid_v12.py --epochs 8 --save langid_rnn_v12.pt
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import csv
import time
import random
import argparse
import unicodedata
from collections import Counter, OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── 超参数 ───────────────────────────────────────────────────────────
SEED = 42
MAX_LEN = 40
EMBED_DIM = 64
HIDDEN_DIM = 64
DROPOUT = 0.3
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 8
NUM_THREADS = 1

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "v12")
VOCAB_PATH = os.path.join(HERE, "data", "vocab", "vocab_v12.txt")

# 作业 12 语种 = 课内 8 个 + ko/th/vi/tr（按字母序，与课堂风格一致）
LABELS = ["ar", "en", "es", "hi", "ja", "ko", "pt", "ru", "th", "tr", "vi", "zh"]


# ─── 1. 文本归一化（必须和建数据时一致）───────────────────────────────
def normalize(text):
    """NFKC + 小写 + 只保留字母(L)与组合符号(M)。"""
    t = unicodedata.normalize("NFKC", text).lower()
    return "".join(c for c in t if unicodedata.category(c)[0] in ("L", "M"))


# ─── 2. 字符表构建（作业核心：从 v12 训练集自己生成）──────────────────
def build_char_vocab(sentences, min_freq=1):
    """
    字符表 = 训练集里所有出现过的字符，每个字符发一个整数编号。
    第 0 号 <PAD>，第 1 号 <UNK>，其余按出现次数从高到低编号。
    """
    freq = Counter()
    for s in sentences:
        freq.update(normalize(s))
    chars = sorted(((c, n) for c, n in freq.items() if n >= min_freq),
                   key=lambda kv: (-kv[1], kv[0]))
    vocab = OrderedDict([("<PAD>", 0), ("<UNK>", 1)])
    for ch, _ in chars:
        vocab[ch] = len(vocab)
    return vocab, freq


def save_vocab(vocab, freq, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 字符\tid\t出现次数\n")
        for ch, idx in vocab.items():
            f.write(f"{ch}\t{idx}\t{freq.get(ch, 0)}\n")


def load_vocab_file(path):
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


# ─── 3. 读数据 ────────────────────────────────────────────────────────
def load_split(data_dir, name):
    path = os.path.join(data_dir, f"{name}.csv")
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


# ─── 4. 模型 ──────────────────────────────────────────────────────────
class LangIDClassifier(nn.Module):
    """Embedding → RNN → 掩码 MaxPool → LayerNorm → Dropout → Linear"""

    def __init__(self, vocab_size, num_classes,
                 embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1)
        h, _ = self.rnn(self.embedding(x))
        h = h.masked_fill(~mask, -1e9)
        pooled = h.max(dim=1)[0]
        pooled = self.dropout(self.norm(pooled))
        return self.fc(pooled)


# ─── 5. 评估 ──────────────────────────────────────────────────────────
def predict(model, loader):
    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for X, y in loader:
            preds += model(X).argmax(dim=1).tolist()
            golds += y.tolist()
    return preds, golds


def macro_f1(preds, golds, num_classes):
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
    return sum(f1s) / len(f1s) if f1s else 0.0


def confusion(preds, golds, labels):
    n = len(labels)
    mat = [[0] * n for _ in range(n)]
    for p, g in zip(preds, golds):
        mat[g][p] += 1
    return mat


def print_confusion(mat, labels, top=5):
    pairs = [(mat[g][p], labels[g], labels[p])
             for g in range(len(labels)) for p in range(len(labels)) if g != p]
    pairs.sort(reverse=True)
    print("  最常混淆的组合（真实 → 预测）：")
    for cnt, g, p in pairs[:top]:
        if cnt == 0:
            break
        print(f"    {g} → {p}: {cnt} 条")


def plot_curve(history, path):
    epochs = list(range(1, len(history["loss"]) + 1))
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(epochs, history["loss"], "b-o", label="train loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax2 = ax1.twinx()
    ax2.plot(epochs, history["dev_acc"], "r-s", label="dev acc")
    ax2.set_ylabel("dev accuracy", color="r")
    ax2.tick_params(axis="y", labelcolor="r")
    ax2.set_ylim(0, 1.05)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="lower right")
    ax1.set_title("12-lang Language ID Training Curve")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"训练曲线已保存：{path}")


# ─── 6. 训练 ──────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--vocab", default=VOCAB_PATH,
                    help="字符表路径；不存在时从训练集自动生成")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--threads", type=int, default=NUM_THREADS)
    ap.add_argument("--save", default=os.path.join(HERE, "langid_rnn_v12.pt"))
    ap.add_argument("--curve", default=os.path.join(HERE, "train_curve.png"))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    random.seed(SEED)
    torch.manual_seed(SEED)

    train_rows = load_split(args.data_dir, "train")
    dev_rows = load_split(args.data_dir, "dev")
    test_rows = load_split(args.data_dir, "test")

    seen = sorted({l for _, l in train_rows})
    extra = [l for l in seen if l not in LABELS]
    missing = [l for l in LABELS if l not in seen]
    if extra or missing:
        raise SystemExit(
            f"标签不匹配。数据有而 LABELS 无：{extra}；LABELS 有而数据无：{missing}\n"
            f"  当前 LABELS（{len(LABELS)} 类）：{' '.join(LABELS)}"
        )
    label2id = {l: i for i, l in enumerate(LABELS)}
    num_classes = len(LABELS)

    # 作业要求：从 v12/train.csv 生成字符表（课件不提供 vocab_v12.txt）
    if os.path.exists(args.vocab):
        vocab = load_vocab_file(args.vocab)
        print(f"已加载字符表：{args.vocab}（{len(vocab)} 个字符）")
    else:
        print("正在从训练集构建字符表……")
        vocab, freq = build_char_vocab([t for t, _ in train_rows])
        save_vocab(vocab, freq, args.vocab)
        print(f"字符表已生成：{args.vocab}（{len(vocab)} 个字符，含 <PAD>/<UNK>）")

    print(f"数据目录：{args.data_dir}")
    print(f"  训练 {len(train_rows)} 条 / 验证 {len(dev_rows)} 条 / 测试 {len(test_rows)} 条")
    print(f"  类别 {num_classes} 个：{' '.join(LABELS)}")
    print(f"  随机猜的准确率基准：{1 / num_classes:.1%}")

    train_loader = DataLoader(TextDataset(train_rows, vocab, label2id),
                              batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(TextDataset(dev_rows, vocab, label2id), batch_size=128)
    test_loader = DataLoader(TextDataset(test_rows, vocab, label2id), batch_size=128)

    model = LangIDClassifier(len(vocab), num_classes)
    print(f"  模型参数量：{sum(p.numel() for p in model.parameters()):,}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = {"loss": [], "dev_acc": []}
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for X, y in train_loader:
            logits = model(X)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        preds, golds = predict(model, dev_loader)
        acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
        avg_loss = total / len(train_loader)
        history["loss"].append(avg_loss)
        history["dev_acc"].append(acc)
        print(f"Epoch {epoch:2d}/{args.epochs}  loss={avg_loss:.4f}  "
              f"dev_acc={acc:.4f}")
    elapsed = time.time() - t0
    print(f"\n训练耗时：{elapsed:.0f} 秒（{args.epochs} 个 epoch，{args.threads} 线程）")

    preds, golds = predict(model, test_loader)
    acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
    f1 = macro_f1(preds, golds, num_classes)
    print(f"\n测试集：accuracy = {acc:.4f}   macro-F1 = {f1:.4f}")
    print_confusion(confusion(preds, golds, LABELS), LABELS)

    print("\n  各语种准确率：")
    for c, lab in enumerate(LABELS):
        idx = [i for i, g in enumerate(golds) if g == c]
        hit = sum(preds[i] == c for i in idx)
        print(f"    {lab}: {hit}/{len(idx)} = {hit / max(1, len(idx)):.3f}")

    print("\n--- 推理示例 ---")
    model.eval()
    demos = [t for t, _ in test_rows[:6]]
    with torch.no_grad():
        for text in demos:
            logits = model(torch.tensor([encode(text, vocab)]))
            prob = torch.softmax(logits, dim=1)[0]
            pid = int(prob.argmax())
            snippet = text[:50].encode("utf-8", "replace").decode("utf-8")
            try:
                print(f"  [{LABELS[pid]} {prob[pid]:.2f}]  {snippet}")
            except UnicodeEncodeError:
                print(f"  [{LABELS[pid]} {prob[pid]:.2f}]  <non-printable text>")

    plot_curve(history, args.curve)

    if args.save:
        torch.save({"state_dict": model.state_dict(), "vocab": dict(vocab),
                    "labels": LABELS, "maxlen": MAX_LEN,
                    "embed_dim": EMBED_DIM, "hidden_dim": HIDDEN_DIM,
                    "dropout": DROPOUT,
                    "test_acc": acc, "test_macro_f1": f1}, args.save)
        print(f"\n模型已保存：{args.save}")


if __name__ == "__main__":
    main()
