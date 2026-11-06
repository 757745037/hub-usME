# -*- coding: utf-8 -*-
"""
train_langid_cls_v12.py —— 多语种「语种识别」分类器（12 语种版）

任务：一句话 → 它是什么语言（12 分类）
模型：Embedding → RNN → 掩码 MaxPool → LayerNorm → Dropout → Linear → Softmax
损失：CrossEntropyLoss      优化器：Adam
数据：data/langid/v12（每类 800 条训练、100 条验证、150 条测试）
运行：CPU 单线程，默认训练 8 轮，保存验证集准确率最高的一轮。

    python train_langid_cls_v12.py
    python eval_langid_v12.py

数据来源：Tatoeba（https://tatoeba.org）逐语种句子导出。
数据许可：CC-BY 2.0 FR（https://creativecommons.org/licenses/by/2.0/fr/）。
"""

import os
import csv
import time
import random
import argparse
import copy
import unicodedata
from collections import Counter, OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── 超参数 ────────────────────────────────────────────────────────────
SEED = 42
MAX_LEN = 40          # 与数据集的 8~40 字符长度窗口一致
EMBED_DIM = 64
HIDDEN_DIM = 64
DROPOUT = 0.3
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 8
NUM_THREADS = 1       # 小模型默认使用 CPU 单线程

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "langid", "v12")
VOCAB_PATH = os.path.join(HERE, "data", "langid", "vocab", "vocab_v12.txt")
MODEL_PATH = os.path.join(HERE, "langid_rnn_v12.pt")

# 阿拉伯语、英语、西班牙语、印地语、日语、韩语、葡萄牙语、俄语、泰语、土耳其语、越南语、中文
LABELS = ["ar", "en", "es", "hi", "ja", "ko", "pt", "ru", "th", "tr", "vi", "zh"]



# ─── 1. 文本归一化（必须和建数据时一致）────────────────────────────────
def normalize(text):
    """
    NFKC 归一化 + 转小写 + 只保留「字母(L)」与「组合符号(M)」。

    为什么保留 M：印地语、泰语的元音是组合标记（कि 里的 ि），
    只留字母会把印地语砍成裸辅音，模型就没法区分。
    """
    t = unicodedata.normalize("NFKC", text).lower()
    return "".join(c for c in t if unicodedata.category(c)[0] in ("L", "M"))


# ─── 2. 字符表构建 ─────────────────────────────────────────────────────
def build_char_vocab(sentences, min_freq=1):
    """
    字符表 = 训练集里所有出现过的字符，每个字符发一个整数编号。

    第 0 号 <PAD>：补齐用（Embedding 里 padding_idx=0，向量恒为 0、不参与训练）
    第 1 号 <UNK>：没见过的字符（字符级下几乎用不到）
    其余按出现次数从高到低编号，次数相同按字符编码排序（保证可复现）

    多语种的关键点：这张表不需要按语种分开建。汉字、拉丁字母、西里尔字母、
    阿拉伯字母、天城文、泰文全都排在同一张表里 —— 这就是"字符级天然多语种"。

    主流程只用训练集调用它，保证新增语种的字符有自己的编号，
    同时避免验证集、测试集的信息泄漏到字符表中。
    """
    freq = Counter()
    for s in sentences:
        freq.update(normalize(s))
    chars = sorted(((c, n) for c, n in freq.items() if n >= min_freq),
                   key=lambda kv: (-kv[1], kv[0]))
    vocab = OrderedDict([("<PAD>", 0), ("<UNK>", 1)])
    for ch, _ in chars:
        vocab[ch] = len(vocab)
    return vocab


def save_vocab_file(vocab, sentences, path):
    """按课堂文件格式保存字符、编号与训练集频次。"""
    freq = Counter(ch for text in sentences for ch in normalize(text))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 字符\tid\t出现次数\n")
        for ch, idx in vocab.items():
            f.write(f"{ch}\t{idx}\t{freq.get(ch, 0)}\n")


def load_vocab_file(path):
    """
    读已经生成好的字符表（每行「字符 \\t id \\t 出现次数」，以 # 开头的是注释）。

    通过 --vocab 显式复用字符表；默认每次从训练集重新构建。
    """
    vocab = OrderedDict()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            ch, idx, _freq = line.rstrip("\n").split("\t")
            vocab[ch] = int(idx)
    if (vocab.get("<PAD>") != 0 or vocab.get("<UNK>") != 1
            or sorted(vocab.values()) != list(range(len(vocab)))):
        raise ValueError("字符表必须包含 <PAD>=0、<UNK>=1，且编号连续不重复")
    return vocab


def encode(text, vocab, maxlen=MAX_LEN):
    ids = [vocab.get(ch, 1) for ch in normalize(text)][:maxlen]
    ids += [0] * (maxlen - len(ids))
    return ids


# ─── 3. 读数据 ─────────────────────────────────────────────────────────
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


# ─── 4. 模型 ───────────────────────────────────────────────────────────
class LangIDClassifier(nn.Module):
    """
    Embedding → RNN → 掩码 MaxPool → LayerNorm → Dropout → Linear

    MaxPool 前面那两行 mask 是为了让补齐位不参与池化：
    补齐位喂进去的是全 0 向量，RNN 照样会往前走，池化时就会把
    "读过一串 0" 的状态也算进来。masked_fill 成 -inf 后取最大值就干净了。
    """

    def __init__(self, vocab_size, num_classes,
                 embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1)              # (B, L, 1)  补齐位为 False
        h, _ = self.rnn(self.embedding(x))         # (B, L, H)
        h = h.masked_fill(~mask, -1e9)
        pooled = h.max(dim=1)[0]                   # (B, H)
        pooled = self.dropout(self.norm(pooled))
        return self.fc(pooled)                     # (B, C)  未过 softmax 的 logits


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


def print_confusion(mat, labels, top=3):
    """只打印错的格子，按错误数排序。"""
    pairs = [(mat[g][p], labels[g], labels[p])
             for g in range(len(labels)) for p in range(len(labels)) if g != p]
    pairs.sort(reverse=True)
    print("  最常混淆的组合（真实 → 预测）：")
    for cnt, g, p in pairs[:top]:
        if cnt == 0:
            break
        print(f"    {g} → {p}: {cnt} 条")


# ─── 6. 训练 ───────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="训练 12 类字符级语种识别模型")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--vocab", default="",
                    help="复用已有字符表；默认从训练集重新构建")
    ap.add_argument("--save-vocab", default=VOCAB_PATH,
                    help="新建字符表的保存路径")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--per-class", type=int, default=0,
                    help="每类随机取 N 条训练，0 表示使用全部训练数据")
    ap.add_argument("--threads", type=int, default=NUM_THREADS)
    ap.add_argument("--save", default=MODEL_PATH, help="保存最佳模型，供 eval_langid_v12.py 使用")
    args = ap.parse_args()
    if args.epochs <= 0 or args.threads <= 0 or args.per_class < 0:
        ap.error("epochs、threads 必须为正整数，per-class 不能为负数")

    torch.set_num_threads(args.threads)
    random.seed(SEED)
    torch.manual_seed(SEED)

    train_rows = load_split(args.data_dir, "train")
    dev_rows = load_split(args.data_dir, "dev")
    test_rows = load_split(args.data_dir, "test")

    # 三个集合都必须覆盖这 12 类，防止少训练某类或评测时遇到未知标签。
    for name, rows in (("train", train_rows), ("dev", dev_rows), ("test", test_rows)):
        seen = {label for _, label in rows}
        missing = sorted(set(LABELS) - seen)
        extra = sorted(seen - set(LABELS))
        if missing or extra:
            ap.error(f"{name} 标签不匹配：缺少 {missing}，额外 {extra}")
        if any(not normalize(text) for text, _ in rows):
            ap.error(f"{name} 存在归一化后为空的文本")
    label2id = {l: i for i, l in enumerate(LABELS)}
    num_classes = len(LABELS)

    if args.per_class:
        by_label = {}
        for t, l in train_rows:
            by_label.setdefault(l, []).append(t)
        train_rows = [(t, l) for l, ts in by_label.items()
                      for t in random.Random(SEED).sample(ts, min(args.per_class, len(ts)))]
        random.Random(SEED).shuffle(train_rows)

    if args.vocab:
        vocab = load_vocab_file(args.vocab)
        vocab_source = args.vocab
        missing_chars = {ch for text, _ in train_rows for ch in normalize(text)} - set(vocab)
        if missing_chars:
            ap.error(f"字符表缺少 {len(missing_chars)} 个训练字符，请去掉 --vocab 重新生成")
    else:
        vocab = build_char_vocab(text for text, _ in train_rows)
        save_vocab_file(vocab, (text for text, _ in train_rows), args.save_vocab)
        vocab_source = args.save_vocab
    print(f"数据目录：{args.data_dir}")
    print(f"  训练 {len(train_rows)} 条 / 验证 {len(dev_rows)} 条 / 测试 {len(test_rows)} 条")
    print(f"  类别 {num_classes} 个：{' '.join(LABELS)}")
    print(f"  字符表 {len(vocab)} 项（含 PAD/UNK，来自 {vocab_source}）")
    print(f"  随机猜的准确率基准：{1/num_classes:.1%}")

    train_loader = DataLoader(TextDataset(train_rows, vocab, label2id),
                              batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(TextDataset(dev_rows, vocab, label2id), batch_size=128)
    test_loader = DataLoader(TextDataset(test_rows, vocab, label2id), batch_size=128)

    model = LangIDClassifier(len(vocab), num_classes)
    print(f"  模型参数量：{sum(p.numel() for p in model.parameters()):,}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    t0 = time.time()
    best_acc, best_epoch, best_state = -1.0, 0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for X, y in train_loader:
            logits = model(X)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
        preds, golds = predict(model, dev_loader)
        acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
        improved = acc > best_acc
        if improved:
            best_acc, best_epoch = acc, epoch
            best_state = copy.deepcopy(model.state_dict())
        print(f"Epoch {epoch:2d}/{args.epochs}  loss={total/len(train_rows):.4f}  "
              f"dev_acc={acc:.4f}{'  * 最佳' if improved else ''}")
    elapsed = time.time() - t0
    print(f"\n训练耗时：{elapsed:.0f} 秒（{args.epochs} 个 epoch，{args.threads} 线程）")

    # 恢复验证集最佳模型后再评测测试集，测试结果不参与选轮次。
    model.load_state_dict(best_state)
    print(f"采用第 {best_epoch} 轮模型（dev_acc={best_acc:.4f}）")
    preds, golds = predict(model, test_loader)
    acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
    f1 = macro_f1(preds, golds, num_classes)
    print(f"\n测试集：accuracy = {acc:.4f}   macro-F1 = {f1:.4f}")
    print_confusion(confusion(preds, golds, LABELS), LABELS)

    # 逐类准确率
    print("\n  各语种准确率：")
    for c, lab in enumerate(LABELS):
        idx = [i for i, g in enumerate(golds) if g == c]
        hit = sum(preds[i] == c for i in idx)
        print(f"    {lab}: {hit}/{len(idx)} = {hit/max(1,len(idx)):.3f}")

    # 推理示例
    print("\n--- 推理示例 ---")
    model.eval()
    demos = [t for t, _ in test_rows[:4]]
    with torch.no_grad():
        for text in demos:
            logits = model(torch.tensor([encode(text, vocab)]))
            prob = torch.softmax(logits, dim=1)[0]
            pid = int(prob.argmax())
            print(f"  [{LABELS[pid]} {prob[pid]:.2f}]  {text[:50]}")

    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "vocab": dict(vocab),
                    "labels": LABELS, "maxlen": MAX_LEN,
                    "embed_dim": EMBED_DIM, "hidden_dim": HIDDEN_DIM,
                    "dropout": DROPOUT, "seed": SEED,
                    "best_epoch": best_epoch, "dev_accuracy": best_acc,
                    "test_accuracy": acc, "test_macro_f1": f1}, args.save)
        print(f"\n模型已保存：{args.save}")


if __name__ == "__main__":
    main()
