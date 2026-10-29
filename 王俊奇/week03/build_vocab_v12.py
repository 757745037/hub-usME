from collections import Counter, OrderedDict
import os
import unicodedata
import csv

name = "v12"
OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "langid")

def save_vocab(vocab, freq, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 字符\tid\t出现次数\n")
        for ch, idx in vocab.items():
            f.write(f"{ch}\t{idx}\t{freq.get(ch, 0)}\n")

def normalize(text):
    """NFKC + 小写 + 只保留字母(L)与组合符号(M)。这就是喂给模型的形式。"""
    t = unicodedata.normalize("NFKC", text).lower()
    return "".join(c for c in t if unicodedata.category(c)[0] in ("L", "M"))

def coverage(vocab, freq, ratio=0.95):
    """前多少个字符覆盖 ratio 的字符出现次数。"""
    total = sum(freq.values())
    acc, n = 0, 0
    for ch in vocab:
        if ch in ("<PAD>", "<UNK>"):
            continue
        n += 1
        acc += freq.get(ch, 0)
        if acc >= ratio * total:
            return n
    return n

# ─── 3. 字符表 ─────────────────────────────────────────────────────────
def build_char_vocab(sentences, min_freq=1):
    """
    字符表构建：统计训练集每个字的出现次数 → 排序 → 编号。

    第 0 号 <PAD>（补齐用，Embedding 里 padding_idx=0）
    第 1 号 <UNK>（字符级下极少用到）
    之后按出现次数从高到低编号，出现次数相同时按字符编码排序（保证可复现）。
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

train_rows = []
with open("./data/langid/v12/train.csv",encoding="utf-8") as f:
    for r in csv.DictReader(f):
        train_rows.append((r["text"], r["label"]))
vocab, freq = build_char_vocab([t for t, _ in train_rows])
save_vocab(vocab, freq, os.path.join(OUT_ROOT, "vocab", f"vocab_{name}.txt"))
print(f"  字符表：{len(vocab)} 个（含 <PAD>/<UNK>），前 {coverage(vocab, freq)} 个覆盖 95%"
        + ("（已写出文件）" if name == "v8" else "（不写文件，留给作业生成）"))