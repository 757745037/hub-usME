# -*- coding: utf-8 -*-
"""
build_vocab_v12.py —— 作业第一步：从 v12 训练集生成 12 语种字符表

课堂发的 vocab_v8.txt 只覆盖 8 个语种；ko / th / vi / tr 的字符不在表里，
直接拿它训练 12 类会让这些语种的大量字符落到 <UNK>。

本脚本调用 train_langid_cls.py 里同一个 build_char_vocab()，统计
data/langid/v12/train.csv 的字符频率，写出同样格式的 vocab_v12.txt：

    # 字符<TAB>id<TAB>出现次数
    <PAD>    0    0
    <UNK>    1    0
    ...

运行：
    python build_vocab_v12.py
"""

import os
from collections import Counter

from train_langid_cls import build_char_vocab, load_split, normalize

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "langid", "v12")
OUT_PATH = os.path.join(HERE, "data", "langid", "vocab", "vocab_v12.txt")


def main():
    rows = load_split(DATA_DIR, "train")
    texts = [t for t, _ in rows]

    freq = Counter()
    for text in texts:
        freq.update(normalize(text))

    vocab = build_char_vocab(texts)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write("# 字符\tid\t出现次数\n")
        for ch, idx in vocab.items():
            f.write(f"{ch}\t{idx}\t{freq.get(ch, 0)}\n")

    total = sum(freq.values())
    acc = 0
    cover_at = 0
    real_chars = [ch for ch in vocab if ch not in ("<PAD>", "<UNK>")]
    for i, ch in enumerate(real_chars, start=1):
        acc += freq.get(ch, 0)
        if acc / total >= 0.95:
            cover_at = i
            break

    print(f"训练集：{len(texts)} 条")
    print(f"字符表：{len(vocab)} 个（含 <PAD>/<UNK>）")
    print(f"字符总出现次数：{total:,}")
    print(f"前 {cover_at} 个字符覆盖 95% 的字符出现次数")
    print(f"已写入：{OUT_PATH}")


if __name__ == "__main__":
    main()
