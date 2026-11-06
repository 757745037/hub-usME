# -*- coding: utf-8 -*-
"""
build_vocab_v12.py —— 为 12 类语种重建字符表（作业第一件事）

为什么必须做这一步：
  课堂发的 vocab_v8.txt 是从 v8 训练集统计出来的，只认识 8 个语种用到的字符。
  v12 多了 ko（谚文）/ th（泰文）/ tr（土耳其语）/ vi（越南语），
  如果直接拿 v8 那张表跑 12 类，新语种的每个字符都会落到 1 号 <UNK>，
  输入对模型来说是一串"未知未知未知"，准确率会直接掉到随机猜的水平。

用法：
    python build_vocab_v12.py                     # 默认生成同目录 vocab_v12.txt
    python build_vocab_v12.py --out my_vocab.txt
    python build_vocab_v12.py --check             # 只看统计，不写文件

产物格式和课堂那张表一致：每行「字符 \\t id \\t 出现次数」，可直接喂给
train_langid_lstm.py --vocab 使用。
"""

import os
import argparse

from train_langid_lstm import (normalize, build_char_vocab_with_freq,
                               write_vocab_file, load_vocab_file, load_split)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "语种识别实战", "data", "langid", "v12")
DEFAULT_OUT = os.path.join(HERE, "vocab_v12.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true", help="只统计，不写文件")
    args = ap.parse_args()

    train_rows = load_split(args.data_dir, "train")
    sentences = [t for t, _ in train_rows]
    vocab, freq = build_char_vocab_with_freq(sentences)

    # 逐语种的字符覆盖情况：能直观看到"新语种到底带来了多少新字符"
    print(f"训练集 {len(train_rows)} 条，来自 {args.data_dir}")
    print(f"字符表大小：{len(vocab)}（含 <PAD>=0 / <UNK>=1）\n")
    print(f"  {'语种':<6}{'句子':>6}{'字符种类':>10}{'字符出现次数':>14}")
    per_lang = {}
    for t, l in train_rows:
        per_lang.setdefault(l, []).append(t)
    for lab in sorted(per_lang):
        chars = set()
        total = 0
        for s in per_lang[lab]:
            n = normalize(s)
            chars |= set(n)
            total += len(n)
        print(f"  {lab:<6}{len(per_lang[lab]):>6}{len(chars):>10}{total:>14}")

    # 累计覆盖率：前 N 个字符能覆盖多少字符出现次数（和 DATA_CARD 对得上）
    print("\n  累计覆盖率：")
    total_chars = sum(freq.values())
    acc = 0
    for n, (_ch, c) in enumerate(sorted(freq.items(), key=lambda kv: -kv[1]), 1):
        acc += c
        if n in (500, 800, 1000, 2000, 3000):
            print(f"    前 {n:4d} 个字符 → 覆盖 {acc/total_chars:.1%}")

    if args.check:
        print("\n（--check：未写文件）")
        return

    write_vocab_file(args.out, vocab, freq)
    print(f"\n已写出：{args.out}（{len(vocab)} 行数据）")

    # 自检：读回来能否覆盖训练集里所有真实字符
    back = load_vocab_file(args.out)
    train_chars = set()
    for s in sentences:
        train_chars |= set(normalize(s))
    miss = train_chars - set(back)
    print(f"自检：训练集字符 {len(train_chars)} 个，词表未覆盖 {len(miss)} 个 "
          f"{'（OK）' if not miss else '（异常：' + ''.join(sorted(miss))[:30] + '）'}")


if __name__ == "__main__":
    main()
