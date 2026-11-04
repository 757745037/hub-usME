# -*- coding: utf-8 -*-
"""
eval_langid.py —— 12 类语种识别模型评测（LSTM 版）

直接复用 train_langid_lstm.py 里的模型定义、字符表和指标函数，
所以**只要改了训练脚本里的模型结构，评测会自动跟着变** —— 打分口径统一。

    python eval_langid.py --model langid_lstm_v12.pt                    # 常规测试集
    python eval_langid.py --model langid_lstm_v12.pt --split dev
    python eval_langid.py --model langid_lstm_v12.pt --split test_hard  # 难度切片
    python eval_langid.py --model langid_lstm_v12.pt --openset          # 开放集拒识

会打印：accuracy / macro-F1 / 逐类 P-R-F1 / 最常混淆的组合 / 按长度分档的准确率

三个难度切片（来自 DATA_CARD）：
  short       2~7 个字符的短句 —— 语种识别最真实的困难场景
  nodeaccent  去掉重音符号（café → cafe）—— 罗曼语族内部互相混淆
  nokana      日语去掉全部假名只剩汉字 —— 和中文本质上无法区分
"""

import os
import csv
import argparse
from collections import Counter

import torch

from train_langid_lstm import (LangIDLSTM, encode, normalize,
                               macro_f1, confusion, print_confusion,
                               MAX_LEN, EMBED_DIM, HIDDEN_DIM, DROPOUT,
                               NUM_LAYERS)

HERE = os.path.dirname(os.path.abspath(__file__))
LANGID_ROOT = os.path.join(os.path.dirname(HERE), "语种识别实战", "data", "langid")
DEFAULT_MODEL = os.path.join(HERE, "langid_lstm_v12.pt")


def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    vocab = ckpt["vocab"]
    labels = ckpt["labels"]
    model = LangIDLSTM(len(vocab), len(labels),
                       embed_dim=ckpt.get("embed_dim", EMBED_DIM),
                       hidden_dim=ckpt.get("hidden_dim", HIDDEN_DIM),
                       dropout=ckpt.get("dropout", DROPOUT),
                       num_layers=ckpt.get("num_layers", NUM_LAYERS),
                       cell=ckpt.get("cell", "lstm"),
                       use_pack=ckpt.get("use_pack", True),
                       bidirectional=ckpt.get("bidirectional", True))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, vocab, labels, ckpt.get("maxlen", MAX_LEN)


def read_csv(path):
    if not os.path.exists(path):
        raise SystemExit(f"找不到文件：{path}")
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def score_batch(model, texts, vocab, maxlen=MAX_LEN):
    """返回每条的 softmax 概率向量。"""
    ids = torch.tensor([encode(t, vocab, maxlen) for t in texts])
    with torch.no_grad():
        return torch.softmax(model(ids), dim=1)


def accuracy_by_length(texts, preds, golds, maxlen=MAX_LEN):
    """语种识别最真实的难点在短文本，按长度分档看才看得见。"""
    buckets = [(2, 7), (7, 12), (12, 20), (20, 30), (30, 999)]
    print("\n  按长度分档的准确率：")
    for lo, hi in buckets:
        idx = [i for i, t in enumerate(texts) if lo <= len(normalize(t)) < hi]
        if not idx:
            continue
        hit = sum(preds[i] == golds[i] for i in idx)
        print(f"    {lo:>3}~{hi if hi < 999 else '+':>3} 字符："
              f"{hit:4d}/{len(idx):4d} = {hit/len(idx):.3f}")


def per_class_report(preds, golds, labels):
    print("\n  逐类指标：")
    print(f"    {'标签':<6}{'样本':>6}{'precision':>12}{'recall':>10}{'f1':>10}")
    for c, lab in enumerate(labels):
        tp = sum(p == c and g == c for p, g in zip(preds, golds))
        fp = sum(p == c and g != c for p, g in zip(preds, golds))
        fn = sum(p != c and g == c for p, g in zip(preds, golds))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        print(f"    {lab:<6}{tp+fn:>6}{prec:>12.3f}{rec:>10.3f}{f1:>10.3f}")


def run_slices(model, vocab, labels):
    """难度集：按 slice 分组算准确率。"""
    rows = read_csv(os.path.join(LANGID_ROOT, "test_hard.csv"))
    label2id = {l: i for i, l in enumerate(labels)}
    print(f"\n=== 难度诊断集 {len(rows)} 条 ===")
    by_slice = {}
    for r in rows:
        by_slice.setdefault(r["slice"], []).append(r)

    for name in ("short", "nodeaccent", "nokana"):
        sub = by_slice.get(name, [])
        sub = [r for r in sub if r["label"] in label2id]     # 只评测模型认识的标签
        if not sub:
            print(f"  [{name}] 无可用样本")
            continue
        probs = score_batch(model, [r["text"] for r in sub], vocab)
        preds = probs.argmax(dim=1).tolist()
        golds = [label2id[r["label"]] for r in sub]
        acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
        print(f"  [{name}] {len(sub):4d} 条  accuracy = {acc:.3f}")
        if name == "nokana":
            c = Counter((labels[g], labels[p]) for p, g in zip(preds, golds))
            for (g, p), n in c.most_common(3):
                print(f"      真实 {g} → 预测 {p}: {n} 条")
        per = Counter(g for g in golds)
        hit = Counter(g for p, g in zip(preds, golds) if p == g)
        print("      " + "  ".join(f"{labels[g]}:{hit[g]}/{per[g]}"
                                   for g in sorted(per)))


def run_openset(model, vocab, labels, dev_path, maxlen=MAX_LEN):
    """
    开放集拒识：训练里没出现过的语种（el/bg/sw/ur）喂给模型，看它敢不敢说"不认识"。

    做法：先用验证集标定阈值 —— 验证集上 max(softmax) 的 5% 分位数，
    即"正常输入里有 95% 的置信度高于它"。测试时低于阈值就判 unknown。
    """
    rows = read_csv(os.path.join(LANGID_ROOT, "test_openset.csv"))
    dev = read_csv(dev_path)
    print(f"\n=== 开放集拒识（{len(rows)} 条未见语种）===")
    print(f"  训练见过的语种：{' '.join(labels)}")
    print(f"  开放集语种：{' '.join(sorted({r['label'] for r in rows}))}")

    dev_probs = score_batch(model, [r["text"] for r in dev], vocab, maxlen)
    dev_conf = dev_probs.max(dim=1)[0]
    thr = torch.quantile(dev_conf, 0.05).item()

    os_probs = score_batch(model, [r["text"] for r in rows], vocab, maxlen)
    os_conf = os_probs.max(dim=1)[0]

    keep = (dev_conf >= thr).float().mean().item()
    rej = (os_conf < thr).float().mean().item()
    print(f"  阈值（验证集 max-prob 的 5% 分位）= {thr:.3f}")
    print(f"  正常输入保留率 = {keep:.1%}   开放集拒识率 = {rej:.1%}")
    print(f"  开放集平均置信度 = {os_conf.mean():.3f} （正常输入 = {dev_conf.mean():.3f}）")

    print("\n  没见过的语种被认成了谁（超过阈值的那部分）：")
    preds = os_probs.argmax(dim=1).tolist()
    for lab in sorted({r["label"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["label"] == lab]
        c = Counter(labels[preds[i]] for i in idx)
        top = ", ".join(f"{k}:{v}" for k, v in c.most_common(3))
        mean_conf = os_conf[idx].mean().item()
        print(f"    {lab}（{len(idx)} 条，平均置信度 {mean_conf:.2f}）→ {top}")

    print("\n  阈值扫描（阈值 | 正常保留 | 开放集拒识）：")
    for t in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        k = (dev_conf >= t).float().mean().item()
        r = (os_conf < t).float().mean().item()
        print(f"    {t:.2f}   |  {k:6.1%}  |  {r:6.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--split", default="test",
                    choices=["test", "dev", "test_hard", "none"])
    ap.add_argument("--openset", action="store_true")
    ap.add_argument("--data-dir",
                    default=os.path.join(LANGID_ROOT, "v12"))
    args = ap.parse_args()

    torch.set_num_threads(1)
    model, vocab, labels, maxlen = load_model(args.model)
    print(f"模型：{os.path.basename(args.model)}   字符表：{len(vocab)}   "
          f"类别：{len(labels)}   最大长度：{maxlen}")

    if args.split == "test_hard":
        run_slices(model, vocab, labels)
    elif args.split != "none":
        rows = read_csv(os.path.join(args.data_dir, f"{args.split}.csv"))
        label2id = {l: i for i, l in enumerate(labels)}
        texts = [r["text"] for r in rows]
        probs = score_batch(model, texts, vocab, maxlen)
        preds = probs.argmax(dim=1).tolist()
        golds = [label2id[r["label"]] for r in rows]
        acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
        print(f"\n=== {args.split}：{len(rows)} 条 ===")
        print(f"  accuracy = {acc:.4f}   "
              f"macro-F1 = {macro_f1(preds, golds, len(labels)):.4f}")
        per_class_report(preds, golds, labels)
        print_confusion(confusion(preds, golds, labels), labels, top=6)
        accuracy_by_length(texts, preds, golds, maxlen)

    if args.openset:
        run_openset(model, vocab, labels,
                    os.path.join(args.data_dir, "dev.csv"), maxlen)


if __name__ == "__main__":
    main()
