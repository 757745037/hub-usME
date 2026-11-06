# -*- coding: utf-8 -*-
"""
eval_langid_v12.py —— 12 类语种识别模型评测

    python eval_langid_v12.py
    python eval_langid_v12.py --split dev
"""

import os
import csv
import argparse

import torch

from train_langid_cls_v12 import (LangIDClassifier, encode,
                              normalize, macro_f1, confusion, print_confusion,
                              EMBED_DIM, HIDDEN_DIM, DROPOUT,
                              DATA_DIR, MODEL_PATH)

def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    vocab = ckpt["vocab"]
    labels = ckpt["labels"]
    model = LangIDClassifier(len(vocab), len(labels),
                             embed_dim=ckpt.get("embed_dim", EMBED_DIM),
                             hidden_dim=ckpt.get("hidden_dim", HIDDEN_DIM),
                             dropout=ckpt.get("dropout", DROPOUT))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, vocab, labels


def read_csv(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def score_batch(model, texts, vocab):
    """返回每条的 softmax 概率向量。"""
    ids = torch.tensor([encode(t, vocab) for t in texts])
    with torch.no_grad():
        return torch.softmax(model(ids), dim=1)


def accuracy_by_length(texts, preds, golds):
    """语种识别最真实的难点在短文本，按长度分档看才看得见。"""
    buckets = [(2, 7), (8, 12), (12, 20), (20, 30), (30, 999)]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    ap.add_argument("--data-dir", default=DATA_DIR)
    args = ap.parse_args()

    torch.set_num_threads(1)
    model, vocab, labels = load_model(args.model)

    rows = read_csv(os.path.join(args.data_dir, f"{args.split}.csv"))
    label2id = {l: i for i, l in enumerate(labels)}
    texts = [r["text"] for r in rows]
    probs = score_batch(model, texts, vocab)
    preds = probs.argmax(dim=1).tolist()
    golds = [label2id[r["label"]] for r in rows]
    acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
    print(f"=== {args.split}：{len(rows)} 条 ===")
    print(f"  accuracy = {acc:.4f}   macro-F1 = {macro_f1(preds, golds, len(labels)):.4f}")
    per_class_report(preds, golds, labels)
    print_confusion(confusion(preds, golds, labels), labels, top=5)
    accuracy_by_length(texts, preds, golds)


if __name__ == "__main__":
    main()
