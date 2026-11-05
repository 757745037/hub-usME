"""Fit smoothed character N-grams on the same windows as the Transformer."""
import argparse
import math
from collections import Counter
from pathlib import Path

import torch

from data import CORPUS, ROOT, read_corpus, vocab_lookup, windows, write_json
from predict import CHECKPOINT

STRENGTHS = (1, 4, 16, 64, 256, 1024)


def counts(lines, lookup, context):
    unigram, bigram, trigram = Counter(), Counter(), Counter()
    for seq in windows(lines, lookup, context):
        unigram.update(seq[1:])
        bigram.update(zip(seq, seq[1:]))
        trigram.update(zip(seq, seq[1:], seq[2:]))
    context2 = Counter()
    context3 = Counter()
    for (a, _), count in bigram.items():
        context2[a] += count
    for (a, b, _), count in trigram.items():
        context3[a, b] += count
    return unigram, bigram, trigram, context2, context3


def scoring_tensors(lines, lookup, context, fitted):
    unigram, bigram, trigram, context2, context3 = fitted
    total = unigram.total() + 0.1 * len(lookup)
    rows = []
    # -1 is an evaluation sentinel. The first target in any window uses bigram context.
    for seq in windows(lines, lookup, context):
        for a, b, c in zip([-1, *seq], seq, seq[1:]):
            rows.append(((unigram[c] + 0.1) / total,
                         bigram[b, c], context2[b], trigram[a, b, c], context3[a, b]))
    return torch.tensor(rows, dtype=torch.float64).T


def nll(probs):
    return -probs.log().mean().item()


def score(rows, alpha2, alpha3):
    p1, c2, n2, c3, n3 = rows
    p2 = (c2 + alpha2 * p1) / (n2 + alpha2)
    p3 = (c3 + alpha3 * p2) / (n3 + alpha3)
    return nll(p1), nll(p2), nll(p3)


def benchmark(corpus, checkpoint):
    torch.set_num_threads(min(8, torch.get_num_threads()))
    split, meta = read_corpus(corpus)
    if meta["sha256"] != checkpoint["data_sha256"]:
        raise ValueError("Corpus checksum differs from checkpoint.")
    if meta["split_sha256"] != checkpoint["split_sha256"]:
        raise ValueError("Data split checksum differs from checkpoint.")
    lookup = vocab_lookup(checkpoint["vocab"])
    context = checkpoint["model_config"]["context"]
    fitted = counts(split["train"], lookup, context)
    val = scoring_tensors(split["val"], lookup, context, fitted)
    best2 = min(STRENGTHS, key=lambda a: score(val, a, 1)[1])
    best3 = min(((a, b) for a in STRENGTHS for b in STRENGTHS),
                key=lambda ab: score(val, *ab)[2])
    test = scoring_tensors(split["test"], lookup, context, fitted)
    result = {}
    for order, pair in enumerate(((1, 1), (best2, 1), best3), start=1):
        val_nll, test_nll = (score(rows, *pair)[order - 1] for rows in (val, test))
        result[f"{order}-gram"] = {
            "alpha2": pair[0] if order >= 2 else None,
            "alpha3": pair[1] if order == 3 else None,
            "val": {"nll": val_nll, "ppl": math.exp(val_nll), "tokens": val.shape[1]},
            "test": {"nll": test_nll, "ppl": math.exp(test_nll), "tokens": test.shape[1]},
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--json-out", type=Path, default=ROOT / "runs" / "ngram.json")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    result = benchmark(args.corpus, checkpoint)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.json_out, result)
    for name, scores in result.items():
        print(f"{name}: val PPL {scores['val']['ppl']:.2f}, test PPL {scores['test']['ppl']:.2f}")


if __name__ == "__main__":
    main()
