"""Character vocabulary, deterministic corpus split, and length-bucketed batches."""
import gzip
import hashlib
import json
import random
import unicodedata
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "data" / "corpus_general.txt.gz"
PAD, BOS, EOS, UNK = range(4)
SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]
CONTEXT = 64
SPLIT_SEED = 42


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


def group_key(line):
    """Keep punctuation, case and whitespace variants in the same split."""
    key = "".join(c for c in line.casefold()
                  if not c.isspace() and not unicodedata.category(c).startswith("P"))
    return key or line


def vocab_lookup(vocab):
    return {char: i for i, char in enumerate(vocab)}


def encode(text, lookup):
    return [lookup.get(char, UNK) for char in text]


def read_corpus(path):
    path = Path(path)
    raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    lines = raw.decode("utf-8-sig").splitlines()
    normalized = [normalize(line) for line in lines]
    unique = list(dict.fromkeys(line for line in normalized if line))
    groups = {}
    for line in unique:
        groups.setdefault(group_key(line), []).append(line)
    keys = list(groups)
    random.Random(SPLIT_SEED).shuffle(keys)
    n_holdout = len(keys) // 20
    if not n_holdout:
        raise ValueError("Corpus must contain at least 20 distinct line groups.")
    partitions = {
        "train": keys[:-2 * n_holdout],
        "val": keys[-2 * n_holdout:-n_holdout],
        "test": keys[-n_holdout:],
    }
    split = {name: [line for key in subset for line in groups[key]]
             for name, subset in partitions.items()}
    split_hash = hashlib.sha256()
    for name, records in split.items():
        split_hash.update(f"{name}\n".encode("utf-8"))
        split_hash.update("\n".join(records).encode("utf-8"))
        split_hash.update(b"\n")
    return split, {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "split_sha256": split_hash.hexdigest(),
        "raw_lines": len(lines),
        "blank_lines": normalized.count(""),
        "duplicate_lines": len(lines) - normalized.count("") - len(unique),
        "split_seed": SPLIT_SEED,
        "groups": {name: len(subset) for name, subset in partitions.items()},
    }


def windows(lines, lookup, context=CONTEXT):
    for line in lines:
        seq = [BOS, *encode(line, lookup), EOS]
        # A one-token overlap preserves the shift at each window boundary.
        for start in range(0, len(seq) - 1, context):
            yield seq[start:start + context + 1]


class BucketBatches:
    def __init__(self, lines, lookup, context, device):
        limits = sorted({min(n, context) for n in (8, 12, 16, 24, 32, 48, context)})
        rows = {width: [] for width in limits}
        self.tokens = 0
        for seq in windows(lines, lookup, context):
            width = next(width for width in limits if len(seq) <= width + 1)
            rows[width].append(seq + [PAD] * (width + 1 - len(seq)))
            self.tokens += len(seq) - 1
        self.groups = [torch.tensor(group, dtype=torch.long, device=device)
                       for group in rows.values() if group]

    def n_batches(self, batch_size):
        return sum((len(group) + batch_size - 1) // batch_size for group in self.groups)

    def batches(self, batch_size, rng=None):
        plan = []
        for group in self.groups:
            if rng is None:
                indices = torch.arange(len(group), device=group.device)
            else:
                indices = torch.randperm(len(group), generator=rng).to(group.device)
            plan.extend((group, part) for part in indices.split(batch_size))
        if rng is not None:
            order = torch.randperm(len(plan), generator=rng).tolist()
            plan = [plan[i] for i in order]
        for group, indices in plan:
            seq = group.index_select(0, indices)
            yield seq[:, :-1].contiguous(), seq[:, 1:].contiguous()


def prepare(path, context, device, max_chars=4000):
    split, stats = read_corpus(path)
    counts = Counter(char for line in split["train"] for char in line)
    vocab = SPECIAL + [c for c, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:max_chars]]
    lookup = vocab_lookup(vocab)
    stats["vocab_size"] = len(vocab)
    stats["splits"] = {}
    for name, lines in split.items():
        chars = sum(map(len, lines))
        unknown = sum(char not in lookup for line in lines for char in line)
        stats["splits"][name] = {
            "lines": len(lines), "characters": chars, "targets": chars + len(lines),
            "oov_chars": unknown, "oov_rate": unknown / chars,
            "long_lines": sum(len(line) + 1 > context for line in lines),
        }
    batches = {name: BucketBatches(split[name], lookup, context, device) for name in ("train", "val")}
    return vocab, stats, batches


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")
