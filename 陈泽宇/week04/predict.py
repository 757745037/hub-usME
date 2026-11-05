"""Generate Chinese continuations or evaluate a saved Transformer checkpoint."""
import argparse
import json
from pathlib import Path

import torch

from data import BOS, CORPUS, EOS, PAD, ROOT, UNK, BucketBatches, encode, normalize
from data import read_corpus, vocab_lookup, write_json
from model import ModelConfig, TransformerLM
from train import evaluate, select_device

CHECKPOINT = ROOT / "checkpoints" / "best.pt"


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = TransformerLM(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval(), checkpoint


def evaluate_checkpoint(model, checkpoint, corpus, split_name, batch_size, device):
    split, metadata = read_corpus(corpus)
    if metadata["sha256"] != checkpoint["data_sha256"]:
        raise ValueError("The evaluation corpus must match the checkpoint's training corpus.")
    if metadata["split_sha256"] != checkpoint["split_sha256"]:
        raise ValueError("The evaluation split must match the checkpoint's data split.")
    lookup = vocab_lookup(checkpoint["vocab"])
    batches = BucketBatches(split[split_name], lookup, model.config.context, device)
    return evaluate(model, batches, batch_size, device)


@torch.inference_mode()
def generate(model, vocab, prompt, max_new=48, temperature=0.85, top_k=40, top_p=0.9, seed=42):
    if max_new < 0 or temperature < 0 or top_k < 0 or not 0 < top_p <= 1:
        raise ValueError("Use nonnegative max_new, temperature, top_k and 0 < top_p <= 1.")
    device = next(model.parameters()).device
    rng = torch.Generator(device=device).manual_seed(seed)
    prompt = normalize(prompt)
    lookup = vocab_lookup(vocab)
    ids = [BOS, *encode(prompt, lookup)]
    prefix_unknown = ids.count(UNK)
    continuation = []
    stopped = False
    for _ in range(max_new):
        x = torch.tensor([ids[-model.config.context:]], dtype=torch.long, device=device)
        logits = model(x, last_only=True)[0, 0].float()
        logits[[PAD, BOS, UNK]] = -torch.inf
        if temperature == 0:
            next_id = int(logits.argmax())
        else:
            sorted_logits, indices = (logits / temperature).topk(min(top_k or len(vocab), len(vocab)))
            probs = sorted_logits.softmax(dim=-1)
            if top_p < 1:
                remove = probs.cumsum(dim=-1) - probs >= top_p
                probs = probs.masked_fill(remove, 0)
            next_id = int(indices[torch.multinomial(probs, 1, generator=rng)].item())
        if next_id == EOS:
            stopped = True
            break
        ids.append(next_id)
        continuation.append(vocab[next_id])
    return {
        "prompt": prompt, "continuation": "".join(continuation),
        "text": prompt + "".join(continuation), "stopped_by_eos": stopped,
        "new_chars": len(continuation), "prompt_unknown_chars": prefix_unknown,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--prompt", default="怎么学习")
    parser.add_argument("--max-new", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.85, help="0 = greedy")
    parser.add_argument("--top-k", type=int, default=40, help="0 disables top-k")
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluate", choices=["val", "test"])
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    device = select_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    if args.evaluate:
        result = evaluate_checkpoint(model, checkpoint, args.corpus, args.evaluate, args.batch_size, device)
        result = {"split": args.evaluate, "epoch": checkpoint["epoch"], "device": str(device), **result}
    else:
        result = generate(model, checkpoint["vocab"], args.prompt, args.max_new,
                          args.temperature, args.top_k, args.top_p, args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.json_out:
        write_json(args.json_out, result)


if __name__ == "__main__":
    main()
