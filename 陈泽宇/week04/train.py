"""Train a causal language model and select its checkpoint on validation PPL."""
import argparse
import json
import math
import platform
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F

from data import CONTEXT, CORPUS, PAD, ROOT, prepare, write_json
from model import ModelConfig, TransformerLM


def select_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))
    return torch.device(name)


def autocast(device, precision):
    if device.type == "cuda" and precision == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def metrics(loss_sum, tokens, correct1=None, correct5=None):
    nll = loss_sum / tokens
    result = {"nll": nll, "ppl": math.exp(nll), "tokens": int(tokens)}
    if correct1 is not None:
        result.update(top1=correct1 / tokens, top5=correct5 / tokens)
    return result


@torch.inference_mode()
def evaluate(model, data, batch_size, device, precision="fp32"):
    model.eval()
    totals = torch.zeros(3, dtype=torch.float64, device=device)
    for x, y in data.batches(batch_size):
        with autocast(device, precision):
            logits = model(x).flatten(0, 1)
            target = y.flatten()
            loss = F.cross_entropy(logits, target, ignore_index=PAD, reduction="sum")
        top = logits.topk(5, dim=-1).indices
        hits = (top == target[:, None]) & (target[:, None] != PAD)
        totals += torch.stack((loss, hits[:, 0].sum(), hits.any(dim=-1).sum()))
    loss, top1, top5 = totals.tolist()
    return metrics(loss, data.tokens, top1, top5)


def schedule(step, total_steps, warmup_steps, lr):
    if step <= warmup_steps:
        return lr * step / max(1, warmup_steps)
    fraction = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return lr * (0.1 + 0.9 * (1 + math.cos(math.pi * fraction)) / 2)


def train(config, corpus, out, device, prepared=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config["seed"])
    rng = torch.Generator().manual_seed(config["seed"])
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    context = config.get("context", CONTEXT)
    if prepared is None:
        prepared = prepare(corpus, context, device)
    vocab, stats, data = prepared
    model_config = ModelConfig(vocab_size=len(vocab), context=context, **config["model"])
    model = TransformerLM(model_config).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": config["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=config["lr"], betas=(0.9, 0.95), fused=device.type == "cuda",
    )
    precision = "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp32"
    if config.get("precision") == "fp32":
        precision = "fp32"
    environment = {
        "python": platform.python_version(), "torch": str(torch.__version__),
        "cuda": torch.version.cuda if device.type == "cuda" else None,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "precision": precision,
    }
    print(f"{out.name} | {parameters:,} params | {environment}", flush=True)
    write_json(out / "data_stats.json", stats)
    batch_size = config["batch_size"]
    steps_per_epoch = data["train"].n_batches(batch_size)
    total_steps = config["epochs"] * steps_per_epoch
    warmup_steps = min(config["warmup_steps"], total_steps // 2)
    best = math.inf
    best_epoch = 0
    step = 0
    history = []
    for epoch in range(1, config["epochs"] + 1):
        started = time.perf_counter()
        model.train()
        train_loss = torch.zeros((), device=device)
        for x, y in data["train"].batches(batch_size, rng):
            step += 1
            lr = schedule(step, total_steps, warmup_steps, config["lr"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with autocast(device, precision):
                logits = model(x).flatten(0, 1)
                loss_sum = F.cross_entropy(logits, y.flatten(), ignore_index=PAD, reduction="sum")
                loss = loss_sum / (y != PAD).sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss_sum.detach()
            if step == 50:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                print(f"50/{steps_per_epoch} steps in {time.perf_counter() - started:.1f}s", flush=True)
        tr = metrics(train_loss.item(), data["train"].tokens)
        val = evaluate(model, data["val"], batch_size, device, precision)
        if not math.isfinite(val["nll"]):
            raise RuntimeError("Validation loss is non-finite.")
        if val["nll"] < best:
            best, best_epoch = val["nll"], epoch
            torch.save({
                "model_config": asdict(model_config),
                "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "vocab": vocab, "data_sha256": stats["sha256"],
                "split_sha256": stats["split_sha256"],
                "epoch": epoch, "validation": val,
            }, out / "best.pt")
        history.append({"epoch": epoch, "step": step, "lr": lr, "train": tr, "val": val,
                        "seconds": time.perf_counter() - started})
        report = {"config": config, "environment": environment, "parameters": parameters,
                  "best_epoch": best_epoch, "history": history}
        write_json(out / "history.json", report)
        print(f"{epoch:02d} | train PPL {tr['ppl']:.2f} | val PPL {val['ppl']:.2f} | "
              f"val top1 {val['top1']:.2%} | {history[-1]['seconds']:.1f}s", flush=True)
        if epoch - best_epoch >= config["patience"]:
            break
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--corpus", type=Path, default=CORPUS, help="UTF-8 txt or txt.gz")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "retrain")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    train(config, args.corpus, args.out, select_device(args.device))


if __name__ == "__main__":
    main()
