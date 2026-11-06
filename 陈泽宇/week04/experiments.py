"""Run a three-point learning-rate sweep and retrain the winner."""
import argparse
import json
from pathlib import Path

from data import CORPUS, ROOT, prepare, write_json
from train import select_device, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "sweep")
    args = parser.parse_args()
    protocol = json.loads((ROOT / "experiment.json").read_text(encoding="utf-8"))
    device = select_device(args.device)
    base = protocol["common"]
    data = prepare(CORPUS, base["context"], device)
    trials = []
    for lr in protocol["learning_rates"]:
        config = dict(base, lr=lr, epochs=protocol["trial_epochs"])
        trial = train(config, CORPUS, args.out / f"lr_{lr:g}", device, prepared=data)
        trials.append(trial)
        write_json(args.out / "trials.json", trials)
    winner = min(trials, key=lambda t: t["history"][t["best_epoch"] - 1]["val"]["nll"])
    config = dict(base, lr=winner["config"]["lr"], epochs=protocol["final_epochs"])
    write_json(args.out / "selected_config.json", config)
    train(config, CORPUS, args.out / "final", device, prepared=data)


if __name__ == "__main__":
    main()
