"""复评已训练模型，并与旧 RNN 比较验证集 PPL。可透传 language_model.py 参数。"""

from pathlib import Path
import subprocess
import sys


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    command = [
        sys.executable, "-u", "-X", "utf8", str(root / "language_model.py"),
        "--evaluate", str(root / "best_model_transformer_matched.pt"),
        "--baseline", str(root / "best_model_rnn.pt"),
        "--corpus", str(root / "corpus_general.txt"),
        "--batch_size", "128",
        *sys.argv[1:],
    ]
    raise SystemExit(subprocess.run(command, cwd=root).returncode)
