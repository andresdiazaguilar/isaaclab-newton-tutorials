"""Evaluate a Go2-Parkour checkpoint: success rate per obstacle type and difficulty level (see notebook 02, Part 6).

Usage (from the repository folder):
    ./run.sh python scripts/evaluate_parkour.py --checkpoint logs/rsl_rl/go2_parkour          # newest run, latest model
    ./run.sh python scripts/evaluate_parkour.py --checkpoint path/to/model_2999.pt --robots_per_tile 16
Prints the table and saves a heat map next to the checkpoint (success_<model>.png).
"""

import argparse
import glob
import os
import re

parser = argparse.ArgumentParser(description="Evaluate a Go2-Parkour checkpoint.")
parser.add_argument("--checkpoint", required=True, help="model_*.pt, a run folder, or a folder of runs")
parser.add_argument("--robots_per_tile", type=int, default=8)
parser.add_argument("--seconds", type=float, default=8.0)
args = parser.parse_args()
os.environ.pop("DISPLAY", None)  # headless

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from go2_parkour.evaluation import evaluate, plot_success


def latest_model(path):
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(path):
        return path
    for folder in [path] + sorted(glob.glob(os.path.join(path, "*/")), reverse=True):
        models = glob.glob(os.path.join(folder, "model_*.pt"))
        if models:
            return max(models, key=lambda p: int(re.search(r"model_(\d+)\.pt$", p).group(1)))
    raise FileNotFoundError(f"no model_*.pt in {path}")


checkpoint = latest_model(args.checkpoint)
print(f"evaluating {checkpoint}")
res = evaluate(checkpoint, robots_per_tile=args.robots_per_tile, seconds=args.seconds)
width = max(len(t) for t in res["types"])
print(f"\n{'':{width}}  " + " ".join(f"L{j:<4}" for j in range(res["success"].shape[1])))
for name, row in zip(res["types"], res["success"]):
    print(f"{name:{width}}  " + " ".join(f"{v:4.0%} " for v in row))
print(f"\nmean success: {res['success'].mean():.0%}")
plot_success(res)
out = os.path.join(os.path.dirname(checkpoint), f"success_{os.path.basename(checkpoint)[:-3]}.png")
plt.savefig(out, dpi=100)
print(f"heat map saved to {out}")
