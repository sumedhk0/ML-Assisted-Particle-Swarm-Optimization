"""Aggregate per-seed JSONL results (e.g. from SLURM array jobs) into a
summary table comparable to experiment.py's in-run stats."""
import argparse
import json
from collections import defaultdict

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="One or more JSONL files")
    args = parser.parse_args()

    grouped = defaultdict(list)
    for path in args.files:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                key = (d["func"], d["dim"], d["variant"], bool(d["use_ml"]))
                grouped[key].append(float(d["best_value"]))

    last_func = None
    for (func, dim, variant, use_ml) in sorted(grouped):
        if func != last_func:
            print(f"\n=== {func}  dim={dim} ===")
            last_func = func
        vals = np.asarray(grouped[(func, dim, variant, use_ml)])
        tag = f"GP-{variant}{'+ML' if use_ml else ''}"
        print(f"  {tag:14s}  n={len(vals):3d}  "
              f"median={np.median(vals):.4e}  mean={vals.mean():.4e}  std={vals.std():.4e}")


if __name__ == "__main__":
    main()
