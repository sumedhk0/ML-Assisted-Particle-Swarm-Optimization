"""Benchmark suite for GP-Directed PSO variants, with/without ML repositioning.

For each (function, variant, ml_state) triple, runs N seeds and prints summary
statistics. Optionally emits per-seed results as JSONL so seeds can be sharded
across SLURM array tasks and re-aggregated later.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from function import _STANDARD
from gp_directed_optimizer import GPDirectedOptimizer, Variant


def run_one(func_name: str, dim: int, variant: Variant, seed: int,
            use_ml: bool, device: str, classifier_path: str) -> float:
    opt = GPDirectedOptimizer(
        num_dim=dim,
        num_particles=max(50, 4 * dim),
        func_name=func_name,
        variant=variant,
        device=device,
        seed=seed,
        use_ml_repositioning=use_ml,
        ml_classifier_path=classifier_path,
    )
    _, best_val = opt.run(max_evals=200 * dim)
    return float(best_val)


def stats_line(tag: str, vals: list[float]) -> str:
    a = np.asarray(vals)
    return (f"  {tag:14s} mean={a.mean():.4e}  median={np.median(a):.4e}  "
            f"std={a.std():.4e}  min={a.min():.4e}  max={a.max():.4e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=51)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--classifier", type=str, default="stuck_classifier.lgb")
    parser.add_argument("--functions", type=str, default=",".join(_STANDARD.keys()))
    parser.add_argument("--variants", type=str, default=",".join(v.value for v in Variant))
    parser.add_argument("--seed-offset", type=int, default=0,
                        help="Useful for SLURM array: seed range becomes [offset, offset+seeds)")
    parser.add_argument("--out", type=str, default=None,
                        help="Optional JSONL path for per-seed results")
    parser.add_argument("--ml-only", action="store_true")
    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available; using CPU")
        device = "cpu"

    funcs = args.functions.split(",")
    variants = [Variant(v) for v in args.variants.split(",")]
    ml_states = [True] if args.ml_only else [False, True]
    has_clf = Path(args.classifier).exists()
    if True in ml_states and not has_clf:
        print(f"WARN: classifier file {args.classifier} not found; ML runs will be skipped")

    out_f = open(args.out, "w") if args.out else None

    for func_name in funcs:
        print(f"\n{'='*80}\nFunction: {func_name}  dim={args.dim}  seeds={args.seeds}\n{'='*80}")
        for v in variants:
            for use_ml in ml_states:
                if use_ml and not has_clf:
                    continue
                results: list[float] = []
                for s in range(args.seeds):
                    seed = args.seed_offset + s
                    val = run_one(func_name, args.dim, v, seed, use_ml, device, args.classifier)
                    results.append(val)
                    if out_f:
                        out_f.write(json.dumps({
                            "func": func_name, "dim": args.dim, "variant": v.value,
                            "use_ml": use_ml, "seed": seed, "best_value": val,
                        }) + "\n")
                        out_f.flush()
                tag = f"GP-{v.value}{'+ML' if use_ml else ''}"
                print(stats_line(tag, results))

    if out_f:
        out_f.close()


if __name__ == "__main__":
    main()
