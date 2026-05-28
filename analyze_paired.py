"""Paired (same-seed) comparison of variants from experiment.py JSONL output.

For every (function, seed) the optimizer is re-seeded with `torch.manual_seed(seed)`,
so each variant starts from the IDENTICAL initial swarm. That makes per-seed
differences attributable to the variant, not the seed. We report:

  - paired wins: for each seed, which variant produced the lowest best_value.
  - mean rank:   rank variants 1..N within each seed (1=best), average per variant.

Usage:
    python analyze_paired.py path/to/*.jsonl

Stdlib-only (no torch / project imports needed).
"""
import argparse
import collections
import glob
import json
import os
import sys


def load_rows(patterns):
    rows = []
    for pat in patterns:
        paths = glob.glob(pat) if any(c in pat for c in "*?[") else [pat]
        if not paths:
            print(f"warning: no files matched {pat!r}", file=sys.stderr)
        for path in paths:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", help="JSONL files (globs OK)")
    args = p.parse_args()

    rows = load_rows(args.files)
    if not rows:
        sys.exit("no rows loaded")

    # (func, seed) -> {variant: best_value}
    by_fs = collections.defaultdict(dict)
    for r in rows:
        by_fs[(r["func"], r["seed"])][r["variant"]] = r["best_value"]

    funcs = sorted({r["func"] for r in rows})
    varis = sorted({r["variant"] for r in rows})

    print(f"Loaded {len(rows)} rows: {len(funcs)} functions x {len(varis)} variants")
    print(f"Variants: {', '.join(varis)}\n")

    overall_wins = collections.Counter()
    overall_rank_sum = collections.defaultdict(float)
    overall_rank_count = collections.defaultdict(int)

    for f in funcs:
        seeds = sorted(s for (ff, s) in by_fs if ff == f)
        wins = collections.Counter()
        ranks = collections.defaultdict(list)
        skipped = 0
        for s in seeds:
            d = by_fs[(f, s)]
            if set(d.keys()) != set(varis):
                skipped += 1
                continue
            sorted_vars = sorted(d, key=d.get)
            wins[sorted_vars[0]] += 1
            for rk, v in enumerate(sorted_vars, 1):
                ranks[v].append(rk)

        n_complete = len(seeds) - skipped
        print(f"=== {f}   (n={n_complete} complete seeds"
              + (f", {skipped} skipped" if skipped else "")
              + ") ===")

        # Win counts
        winners_sorted = sorted(varis, key=lambda v: (-wins[v], v))
        print("  paired wins (lowest best_value per seed):")
        for v in winners_sorted:
            bar = "#" * wins[v]
            print(f"    {v:<3}  {wins[v]:>3d}/{n_complete}  {bar}")

        # Mean rank
        rank_pairs = []
        for v in varis:
            if ranks[v]:
                rank_pairs.append((sum(ranks[v]) / len(ranks[v]), v))
        rank_pairs.sort()
        print("  mean rank (1 = best on that seed):")
        for mr, v in rank_pairs:
            print(f"    {v:<3}  {mr:.2f}")
        print()

        for v in varis:
            overall_wins[v] += wins[v]
            overall_rank_sum[v] += sum(ranks[v])
            overall_rank_count[v] += len(ranks[v])

    # Cross-function summary
    print("=== overall (all functions pooled) ===")
    total = sum(overall_wins.values())
    print(f"  paired wins:")
    for v in sorted(varis, key=lambda v: (-overall_wins[v], v)):
        print(f"    {v:<3}  {overall_wins[v]:>3d}/{total}")
    print(f"  mean rank:")
    overall_ranks = [(overall_rank_sum[v] / overall_rank_count[v], v)
                     for v in varis if overall_rank_count[v]]
    for mr, v in sorted(overall_ranks):
        print(f"    {v:<3}  {mr:.2f}")


if __name__ == "__main__":
    main()
