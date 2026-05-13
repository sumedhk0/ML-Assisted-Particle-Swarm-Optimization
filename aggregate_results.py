"""Aggregate per-seed JSONL results (e.g. from SLURM array jobs) into a
summary table comparable to experiment.py's in-run stats."""
import argparse
import json
from collections import defaultdict

import numpy as np

from efficiency_metrics import parse_target_values, target_label
from rescue_policies import (
    RESCUE_POLICY_HEURISTIC_PLATEAU,
    RESCUE_POLICY_LEARNED,
    RESCUE_POLICY_NONE,
    RESCUE_POLICY_RANDOM,
)


def result_tag(variant: str, rescue_policy: str) -> str:
    if variant == "PSO":
        return variant
    if rescue_policy == RESCUE_POLICY_NONE:
        suffix = ""
    elif rescue_policy == RESCUE_POLICY_LEARNED:
        suffix = "+ML"
    elif rescue_policy == RESCUE_POLICY_RANDOM:
        suffix = "+RANDOM"
    elif rescue_policy == RESCUE_POLICY_HEURISTIC_PLATEAU:
        suffix = "+HEURISTIC"
    else:
        suffix = f"+{rescue_policy.upper()}"
    return f"GP-{variant}{suffix}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="One or more JSONL files")
    parser.add_argument(
        "--targets",
        type=str,
        default=None,
        help="Optional comma-delimited objective thresholds to summarize first-hit stats",
    )
    args = parser.parse_args()
    targets = parse_target_values(args.targets)

    grouped = defaultdict(list)
    for path in args.files:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                rescue_policy = d.get("rescue_policy")
                if rescue_policy is None:
                    rescue_policy = RESCUE_POLICY_LEARNED if bool(d["use_ml"]) else RESCUE_POLICY_NONE
                key = (d["func"], d["dim"], d["variant"], rescue_policy)
                grouped[key].append(d)

    last_func = None
    for (func, dim, variant, rescue_policy) in sorted(grouped):
        if func != last_func:
            print(f"\n=== {func}  dim={dim} ===")
            last_func = func
        rows = grouped[(func, dim, variant, rescue_policy)]
        vals = np.asarray([float(row["best_value"]) for row in rows])
        tag = result_tag(variant, rescue_policy)
        line = (
            f"  {tag:14s}  n={len(vals):3d}  "
            f"median={np.median(vals):.4e}  mean={vals.mean():.4e}  std={vals.std():.4e}"
        )
        if all("wall_time_sec" in row for row in rows):
            wall = np.asarray([float(row["wall_time_sec"]) for row in rows])
            line += f"  median_wall={np.median(wall):.2f}s"
        print(line)

        timing_fields = [
            "time_gp_fit_sec",
            "time_feature_sec",
            "time_inference_sec",
            "time_acquisition_sec",
            "time_rescue_reset_sec",
            "n_rescue_events",
            "n_particles_rescued",
        ]
        if all(field in row for row in rows for field in timing_fields):
            gp_fit = np.asarray([float(row["time_gp_fit_sec"]) for row in rows])
            feature = np.asarray([float(row["time_feature_sec"]) for row in rows])
            inference = np.asarray([float(row["time_inference_sec"]) for row in rows])
            acquisition = np.asarray([float(row["time_acquisition_sec"]) for row in rows])
            reset = np.asarray([float(row["time_rescue_reset_sec"]) for row in rows])
            rescue_events = np.asarray([int(row["n_rescue_events"]) for row in rows])
            rescued = np.asarray([int(row["n_particles_rescued"]) for row in rows])
            print(
                "    timing: "
                f"gp_fit={np.median(gp_fit):.2f}s  "
                f"feature={np.median(feature):.2f}s  "
                f"infer={np.median(inference):.2f}s  "
                f"acq={np.median(acquisition):.2f}s  "
                f"reset={np.median(reset):.2f}s  "
                f"rescues={np.median(rescue_events):.0f}  "
                f"rescued_particles={np.median(rescued):.0f}"
            )

        if all("stop_reason" in row for row in rows):
            stop_counts = defaultdict(int)
            for row in rows:
                stop_counts[str(row["stop_reason"])] += 1
            summary = ", ".join(
                f"{reason}={count}" for reason, count in sorted(stop_counts.items())
            )
            print(f"    stop_reasons: {summary}")

        for target in targets:
            label = target_label(target)
            hit_rows = []
            for row in rows:
                target_hits = row.get("target_hits")
                if not isinstance(target_hits, dict):
                    continue
                hit_record = target_hits.get(label)
                if isinstance(hit_record, dict) and bool(hit_record.get("hit")):
                    hit_rows.append(hit_record)
            if not hit_rows:
                print(f"    target<={label}: no hit data")
                continue
            elapsed = np.asarray([float(hit["elapsed_sec"]) for hit in hit_rows])
            evals = np.asarray([int(hit["eval_count"]) for hit in hit_rows])
            hit_rate = len(hit_rows) / len(rows)
            print(
                f"    target<={label}: hit_rate={hit_rate:.3f}  "
                f"median_elapsed={np.median(elapsed):.2f}s  median_evals={np.median(evals):.0f}"
            )


if __name__ == "__main__":
    main()
