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

from efficiency_metrics import (
    compute_target_hits,
    curve_file_name,
    parse_target_values,
    save_convergence_curve,
)
from function import _STANDARD
from gp_directed_optimizer import GPDirectedOptimizer, Variant
from rescue_policies import (
    RESCUE_POLICY_HEURISTIC_PLATEAU,
    RESCUE_POLICY_LEARNED,
    RESCUE_POLICY_NONE,
    RESCUE_POLICY_RANDOM,
    SUPPORTED_RESCUE_POLICIES,
)
from trace_utils import TraceRecorder, trace_file_name
from vanilla_pso import VanillaPSOOptimizer


PSO_REFERENCE_VARIANT = "PSO"
SUPPORTED_VARIANTS = [v.value for v in Variant] + [PSO_REFERENCE_VARIANT]
LEGACY_ML_POLICIES = [RESCUE_POLICY_NONE, RESCUE_POLICY_LEARNED]


def result_tag(variant: str, rescue_policy: str) -> str:
    if variant == PSO_REFERENCE_VARIANT:
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


def run_one(func_name: str, dim: int, variant: str, seed: int,
            rescue_policy: str, device: str, classifier_path: str,
            max_evals: int, max_wall_time_sec: float | None = None,
            trace_dir: str | None = None,
            trace_every: int = 1, curve_dir: str | None = None,
            targets: list[float] | None = None) -> dict[str, object]:
    trace_recorder = None
    n_particles = max(50, 4 * dim)
    targets = [] if targets is None else targets
    use_ml = rescue_policy == RESCUE_POLICY_LEARNED
    if trace_dir:
        trace_path = Path(trace_dir) / trace_file_name(
            func_name, dim, variant, use_ml, seed, rescue_policy=rescue_policy
        )
        trace_recorder = TraceRecorder(
            trace_path,
            func_name=func_name,
            dim=dim,
            n_particles=n_particles,
            variant=variant,
            use_ml=use_ml,
            rescue_policy=rescue_policy,
            seed=seed,
            trace_every=trace_every,
        )
    if variant == PSO_REFERENCE_VARIANT:
        if rescue_policy != RESCUE_POLICY_NONE:
            raise ValueError("Vanilla PSO reference does not support rescue policies")
        opt = VanillaPSOOptimizer(
            num_dim=dim,
            num_particles=n_particles,
            func_name=func_name,
            device=device,
            seed=seed,
            trace_recorder=trace_recorder,
        )
    else:
        opt = GPDirectedOptimizer(
            num_dim=dim,
            num_particles=n_particles,
            func_name=func_name,
            variant=Variant(variant),
            device=device,
            seed=seed,
            use_ml_repositioning=use_ml,
            ml_classifier_path=classifier_path,
            rescue_policy=rescue_policy,
            trace_recorder=trace_recorder,
        )
    _, best_val = opt.run(max_evals=max_evals, max_wall_time_sec=max_wall_time_sec)
    if curve_dir:
        curve_path = Path(curve_dir) / curve_file_name(
            func_name, dim, variant, use_ml, seed, rescue_policy=rescue_policy
        )
        save_convergence_curve(
            curve_path,
            func_name=func_name,
            dim=dim,
            n_particles=n_particles,
            variant=variant,
            use_ml=use_ml,
            rescue_policy=rescue_policy,
            seed=seed,
            stop_reason=opt.stop_reason,
            max_evals=max_evals,
            max_wall_time_sec=max_wall_time_sec,
            time_gp_fit_sec=opt.time_gp_fit_sec,
            time_feature_sec=opt.time_feature_sec,
            time_inference_sec=opt.time_inference_sec,
            time_acquisition_sec=opt.time_acquisition_sec,
            time_rescue_reset_sec=opt.time_rescue_reset_sec,
            n_rescue_events=opt.n_rescue_events,
            n_particles_rescued=opt.n_particles_rescued,
            iteration_history=opt.iteration_history,
            eval_history=opt.eval_history,
            elapsed_history_sec=opt.elapsed_history_sec,
            best_values=opt.history,
        )

    row: dict[str, object] = {
        "func": func_name,
        "dim": dim,
        "variant": variant,
        "use_ml": use_ml,
        "rescue_policy": rescue_policy,
        "seed": seed,
        "best_value": float(best_val),
        "wall_time_sec": float(opt.total_wall_time_sec),
        "eval_count": int(opt.eval_count),
        "n_iterations": int(opt.iter_idx),
        "n_particles": int(n_particles),
        "stop_reason": opt.stop_reason,
        "max_evals": int(max_evals),
        "max_wall_time_sec": (
            None if max_wall_time_sec is None else float(max_wall_time_sec)
        ),
        "time_gp_fit_sec": float(opt.time_gp_fit_sec),
        "time_feature_sec": float(opt.time_feature_sec),
        "time_inference_sec": float(opt.time_inference_sec),
        "time_acquisition_sec": float(opt.time_acquisition_sec),
        "time_rescue_reset_sec": float(opt.time_rescue_reset_sec),
        "n_rescue_events": int(opt.n_rescue_events),
        "n_particles_rescued": int(opt.n_particles_rescued),
    }
    if targets:
        row["target_hits"] = compute_target_hits(
            best_values=opt.history,
            iteration_history=opt.iteration_history,
            eval_history=opt.eval_history,
            elapsed_history_sec=opt.elapsed_history_sec,
            targets=targets,
        )
    return row


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
    parser.add_argument("--variants", type=str, default=",".join(SUPPORTED_VARIANTS))
    parser.add_argument("--max-evals", type=int, default=None,
                        help="Defaults to 200 * dim when omitted")
    parser.add_argument("--max-wall-time-sec", type=float, default=None,
                        help="Optional wall-clock budget per seed run")
    parser.add_argument("--seed-offset", type=int, default=0,
                        help="Useful for SLURM array: seed range becomes [offset, offset+seeds)")
    parser.add_argument("--out", type=str, default=None,
                        help="Optional JSONL path for per-seed results")
    parser.add_argument("--trace-dir", type=str, default=None,
                        help="Optional directory for per-seed trajectory traces")
    parser.add_argument("--trace-every", type=int, default=1,
                        help="Record every k-th iteration when tracing")
    parser.add_argument("--curve-dir", type=str, default=None,
                        help="Optional directory for lightweight best-so-far convergence curves")
    parser.add_argument("--targets", type=str, default=None,
                        help="Optional comma-delimited objective thresholds for first-hit metrics")
    parser.add_argument(
        "--rescue-policies",
        type=str,
        default=None,
        help="Optional comma-delimited rescue policies. "
             f"Available: {', '.join(SUPPORTED_RESCUE_POLICIES)}",
    )
    parser.add_argument("--ml-only", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()

    if args.ml_only and args.baseline_only:
        parser.error("--ml-only and --baseline-only are mutually exclusive")

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available; using CPU")
        device = "cpu"

    funcs = args.functions.split(",")
    variants = args.variants.split(",")
    invalid_variants = sorted(set(variants) - set(SUPPORTED_VARIANTS))
    if invalid_variants:
        parser.error(f"unsupported variants: {', '.join(invalid_variants)}")
    if args.trace_every < 1:
        parser.error("--trace-every must be >= 1")
    targets = parse_target_values(args.targets)
    max_evals = args.max_evals if args.max_evals is not None else 200 * args.dim
    if args.max_wall_time_sec is not None and args.max_wall_time_sec <= 0:
        parser.error("--max-wall-time-sec must be > 0")
    if args.rescue_policies is not None:
        rescue_policies = [part.strip() for part in args.rescue_policies.split(",") if part.strip()]
        invalid_policies = sorted(set(rescue_policies) - set(SUPPORTED_RESCUE_POLICIES))
        if invalid_policies:
            parser.error(f"unsupported rescue policies: {', '.join(invalid_policies)}")
        if args.ml_only or args.baseline_only:
            parser.error("--rescue-policies cannot be combined with --ml-only/--baseline-only")
    elif args.ml_only:
        rescue_policies = [RESCUE_POLICY_LEARNED]
    elif args.baseline_only:
        rescue_policies = [RESCUE_POLICY_NONE]
    else:
        rescue_policies = list(LEGACY_ML_POLICIES)
    has_clf = Path(args.classifier).exists()
    if RESCUE_POLICY_LEARNED in rescue_policies and not has_clf:
        print(f"WARN: classifier file {args.classifier} not found; learned rescue runs will be skipped")

    out_f = open(args.out, "w") if args.out else None
    if args.trace_dir:
        Path(args.trace_dir).mkdir(parents=True, exist_ok=True)
    if args.curve_dir:
        Path(args.curve_dir).mkdir(parents=True, exist_ok=True)

    for func_name in funcs:
        print(f"\n{'='*80}\nFunction: {func_name}  dim={args.dim}  seeds={args.seeds}\n{'='*80}")
        for v in variants:
            for rescue_policy in rescue_policies:
                use_ml = rescue_policy == RESCUE_POLICY_LEARNED
                if v == PSO_REFERENCE_VARIANT and rescue_policy != RESCUE_POLICY_NONE:
                    continue
                if rescue_policy == RESCUE_POLICY_LEARNED and not has_clf:
                    continue
                results: list[float] = []
                for s in range(args.seeds):
                    seed = args.seed_offset + s
                    row = run_one(
                        func_name, args.dim, v, seed, rescue_policy,
                        device, args.classifier, max_evals,
                        max_wall_time_sec=args.max_wall_time_sec,
                        trace_dir=args.trace_dir, trace_every=args.trace_every,
                        curve_dir=args.curve_dir, targets=targets,
                    )
                    results.append(float(row["best_value"]))
                    if out_f:
                        out_f.write(json.dumps(row) + "\n")
                        out_f.flush()
                tag = result_tag(v, rescue_policy)
                print(stats_line(tag, results))

    if out_f:
        out_f.close()


if __name__ == "__main__":
    main()
