"""Smoke-test runner for GP-Directed PSO with optional ML repositioning."""
import argparse
from pathlib import Path

import torch

from efficiency_metrics import compute_target_hits, parse_target_values, save_convergence_curve
from gp_directed_optimizer import GPDirectedOptimizer, Variant
from rescue_policies import (
    RESCUE_POLICY_LEARNED,
    RESCUE_POLICY_NONE,
    SUPPORTED_RESCUE_POLICIES,
)
from trace_utils import TraceRecorder


parser = argparse.ArgumentParser()
parser.add_argument("--dim", type=int, default=10)
parser.add_argument("--func", type=str, default="sphere")
parser.add_argument("--variant", type=str, default="A3")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--use-ml", action="store_true")
parser.add_argument("--classifier", type=str, default="stuck_classifier.lgb")
parser.add_argument("--ml-period", type=int, default=5)
parser.add_argument(
    "--rescue-policy",
    type=str,
    default=None,
    choices=SUPPORTED_RESCUE_POLICIES,
    help="Optional particle rescue policy. Defaults to learned when --use-ml is set, otherwise none.",
)
parser.add_argument("--trace-out", type=str, default=None)
parser.add_argument("--trace-every", type=int, default=1)
parser.add_argument("--curve-out", type=str, default=None)
parser.add_argument("--targets", type=str, default=None)
parser.add_argument("--max-wall-time-sec", type=float, default=None)
args = parser.parse_args()

device = args.device
if device.startswith("cuda") and not torch.cuda.is_available():
    print("CUDA not available; using CPU")
    device = "cpu"

n_particles = max(50, 4 * args.dim)
max_evals = 200 * args.dim
rescue_policy = args.rescue_policy
if rescue_policy is None:
    rescue_policy = RESCUE_POLICY_LEARNED if args.use_ml else RESCUE_POLICY_NONE
elif args.use_ml and rescue_policy != RESCUE_POLICY_LEARNED:
    parser.error("--use-ml cannot be combined with --rescue-policy other than learned")
trace_recorder = None
if args.trace_out:
    trace_recorder = TraceRecorder(
        args.trace_out,
        func_name=args.func,
        dim=args.dim,
        n_particles=n_particles,
        variant=args.variant,
        use_ml=(rescue_policy == RESCUE_POLICY_LEARNED),
        rescue_policy=rescue_policy,
        seed=args.seed,
        trace_every=args.trace_every,
    )

opt = GPDirectedOptimizer(
    num_dim=args.dim,
    num_particles=n_particles,
    func_name=args.func,
    variant=Variant(args.variant),
    device=device,
    seed=args.seed,
    use_ml_repositioning=(rescue_policy == RESCUE_POLICY_LEARNED),
    ml_classifier_path=args.classifier,
    ml_period=args.ml_period,
    rescue_policy=rescue_policy,
    trace_recorder=trace_recorder,
)
best_pos, best_val = opt.run(
    max_evals=max_evals,
    max_wall_time_sec=args.max_wall_time_sec,
)
if args.curve_out:
    save_convergence_curve(
        Path(args.curve_out),
        func_name=args.func,
        dim=args.dim,
        n_particles=n_particles,
        variant=args.variant,
        use_ml=(rescue_policy == RESCUE_POLICY_LEARNED),
        rescue_policy=rescue_policy,
        seed=args.seed,
        stop_reason=opt.stop_reason,
        max_evals=max_evals,
        max_wall_time_sec=args.max_wall_time_sec,
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
suffix = ""
if rescue_policy == RESCUE_POLICY_LEARNED:
    suffix = "+ML"
elif rescue_policy != RESCUE_POLICY_NONE:
    suffix = f"+{rescue_policy.upper()}"
print(f"Variant:     {args.variant}{suffix}")
print(f"Best value:  {best_val:.6e}")
print(f"Evaluations: {opt.eval_count}")
print(f"Wall time:   {opt.total_wall_time_sec:.3f}s")
print(f"Stop:        {opt.stop_reason}")
print(f"Memory size: {opt.memory.size}")
print(f"Rescue:      {rescue_policy}")
if args.trace_out:
    print(f"Trace file:   {args.trace_out}")
if args.curve_out:
    print(f"Curve file:   {args.curve_out}")
targets = parse_target_values(args.targets)
if targets:
    hits = compute_target_hits(
        best_values=opt.history,
        iteration_history=opt.iteration_history,
        eval_history=opt.eval_history,
        elapsed_history_sec=opt.elapsed_history_sec,
        targets=targets,
    )
    for label, record in hits.items():
        if record["hit"]:
            print(
                f"Target <= {label}: iter={record['iteration']} "
                f"evals={record['eval_count']} elapsed={record['elapsed_sec']:.3f}s"
            )
        else:
            print(f"Target <= {label}: not reached")
