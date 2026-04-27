"""Smoke-test runner for plain PSO on the new Swarm API."""
import argparse
from pathlib import Path

import torch

from efficiency_metrics import compute_target_hits, parse_target_values, save_convergence_curve
from trace_utils import TraceRecorder
from vanilla_pso import VanillaPSOOptimizer


parser = argparse.ArgumentParser()
parser.add_argument("--dim", type=int, default=3)
parser.add_argument("--func", type=str, default="sphere")
parser.add_argument("--n", type=int, default=50)
parser.add_argument("--iters", type=int, default=1000)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--seed", type=int, default=0)
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

trace_recorder = None
if args.trace_out:
    trace_recorder = TraceRecorder(
        args.trace_out,
        func_name=args.func,
        dim=args.dim,
        n_particles=args.n,
        variant="PSO",
        use_ml=False,
        rescue_policy="none",
        seed=args.seed,
        trace_every=args.trace_every,
    )

opt = VanillaPSOOptimizer(
    num_dim=args.dim,
    num_particles=args.n,
    func_name=args.func,
    device=device,
    seed=args.seed,
    trace_recorder=trace_recorder,
)
best_pos, best_val = opt.run(
    max_evals=args.n * (args.iters + 1),
    max_wall_time_sec=args.max_wall_time_sec,
)

if args.curve_out:
    save_convergence_curve(
        Path(args.curve_out),
        func_name=args.func,
        dim=args.dim,
        n_particles=args.n,
        variant="PSO",
        use_ml=False,
        rescue_policy="none",
        seed=args.seed,
        stop_reason=opt.stop_reason,
        max_evals=args.n * (args.iters + 1),
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

print(f"best position: {best_pos.tolist()}")
print(f"best value:    {best_val:.6e}")
print(f"wall time:     {opt.total_wall_time_sec:.3f}s")
print(f"evaluations:   {opt.eval_count}")
print(f"stop reason:   {opt.stop_reason}")
if args.trace_out:
    print(f"trace file:    {args.trace_out}")
if args.curve_out:
    print(f"curve file:    {args.curve_out}")
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
                f"target <= {label}: hit at iter={record['iteration']} "
                f"evals={record['eval_count']} elapsed={record['elapsed_sec']:.3f}s"
            )
        else:
            print(f"target <= {label}: not reached")
