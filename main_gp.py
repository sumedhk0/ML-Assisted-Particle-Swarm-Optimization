"""Smoke-test runner for GP-Directed PSO with optional ML repositioning."""
import argparse

import torch

from gp_directed_optimizer import GPDirectedOptimizer, Variant


parser = argparse.ArgumentParser()
parser.add_argument("--dim", type=int, default=10)
parser.add_argument("--func", type=str, default="sphere")
parser.add_argument("--variant", type=str, default="A3")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--use-ml", action="store_true")
parser.add_argument("--classifier", type=str, default="stuck_classifier.lgb")
parser.add_argument("--ml-period", type=int, default=5)
args = parser.parse_args()

device = args.device
if device.startswith("cuda") and not torch.cuda.is_available():
    print("CUDA not available; using CPU")
    device = "cpu"

n_particles = max(50, 4 * args.dim)
max_evals = 200 * args.dim

opt = GPDirectedOptimizer(
    num_dim=args.dim,
    num_particles=n_particles,
    func_name=args.func,
    variant=Variant(args.variant),
    device=device,
    seed=args.seed,
    use_ml_repositioning=args.use_ml,
    ml_classifier_path=args.classifier,
    ml_period=args.ml_period,
)
best_pos, best_val = opt.run(max_evals=max_evals)
print(f"Variant:     {args.variant}{'+ML' if args.use_ml else ''}")
print(f"Best value:  {best_val:.6e}")
print(f"Evaluations: {opt.eval_count}")
print(f"Memory size: {opt.memory.size}")
