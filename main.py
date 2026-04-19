"""Smoke-test runner for plain PSO on the new Swarm API."""
import argparse

import torch

from function import get_function
from swarm import Swarm


parser = argparse.ArgumentParser()
parser.add_argument("--dim", type=int, default=3)
parser.add_argument("--func", type=str, default="sphere")
parser.add_argument("--n", type=int, default=50)
parser.add_argument("--iters", type=int, default=1000)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

device = args.device
if device.startswith("cuda") and not torch.cuda.is_available():
    print("CUDA not available; using CPU")
    device = "cpu"

func = get_function(args.func, args.dim, device=device)
swarm = Swarm(args.n, args.dim, func, device=device, seed=args.seed)

for i in range(args.iters):
    omega = max(0.4, 0.9 - 0.5 * i / args.iters)
    swarm.step_standard(omega=omega, phi_p=2.05, phi_g=2.05, constriction=0.729)

print(f"best position: {swarm.global_best.tolist()}")
print(f"best value:    {swarm.global_best_value:.6e}")
