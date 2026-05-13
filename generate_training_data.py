"""Generate labeled training data for the stuck-particle classifier.

For each synthetic PSO run:
 - Create a random objective (standard benchmark or Gaussian mixture).
 - Run plain-PSO for `n_iters` steps.
 - At every `sample_every`-th iteration, refit a GP on the accumulated memory
   and record the 11 per-particle features.
 - At run end, label each particle "stuck" (1) if its final pbest is more than
   `label_thresh * domain_diameter` from the known global optimum.

Training runs sample their dimensionality from DIM_BUCKET so the classifier
sees a mix of scales.
"""
import argparse

import numpy as np
import torch

from function import get_function, GaussianMixture, _STANDARD
from swarm import Swarm
from gp_surrogate import GPSurrogate
from memory_manager import MemoryManager
from features import extract_features


STANDARD_FUNCS = list(_STANDARD.keys())
DIM_BUCKET = [5, 10, 20, 50]


def _run_one(func, dim: int, n_iters: int, sample_every: int,
             device: str, seed: int, label_thresh: float = 0.1):
    torch.manual_seed(seed)
    n_particles = max(50, 4 * dim)

    swarm = Swarm(n_particles, dim, func, device=device, seed=seed)
    # Keep per-iteration GP fit cheap during data generation
    gp = GPSurrogate(dim=dim, device=device, fit_iters=40)
    memory = MemoryManager()
    memory.initialize(swarm.positions.clone(), swarm.last_values.clone())

    X_mem, y_mem = memory.get_training_data()
    gp.fit(X_mem, y_mem)

    feats_all = []
    for it in range(1, n_iters + 1):
        swarm.step_standard(omega=0.7, phi_p=2.05, phi_g=2.05, constriction=0.729)
        memory.update(gp, swarm.positions.clone(), swarm.last_values.clone())

        if it % sample_every == 0:
            X_mem, y_mem = memory.get_training_data()
            gp.fit(X_mem, y_mem)
            feats = extract_features(swarm, gp, memory, it, n_iters)
            feats_all.append(feats.detach().cpu().numpy())

    dists = (swarm.pbest - func.global_optimum).norm(dim=-1).detach().cpu().numpy()
    diameter = func.domain_diameter()
    per_particle_label = (dists > label_thresh * diameter).astype(np.int32)

    # Each sampled iteration gets the same per-particle label (end-of-run outcome)
    X = np.concatenate(feats_all, axis=0)
    y = np.tile(per_particle_label, len(feats_all))
    return X, y


def _parse_function_choices(functions: str | None) -> list[str] | None:
    if functions is None:
        return None
    names = [name.strip() for name in functions.split(",") if name.strip()]
    if not names:
        raise ValueError("At least one function name is required")

    valid = set(STANDARD_FUNCS) | {"gaussian_mixture"}
    unknown = [name for name in names if name not in valid]
    if unknown:
        raise ValueError(
            f"Unknown function(s): {unknown}. "
            f"Available: {sorted(valid)}"
        )
    return names


def _parse_dim_choices(dims: str | None) -> list[int]:
    if dims is None:
        return list(DIM_BUCKET)
    values = [int(part.strip()) for part in dims.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one dimension is required")
    if any(dim <= 0 for dim in values):
        raise ValueError(f"Dimensions must be positive: {values}")
    return values


def generate(n_runs: int, out_path: str, device: str,
             n_iters: int = 200, sample_every: int = 5,
             function_choices: list[str] | None = None,
             dim_choices: list[int] | None = None):
    dim_bucket = list(DIM_BUCKET) if dim_choices is None else list(dim_choices)
    all_X, all_y, all_run = [], [], []

    for run_idx in range(n_runs):
        rng = np.random.default_rng(run_idx)
        dim = int(rng.choice(dim_bucket))

        if function_choices is None:
            if rng.random() < 0.5:
                name = str(rng.choice(STANDARD_FUNCS))
            else:
                name = "gaussian_mixture"
        else:
            name = str(rng.choice(function_choices))

        if name == "gaussian_mixture":
            func = GaussianMixture(dim=dim, seed=run_idx + 10_000, device=device)
            log_name = f"gm_{run_idx}"
        else:
            func = get_function(name, dim, device=device)
            log_name = name

        try:
            X, y = _run_one(func, dim, n_iters, sample_every, device, seed=run_idx)
        except Exception as e:
            print(f"[run {run_idx}] failed ({e}); skipping")
            continue

        all_X.append(X)
        all_y.append(y)
        all_run.append(np.full(len(y), run_idx, dtype=np.int32))
        print(f"[run {run_idx:4d}] dim={dim:3d} func={log_name:16s} "
              f"rows={len(y):5d} pos_rate={y.mean():.3f}")

    X = np.concatenate(all_X)
    y = np.concatenate(all_y)
    run_ids = np.concatenate(all_run)
    np.savez_compressed(out_path, X=X, y=y, run_ids=run_ids)
    print(f"\nSaved {len(y):,} rows to {out_path}  (pos_rate={y.mean():.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--out", type=str, default="training_data.npz")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument(
        "--functions",
        type=str,
        default=None,
        help="Optional comma-separated function names. "
             "Use standard benchmark names and/or gaussian_mixture.",
    )
    parser.add_argument(
        "--dims",
        type=str,
        default=None,
        help="Optional comma-separated list of dimensions.",
    )
    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU")
        device = "cpu"

    function_choices = _parse_function_choices(args.functions)
    dim_choices = _parse_dim_choices(args.dims)

    generate(args.runs, args.out, device=device,
             n_iters=args.iters,
             sample_every=args.sample_every,
             function_choices=function_choices,
             dim_choices=dim_choices)
