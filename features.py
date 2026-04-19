"""Per-particle feature extraction for the stuck-particle classifier.

Returns an (N, 11) tensor on the swarm's device. All 11 features are computed
from swarm state + a fitted GP + memory. The LightGBM classifier consumes the
CPU numpy form; the caller is responsible for the device transfer.
"""
import torch


FEATURE_NAMES = [
    "velocity_norm",
    "pbest_plateau",
    "value_rank",
    "dist_to_centroid",
    "dist_to_gbest",
    "gp_mean",
    "gp_std",
    "gp_pred_err",
    "min_dist_to_memory",
    "iter_progress",
    "recent_value_var",
]


def extract_features(swarm, gp, memory, iter_idx: int, max_iters: int) -> torch.Tensor:
    device = swarm.device
    N = swarm.n

    # 1. velocity magnitude
    v_norm = swarm.velocities.norm(dim=-1)

    # 2. pbest plateau length
    plateau = swarm.pbest_plateau.float()

    # 3. value rank in swarm, normalized to [0, 1]
    values = swarm.last_values
    ranks = values.argsort().argsort().float() / max(N - 1, 1)

    # 4. distance to swarm centroid
    centroid = swarm.positions.mean(dim=0, keepdim=True)
    dist_to_centroid = (swarm.positions - centroid).norm(dim=-1)

    # 5. distance to global best
    dist_to_gbest = (swarm.positions - swarm.global_best).norm(dim=-1)

    # 6, 7. GP posterior mean and std at current positions
    pred_mean, pred_std = gp.predict(swarm.positions, return_std=True)

    # 8. absolute prediction error (how much the GP is wrong here)
    pred_err = (values - pred_mean).abs()

    # 9. distance to nearest point in GP memory
    X_mem, _ = memory.get_training_data()
    # (N, 1, D) - (1, M, D) -> (N, M, D), norm -> (N, M), min -> (N,)
    diffs = swarm.positions.unsqueeze(1) - X_mem.unsqueeze(0)
    min_dist = diffs.norm(dim=-1).min(dim=1).values

    # 10. iteration progress (0 to 1)
    progress = torch.full((N,), float(iter_idx) / max(max_iters, 1), device=device)

    # 11. variance of this particle's recent values (ring buffer)
    recent_var = swarm.value_history.var(dim=-1)

    return torch.stack([
        v_norm, plateau, ranks, dist_to_centroid, dist_to_gbest,
        pred_mean, pred_std, pred_err, min_dist, progress, recent_var,
    ], dim=-1)
