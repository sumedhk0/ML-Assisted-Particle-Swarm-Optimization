"""ML-driven particle repositioning layer.

Every K iterations the owning optimizer calls `reposition(...)`. We:
 1. Extract features for all N particles.
 2. Score each with the trained LightGBM classifier.
 3. Flag the top `top_k_frac` by P(stuck), skipping any particle the base
    variant just moved.
 4. Split the flagged set between exploitation (LCB minimum) and exploration
    (max uncertainty). Default split is 50/50; with `use_uncertainty_split`
    it becomes a continuous ratio derived from GP posterior std.
 5. Place the chosen particles. Default uses one shared target per group +
    Gaussian jitter; with `use_batch_acq` each group uses local-penalization
    batch acquisition for diverse, non-redundant targets.
"""
import numpy as np
import torch

from features import extract_features


class MLRepositioner:
    def __init__(self, classifier_path: str,
                 top_k_frac: float = 0.2, jitter_frac: float = 0.05,
                 use_batch_acq: bool = False,
                 use_uncertainty_split: bool = False):
        import lightgbm as lgb
        self.booster = lgb.Booster(model_file=classifier_path)
        self.top_k_frac = top_k_frac
        self.jitter_frac = jitter_frac
        self.use_batch_acq = use_batch_acq
        self.use_uncertainty_split = use_uncertainty_split

    def reposition(self, swarm, gp, memory, iter_idx: int, max_iters: int,
                   skip_indices=None):
        device = swarm.device
        skip = set(int(i) for i in (skip_indices or []))

        X = extract_features(swarm, gp, memory, iter_idx, max_iters)
        probs = self.booster.predict(X.detach().cpu().numpy())
        probs = np.asarray(probs, dtype=np.float64)
        # Exclude variant-moved particles by giving them an impossible score
        for i in skip:
            probs[i] = -1.0

        k = max(1, int(swarm.n * self.top_k_frac))
        candidates = np.argsort(probs)[-k:]
        flagged = [int(i) for i in candidates if probs[i] >= 0.0]
        if not flagged:
            return

        # Cache reference targets only when needed, in the same order as the
        # original v1 code so RNG state matches bit-for-bit when both new
        # flags are off.
        lcb_target = None
        unc_target = None

        if self.use_uncertainty_split:
            lcb_target, _ = gp.find_lcb_minimum(swarm.bounds, kappa=1.6)
            unc_target, max_std = gp.find_max_uncertainty(swarm.bounds)
            _, std_at_lcb = gp.predict(lcb_target.unsqueeze(0))
            ratio = float(std_at_lcb) / (float(max_std) + 1e-9)
            explore_frac = float(np.clip(1.0 - ratio, 0.2, 0.8))
            n_explore = max(1, int(round(len(flagged) * explore_frac)))
            n_exploit = len(flagged) - n_explore
        else:
            n_exploit = len(flagged) // 2
            n_explore = len(flagged) - n_exploit

        exploit_idx = flagged[:n_exploit]
        explore_idx = flagged[n_exploit:n_exploit + n_explore]

        low, high = swarm.bounds
        jitter_sigma = self.jitter_frac * (high - low)

        indices: list[int] = []
        positions: list[torch.Tensor] = []

        if exploit_idx:
            if self.use_batch_acq:
                targets = gp.find_batch_lcb_minimum(
                    swarm.bounds, K=len(exploit_idx), kappa=1.6)
                for i, t in zip(exploit_idx, targets):
                    positions.append(t.clamp(low, high))
                    indices.append(i)
            else:
                if lcb_target is None:
                    lcb_target, _ = gp.find_lcb_minimum(swarm.bounds, kappa=1.6)
                for i in exploit_idx:
                    p = lcb_target + torch.randn(swarm.dim, device=device) * jitter_sigma
                    positions.append(p.clamp(low, high))
                    indices.append(i)

        if explore_idx:
            if self.use_batch_acq:
                targets = gp.find_batch_max_uncertainty(
                    swarm.bounds, K=len(explore_idx))
                for i, t in zip(explore_idx, targets):
                    positions.append(t.clamp(low, high))
                    indices.append(i)
            else:
                if unc_target is None:
                    unc_target, _ = gp.find_max_uncertainty(swarm.bounds)
                for i in explore_idx:
                    p = unc_target + torch.randn(swarm.dim, device=device) * jitter_sigma
                    positions.append(p.clamp(low, high))
                    indices.append(i)

        idx_tensor = torch.tensor(indices, device=device, dtype=torch.long)
        pos_tensor = torch.stack(positions)
        swarm.reset_particles(idx_tensor, pos_tensor)
