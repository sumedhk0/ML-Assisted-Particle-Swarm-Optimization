"""ML-driven particle repositioning layer.

Every K iterations the owning optimizer calls `reposition(...)`. We:
 1. Extract features for all N particles.
 2. Score each with the trained LightGBM classifier.
 3. Flag the top `top_k_frac` by P(stuck), skipping any particle the base
    variant just moved.
 4. Split the flagged set 50/50: half teleport to the GP's LCB minimum
    (exploitation-with-uncertainty), half teleport to the GP's max-uncertainty
    point (pure exploration). Gaussian jitter breaks ties.
"""
import numpy as np
import torch

from features import extract_features


class MLRepositioner:
    def __init__(self, classifier_path: str,
                 top_k_frac: float = 0.2, jitter_frac: float = 0.05):
        import lightgbm as lgb
        self.booster = lgb.Booster(model_file=classifier_path)
        self.top_k_frac = top_k_frac
        self.jitter_frac = jitter_frac

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

        n_exploit = len(flagged) // 2
        exploit_idx = flagged[:n_exploit]
        explore_idx = flagged[n_exploit:]

        low, high = swarm.bounds
        jitter_sigma = self.jitter_frac * (high - low)

        indices: list[int] = []
        positions: list[torch.Tensor] = []

        if exploit_idx:
            target, _ = gp.find_lcb_minimum(swarm.bounds, kappa=1.6)
            for i in exploit_idx:
                p = target + torch.randn(swarm.dim, device=device) * jitter_sigma
                positions.append(p.clamp(low, high))
                indices.append(i)

        if explore_idx:
            target, _ = gp.find_max_uncertainty(swarm.bounds)
            for i in explore_idx:
                p = target + torch.randn(swarm.dim, device=device) * jitter_sigma
                positions.append(p.clamp(low, high))
                indices.append(i)

        idx_tensor = torch.tensor(indices, device=device, dtype=torch.long)
        pos_tensor = torch.stack(positions)
        swarm.reset_particles(idx_tensor, pos_tensor)
