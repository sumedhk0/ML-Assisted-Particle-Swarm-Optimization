"""Particle rescue policy implementations for GP-guided PSO."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import time

import numpy as np
import torch

from features import extract_features


RESCUE_POLICY_NONE = "none"
RESCUE_POLICY_LEARNED = "learned"
RESCUE_POLICY_RANDOM = "random"
RESCUE_POLICY_HEURISTIC_PLATEAU = "heuristic_plateau"
SUPPORTED_RESCUE_POLICIES = [
    RESCUE_POLICY_NONE,
    RESCUE_POLICY_LEARNED,
    RESCUE_POLICY_RANDOM,
    RESCUE_POLICY_HEURISTIC_PLATEAU,
]


@dataclass
class RescueStats:
    time_feature_sec: float = 0.0
    time_inference_sec: float = 0.0
    time_acquisition_sec: float = 0.0
    time_rescue_reset_sec: float = 0.0
    n_rescue_events: int = 0
    n_particles_rescued: int = 0

    def merge(self, other: "RescueStats") -> None:
        self.time_feature_sec += other.time_feature_sec
        self.time_inference_sec += other.time_inference_sec
        self.time_acquisition_sec += other.time_acquisition_sec
        self.time_rescue_reset_sec += other.time_rescue_reset_sec
        self.n_rescue_events += other.n_rescue_events
        self.n_particles_rescued += other.n_particles_rescued


class RescuePolicy(ABC):
    """Base class for selecting and relocating particles."""

    def __init__(self, *, top_k_frac: float = 0.2, jitter_frac: float = 0.05,
                 use_batch_acq: bool = False,
                 use_uncertainty_split: bool = False):
        self.top_k_frac = top_k_frac
        self.jitter_frac = jitter_frac
        self.use_batch_acq = use_batch_acq
        self.use_uncertainty_split = use_uncertainty_split

    @property
    @abstractmethod
    def policy_name(self) -> str:
        """Stable policy identifier for reporting and CLI plumbing."""

    def reposition(self, swarm, gp, memory, iter_idx: int, max_iters: int, skip_indices=None):
        flagged, stats = self.select_particles(
            swarm=swarm,
            gp=gp,
            memory=memory,
            iter_idx=iter_idx,
            max_iters=max_iters,
            skip_indices=skip_indices,
        )
        if not flagged:
            return stats

        device = swarm.device
        low, high = swarm.bounds
        jitter_sigma = self.jitter_frac * (high - low)

        # Lazy targets: with both v2 flags off, computation order matches the
        # pre-v2 baseline bit-for-bit so reproducibility is preserved.
        lcb_target = None
        unc_target = None

        started = time.perf_counter()
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

        indices: list[int] = []
        positions: list[torch.Tensor] = []

        if exploit_idx:
            if self.use_batch_acq:
                targets = gp.find_batch_lcb_minimum(
                    swarm.bounds, K=len(exploit_idx), kappa=1.6)
                for idx, t in zip(exploit_idx, targets):
                    positions.append(t.clamp(low, high))
                    indices.append(idx)
            else:
                if lcb_target is None:
                    lcb_target, _ = gp.find_lcb_minimum(swarm.bounds, kappa=1.6)
                for idx in exploit_idx:
                    p = lcb_target + torch.randn(swarm.dim, device=device) * jitter_sigma
                    positions.append(p.clamp(low, high))
                    indices.append(idx)

        if explore_idx:
            if self.use_batch_acq:
                targets = gp.find_batch_max_uncertainty(
                    swarm.bounds, K=len(explore_idx))
                for idx, t in zip(explore_idx, targets):
                    positions.append(t.clamp(low, high))
                    indices.append(idx)
            else:
                if unc_target is None:
                    unc_target, _ = gp.find_max_uncertainty(swarm.bounds)
                for idx in explore_idx:
                    p = unc_target + torch.randn(swarm.dim, device=device) * jitter_sigma
                    positions.append(p.clamp(low, high))
                    indices.append(idx)
        stats.time_acquisition_sec += time.perf_counter() - started

        idx_tensor = torch.tensor(indices, device=device, dtype=torch.long)
        pos_tensor = torch.stack(positions)
        started = time.perf_counter()
        swarm.reset_particles(idx_tensor, pos_tensor)
        stats.time_rescue_reset_sec += time.perf_counter() - started
        stats.n_rescue_events += 1
        stats.n_particles_rescued += len(indices)
        return stats

    @abstractmethod
    def select_particles(
        self, swarm, gp, memory, iter_idx: int, max_iters: int, skip_indices=None
    ) -> tuple[list[int], RescueStats]:
        """Return the particle indices to rescue."""

    def _eligible_indices(self, swarm, skip_indices=None) -> list[int]:
        skip = {int(i) for i in (skip_indices or [])}
        return [idx for idx in range(swarm.n) if idx not in skip]

    def _rescue_count(self, swarm, eligible_count: int) -> int:
        if eligible_count <= 0:
            return 0
        return min(eligible_count, max(1, int(swarm.n * self.top_k_frac)))


class LearnedRescuePolicy(RescuePolicy):
    """Classifier-guided rescue policy."""

    policy_name = RESCUE_POLICY_LEARNED

    def __init__(self, classifier_path: str, *, top_k_frac: float = 0.2, jitter_frac: float = 0.05,
                 use_batch_acq: bool = False, use_uncertainty_split: bool = False):
        super().__init__(top_k_frac=top_k_frac, jitter_frac=jitter_frac,
                         use_batch_acq=use_batch_acq,
                         use_uncertainty_split=use_uncertainty_split)
        import lightgbm as lgb

        self.booster = lgb.Booster(model_file=classifier_path)

    def select_particles(
        self, swarm, gp, memory, iter_idx: int, max_iters: int, skip_indices=None
    ) -> tuple[list[int], RescueStats]:
        stats = RescueStats()
        eligible = self._eligible_indices(swarm, skip_indices=skip_indices)
        k = self._rescue_count(swarm, len(eligible))
        if k == 0:
            return [], stats

        started = time.perf_counter()
        X = extract_features(swarm, gp, memory, iter_idx, max_iters)
        stats.time_feature_sec += time.perf_counter() - started
        started = time.perf_counter()
        probs = self.booster.predict(X.detach().cpu().numpy())
        stats.time_inference_sec += time.perf_counter() - started
        probs = np.asarray(probs, dtype=np.float64)
        ineligible = np.ones(swarm.n, dtype=bool)
        ineligible[eligible] = False
        probs[ineligible] = -1.0

        candidates = np.argsort(probs)[-k:]
        flagged = [int(i) for i in candidates if probs[i] >= 0.0]
        return flagged, stats


class RandomRescuePolicy(RescuePolicy):
    """Random rescue policy with the same rescue count as learned rescue."""

    policy_name = RESCUE_POLICY_RANDOM

    def select_particles(
        self, swarm, gp, memory, iter_idx: int, max_iters: int, skip_indices=None
    ) -> tuple[list[int], RescueStats]:
        eligible = self._eligible_indices(swarm, skip_indices=skip_indices)
        k = self._rescue_count(swarm, len(eligible))
        if k == 0:
            return [], RescueStats()

        perm = torch.randperm(len(eligible), device=swarm.device)[:k].detach().cpu().tolist()
        return [eligible[idx] for idx in perm], RescueStats()


class HeuristicPlateauRescuePolicy(RescuePolicy):
    """Heuristic rescue policy using plateau length, then current value."""

    policy_name = RESCUE_POLICY_HEURISTIC_PLATEAU

    def select_particles(
        self, swarm, gp, memory, iter_idx: int, max_iters: int, skip_indices=None
    ) -> tuple[list[int], RescueStats]:
        eligible = self._eligible_indices(swarm, skip_indices=skip_indices)
        k = self._rescue_count(swarm, len(eligible))
        if k == 0:
            return [], RescueStats()

        plateau = swarm.pbest_plateau.detach().cpu()
        current_values = swarm.last_values.detach().cpu()
        ranked = sorted(
            eligible,
            key=lambda idx: (int(plateau[idx]), float(current_values[idx])),
            reverse=True,
        )
        return ranked[:k], RescueStats()


def build_rescue_policy(
    policy_name: str,
    *,
    classifier_path: str,
    top_k_frac: float = 0.2,
    jitter_frac: float = 0.05,
    use_batch_acq: bool = False,
    use_uncertainty_split: bool = False,
) -> RescuePolicy | None:
    if policy_name == RESCUE_POLICY_NONE:
        return None
    if policy_name == RESCUE_POLICY_LEARNED:
        return LearnedRescuePolicy(
            classifier_path=classifier_path,
            top_k_frac=top_k_frac,
            jitter_frac=jitter_frac,
            use_batch_acq=use_batch_acq,
            use_uncertainty_split=use_uncertainty_split,
        )
    if policy_name == RESCUE_POLICY_RANDOM:
        return RandomRescuePolicy(
            top_k_frac=top_k_frac, jitter_frac=jitter_frac,
            use_batch_acq=use_batch_acq,
            use_uncertainty_split=use_uncertainty_split,
        )
    if policy_name == RESCUE_POLICY_HEURISTIC_PLATEAU:
        return HeuristicPlateauRescuePolicy(
            top_k_frac=top_k_frac, jitter_frac=jitter_frac,
            use_batch_acq=use_batch_acq,
            use_uncertainty_split=use_uncertainty_split,
        )
    raise ValueError(
        f"Unknown rescue policy: {policy_name}. "
        f"Available: {SUPPORTED_RESCUE_POLICIES}"
    )
