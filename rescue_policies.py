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

    def __init__(self, *, top_k_frac: float = 0.2, jitter_frac: float = 0.05):
        self.top_k_frac = top_k_frac
        self.jitter_frac = jitter_frac

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

        n_exploit = len(flagged) // 2
        exploit_idx = flagged[:n_exploit]
        explore_idx = flagged[n_exploit:]

        indices: list[int] = []
        positions: list[torch.Tensor] = []

        started = time.perf_counter()
        if exploit_idx:
            target, _ = gp.find_lcb_minimum(swarm.bounds, kappa=1.6)
            for idx in exploit_idx:
                p = target + torch.randn(swarm.dim, device=device) * jitter_sigma
                positions.append(p.clamp(low, high))
                indices.append(idx)

        if explore_idx:
            target, _ = gp.find_max_uncertainty(swarm.bounds)
            for idx in explore_idx:
                p = target + torch.randn(swarm.dim, device=device) * jitter_sigma
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

    def __init__(self, classifier_path: str, *, top_k_frac: float = 0.2, jitter_frac: float = 0.05):
        super().__init__(top_k_frac=top_k_frac, jitter_frac=jitter_frac)
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
) -> RescuePolicy | None:
    if policy_name == RESCUE_POLICY_NONE:
        return None
    if policy_name == RESCUE_POLICY_LEARNED:
        return LearnedRescuePolicy(
            classifier_path=classifier_path,
            top_k_frac=top_k_frac,
            jitter_frac=jitter_frac,
        )
    if policy_name == RESCUE_POLICY_RANDOM:
        return RandomRescuePolicy(top_k_frac=top_k_frac, jitter_frac=jitter_frac)
    if policy_name == RESCUE_POLICY_HEURISTIC_PLATEAU:
        return HeuristicPlateauRescuePolicy(top_k_frac=top_k_frac, jitter_frac=jitter_frac)
    raise ValueError(
        f"Unknown rescue policy: {policy_name}. "
        f"Available: {SUPPORTED_RESCUE_POLICIES}"
    )
