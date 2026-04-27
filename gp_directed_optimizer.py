"""GP-Directed PSO driver (GPyTorch + Swarm tensors).

Holds one Swarm and one GPSurrogate. Each iteration:
 1. Refit GP on memory.
 2. Variant-specific movement step.
 3. Update memory with surprising observations (chi-selection).
 4. Optional: ML-driven repositioning every `ml_period` iterations.
"""
from enum import Enum
import time

import torch

from function import Function, get_function
from rescue_policies import (
    RESCUE_POLICY_LEARNED,
    RESCUE_POLICY_NONE,
    SUPPORTED_RESCUE_POLICIES,
    build_rescue_policy,
)
from swarm import Swarm
from gp_surrogate import GPSurrogate
from memory_manager import MemoryManager


class Variant(Enum):
    A1 = "A1"
    A2 = "A2"
    A3 = "A3"
    B = "B"
    C1 = "C1"
    C2 = "C2"


VARIANT_PARAMS = {
    Variant.A1: {"omega": 0.42, "phi_p": 1.20, "phi_g": 1.20, "phi_h": 0.75},
    Variant.A2: {"omega": 0.42, "phi_p": 1.55, "phi_g": 0.75, "phi_h": 0.75},
    Variant.A3: {"omega": 0.42, "phi_p": 0.75, "phi_g": 1.55, "phi_h": 0.75},
    Variant.B:  {"omega": 0.42, "phi_p": 1.20, "phi_g": 1.20},
    Variant.C1: {"omega": 0.42, "phi_p": 1.20, "phi_g": 1.20},
    Variant.C2: {"omega": 0.42, "phi_p": 1.20, "phi_g": 1.20},
}


class GPDirectedOptimizer:
    def __init__(self, num_dim: int, num_particles: int, func_name: str,
                 variant: Variant, device: str = "cpu", seed: int | None = None,
                 function: Function | None = None,
                 use_ml_repositioning: bool = False,
                 ml_classifier_path: str = "stuck_classifier.lgb",
                 ml_period: int = 5,
                 rescue_policy: str | None = None,
                 trace_recorder=None):
        if seed is not None:
            torch.manual_seed(seed)

        self.dim = num_dim
        self.n = num_particles
        self.variant = variant
        self.params = VARIANT_PARAMS[variant]
        self.device = device

        self.func = function if function is not None else get_function(func_name, num_dim, device=device)
        self.bounds = self.func.bounds

        self.swarm = Swarm(num_particles, num_dim, self.func, device=device, seed=seed)
        self.eval_count = num_particles
        self.history = [self.swarm.global_best_value]
        self.iter_idx = 0
        self.iteration_history = [0]
        self.eval_history = [self.eval_count]
        self.elapsed_history_sec = [0.0]
        self.total_wall_time_sec = 0.0
        self._run_started_at: float | None = None
        self.stop_reason = "eval_budget"
        self.time_gp_fit_sec = 0.0
        self.time_feature_sec = 0.0
        self.time_inference_sec = 0.0
        self.time_acquisition_sec = 0.0
        self.time_rescue_reset_sec = 0.0
        self.n_rescue_events = 0
        self.n_particles_rescued = 0

        # Acquisition cost scales with dimension
        self.acq_n_starts = max(50, 5 * num_dim)

        self.gp = GPSurrogate(dim=num_dim, device=device)
        self.memory = MemoryManager()
        self.memory.initialize(self.swarm.positions.clone(), self.swarm.last_values.clone())

        self.use_ml = use_ml_repositioning
        self.ml_period = ml_period
        if rescue_policy is None:
            rescue_policy = RESCUE_POLICY_LEARNED if use_ml_repositioning else RESCUE_POLICY_NONE
        if rescue_policy not in SUPPORTED_RESCUE_POLICIES:
            raise ValueError(
                f"Unsupported rescue policy: {rescue_policy}. "
                f"Available: {SUPPORTED_RESCUE_POLICIES}"
            )
        self.rescue_policy_name = rescue_policy
        self.rescue_policy = None
        self.trace_recorder = trace_recorder
        if rescue_policy != RESCUE_POLICY_NONE:
            self.rescue_policy = build_rescue_policy(
                rescue_policy,
                classifier_path=ml_classifier_path,
            )
        if self.trace_recorder is not None and self.trace_recorder.should_record(0):
            self.trace_recorder.capture(0, self.eval_count, self.swarm, elapsed_sec=0.0)

    def run(self, max_evals: int, max_wall_time_sec: float | None = None):
        max_iters = max_evals // self.n
        self._run_started_at = time.perf_counter()
        try:
            while self.eval_count + self.n <= max_evals:
                if self._wall_time_exhausted(max_wall_time_sec):
                    self.stop_reason = "wall_time"
                    break
                if not self._iteration(max_iters, max_wall_time_sec):
                    self.stop_reason = "wall_time"
                    break
            else:
                self.stop_reason = "eval_budget"
        finally:
            if self._run_started_at is not None:
                self.total_wall_time_sec = time.perf_counter() - self._run_started_at
            if self.trace_recorder is not None:
                self.trace_recorder.finalize()
        return self.swarm.global_best.clone(), self.swarm.global_best_value

    def _wall_time_exhausted(self, max_wall_time_sec: float | None) -> bool:
        if max_wall_time_sec is None or self._run_started_at is None:
            return False
        return (time.perf_counter() - self._run_started_at) >= max_wall_time_sec

    def _record_rescue_stats(self, stats) -> None:
        self.time_feature_sec += stats.time_feature_sec
        self.time_inference_sec += stats.time_inference_sec
        self.time_acquisition_sec += stats.time_acquisition_sec
        self.time_rescue_reset_sec += stats.time_rescue_reset_sec
        self.n_rescue_events += stats.n_rescue_events
        self.n_particles_rescued += stats.n_particles_rescued

    def _iteration(self, max_iters: int, max_wall_time_sec: float | None) -> bool:
        next_iter_idx = self.iter_idx + 1
        X_mem, y_mem = self.memory.get_training_data()
        started = time.perf_counter()
        self.gp.fit(X_mem, y_mem)
        self.time_gp_fit_sec += time.perf_counter() - started
        if self._wall_time_exhausted(max_wall_time_sec):
            return False

        self.iter_idx = next_iter_idx

        moved_by_variant: list[int] = []
        if self.variant in (Variant.A1, Variant.A2, Variant.A3):
            self._step_a()
        elif self.variant == Variant.B:
            moved_by_variant.append(self._step_b())
        else:  # C1, C2
            moved_by_variant.append(self._step_c())

        self.eval_count += self.n

        self.memory.update(
            self.gp,
            self.swarm.positions.clone(),
            self.swarm.last_values.clone(),
        )

        if (
            not self._wall_time_exhausted(max_wall_time_sec)
            and self.rescue_policy is not None
            and self.iter_idx > 0
            and self.iter_idx % self.ml_period == 0
        ):
            rescue_stats = self.rescue_policy.reposition(
                swarm=self.swarm, gp=self.gp, memory=self.memory,
                iter_idx=self.iter_idx, max_iters=max_iters,
                skip_indices=moved_by_variant,
            )
            self._record_rescue_stats(rescue_stats)

        self.history.append(self.swarm.global_best_value)
        self.iteration_history.append(self.iter_idx)
        self.eval_history.append(self.eval_count)
        elapsed_sec = time.perf_counter() - self._run_started_at
        self.elapsed_history_sec.append(elapsed_sec)
        if self.trace_recorder is not None and self.trace_recorder.should_record(self.iter_idx):
            self.trace_recorder.capture(
                self.iter_idx, self.eval_count, self.swarm, elapsed_sec=elapsed_sec
            )
        return not self._wall_time_exhausted(max_wall_time_sec)

    def _step_a(self):
        started = time.perf_counter()
        h, _ = self.gp.find_minimum(self.bounds, n_starts=self.acq_n_starts)
        self.time_acquisition_sec += time.perf_counter() - started
        self.swarm.step_variant_a(
            omega=self.params["omega"], phi_p=self.params["phi_p"],
            phi_g=self.params["phi_g"], phi_h=self.params["phi_h"], h=h,
        )

    def _step_b(self) -> int:
        self.swarm.step_standard(
            omega=self.params["omega"], phi_p=self.params["phi_p"],
            phi_g=self.params["phi_g"], constriction=1.0,
        )
        worst = self.swarm.worst_particle_index()
        started = time.perf_counter()
        h, _ = self.gp.find_minimum(self.bounds, n_starts=self.acq_n_starts)
        self.time_acquisition_sec += time.perf_counter() - started
        self.swarm.reset_particles(
            torch.tensor([worst], device=self.device, dtype=torch.long),
            h.unsqueeze(0),
        )
        return worst

    def _step_c(self) -> int:
        self.swarm.step_standard(
            omega=self.params["omega"], phi_p=self.params["phi_p"],
            phi_g=self.params["phi_g"], constriction=1.0,
        )
        worst = self.swarm.worst_particle_index()
        started = time.perf_counter()
        if self.variant == Variant.C1:
            target, _ = self.gp.find_lcb_minimum(
                self.bounds, kappa=1.6, n_starts=self.acq_n_starts)
        else:
            target, _ = self.gp.find_max_uncertainty(
                self.bounds, n_starts=self.acq_n_starts)
        self.time_acquisition_sec += time.perf_counter() - started
        self.swarm.reset_particles(
            torch.tensor([worst], device=self.device, dtype=torch.long),
            target.unsqueeze(0),
        )
        return worst
