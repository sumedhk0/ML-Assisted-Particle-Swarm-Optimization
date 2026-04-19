"""GP-Directed PSO driver (GPyTorch + Swarm tensors).

Holds one Swarm and one GPSurrogate. Each iteration:
 1. Refit GP on memory.
 2. Variant-specific movement step.
 3. Update memory with surprising observations (chi-selection).
 4. Optional: ML-driven repositioning every `ml_period` iterations.
"""
from enum import Enum

import torch

from function import Function, get_function
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
                 ml_period: int = 5):
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

        # Acquisition cost scales with dimension
        self.acq_n_starts = max(50, 5 * num_dim)

        self.gp = GPSurrogate(dim=num_dim, device=device)
        self.memory = MemoryManager()
        self.memory.initialize(self.swarm.positions.clone(), self.swarm.last_values.clone())

        self.use_ml = use_ml_repositioning
        self.ml_period = ml_period
        self.ml_repositioner = None
        if use_ml_repositioning:
            from ml_repositioner import MLRepositioner
            self.ml_repositioner = MLRepositioner(classifier_path=ml_classifier_path)

    def run(self, max_evals: int):
        max_iters = max_evals // self.n
        while self.eval_count + self.n <= max_evals:
            self._iteration(max_iters)
        return self.swarm.global_best.clone(), self.swarm.global_best_value

    def _iteration(self, max_iters: int):
        self.iter_idx += 1

        X_mem, y_mem = self.memory.get_training_data()
        self.gp.fit(X_mem, y_mem)

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

        if self.use_ml and self.iter_idx > 0 and self.iter_idx % self.ml_period == 0:
            self.ml_repositioner.reposition(
                swarm=self.swarm, gp=self.gp, memory=self.memory,
                iter_idx=self.iter_idx, max_iters=max_iters,
                skip_indices=moved_by_variant,
            )

        self.history.append(self.swarm.global_best_value)

    def _step_a(self):
        h, _ = self.gp.find_minimum(self.bounds, n_starts=self.acq_n_starts)
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
        h, _ = self.gp.find_minimum(self.bounds, n_starts=self.acq_n_starts)
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
        if self.variant == Variant.C1:
            target, _ = self.gp.find_lcb_minimum(
                self.bounds, kappa=1.6, n_starts=self.acq_n_starts)
        else:
            target, _ = self.gp.find_max_uncertainty(
                self.bounds, n_starts=self.acq_n_starts)
        self.swarm.reset_particles(
            torch.tensor([worst], device=self.device, dtype=torch.long),
            target.unsqueeze(0),
        )
        return worst
