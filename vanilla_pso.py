"""Plain PSO optimizer used as a non-GP reference baseline."""

import time

from function import Function, get_function
from swarm import Swarm


class VanillaPSOOptimizer:
    """Canonical PSO without GP guidance or ML repositioning."""

    def __init__(
        self,
        num_dim: int,
        num_particles: int,
        func_name: str,
        device: str = "cpu",
        seed: int | None = None,
        function: Function | None = None,
        omega: float = 0.7,
        phi_p: float = 2.05,
        phi_g: float = 2.05,
        constriction: float = 0.729,
        trace_recorder=None,
    ):
        self.dim = num_dim
        self.n = num_particles
        self.device = device
        self.omega = omega
        self.phi_p = phi_p
        self.phi_g = phi_g
        self.constriction = constriction
        self.trace_recorder = trace_recorder
        self.iter_idx = 0
        self.iteration_history = [0]
        self.eval_history = []
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

        self.func = function if function is not None else get_function(
            func_name, num_dim, device=device
        )
        self.swarm = Swarm(num_particles, num_dim, self.func, device=device, seed=seed)
        self.eval_count = num_particles
        self.history = [self.swarm.global_best_value]
        self.eval_history = [self.eval_count]
        if self.trace_recorder is not None and self.trace_recorder.should_record(0):
            self.trace_recorder.capture(0, self.eval_count, self.swarm, elapsed_sec=0.0)

    def run(self, max_evals: int, max_wall_time_sec: float | None = None):
        self._run_started_at = time.perf_counter()
        try:
            while self.eval_count + self.n <= max_evals:
                if self._wall_time_exhausted(max_wall_time_sec):
                    self.stop_reason = "wall_time"
                    break
                self.iter_idx += 1
                self.swarm.step_standard(
                    omega=self.omega,
                    phi_p=self.phi_p,
                    phi_g=self.phi_g,
                    constriction=self.constriction,
                )
                self.eval_count += self.n
                self.history.append(self.swarm.global_best_value)
                self.iteration_history.append(self.iter_idx)
                self.eval_history.append(self.eval_count)
                elapsed_sec = time.perf_counter() - self._run_started_at
                self.elapsed_history_sec.append(elapsed_sec)
                if self.trace_recorder is not None and self.trace_recorder.should_record(self.iter_idx):
                    self.trace_recorder.capture(
                        self.iter_idx, self.eval_count, self.swarm, elapsed_sec=elapsed_sec
                    )
                if self._wall_time_exhausted(max_wall_time_sec):
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
