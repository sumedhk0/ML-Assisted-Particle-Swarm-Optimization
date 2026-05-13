"""Trajectory tracing helpers for PSO runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class TraceRecorder:
    """Collects per-iteration swarm snapshots and writes them to a compressed NPZ."""

    def __init__(
        self,
        out_path: str | Path,
        *,
        func_name: str,
        dim: int,
        n_particles: int,
        variant: str,
        use_ml: bool,
        rescue_policy: str,
        seed: int,
        trace_every: int = 1,
    ):
        if trace_every < 1:
            raise ValueError("trace_every must be >= 1")

        self.out_path = Path(out_path)
        self.trace_every = int(trace_every)
        self.metadata = {
            "func_name": func_name,
            "dim": int(dim),
            "n_particles": int(n_particles),
            "variant": variant,
            "use_ml": bool(use_ml),
            "rescue_policy": rescue_policy,
            "seed": int(seed),
        }

        self.iterations: list[int] = []
        self.eval_counts: list[int] = []
        self.elapsed_times_sec: list[float] = []
        self.positions: list[np.ndarray] = []
        self.pbest_positions: list[np.ndarray] = []
        self.current_values: list[np.ndarray] = []
        self.pbest_values: list[np.ndarray] = []
        self.gbest_positions: list[np.ndarray] = []
        self.gbest_values: list[float] = []

    def should_record(self, iteration: int) -> bool:
        return iteration == 0 or (iteration % self.trace_every == 0)

    def capture(self, iteration: int, eval_count: int, swarm, elapsed_sec: float = 0.0) -> None:
        self.iterations.append(int(iteration))
        self.eval_counts.append(int(eval_count))
        self.elapsed_times_sec.append(float(elapsed_sec))
        self.positions.append(
            swarm.positions.detach().cpu().to(dtype=torch.float32).numpy()
        )
        self.pbest_positions.append(
            swarm.pbest.detach().cpu().to(dtype=torch.float32).numpy()
        )
        self.current_values.append(
            swarm.last_values.detach().cpu().to(dtype=torch.float32).numpy()
        )
        self.pbest_values.append(
            swarm.pbest_values.detach().cpu().to(dtype=torch.float32).numpy()
        )
        self.gbest_positions.append(
            swarm.global_best.detach().cpu().to(dtype=torch.float32).numpy()
        )
        self.gbest_values.append(float(swarm.global_best_value))

    def finalize(self) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.out_path,
            iterations=np.asarray(self.iterations, dtype=np.int32),
            eval_counts=np.asarray(self.eval_counts, dtype=np.int32),
            elapsed_times_sec=np.asarray(self.elapsed_times_sec, dtype=np.float32),
            positions=np.asarray(self.positions, dtype=np.float32),
            pbest_positions=np.asarray(self.pbest_positions, dtype=np.float32),
            current_values=np.asarray(self.current_values, dtype=np.float32),
            pbest_values=np.asarray(self.pbest_values, dtype=np.float32),
            gbest_positions=np.asarray(self.gbest_positions, dtype=np.float32),
            gbest_values=np.asarray(self.gbest_values, dtype=np.float32),
            **self.metadata,
        )


def trace_file_name(
    func_name: str,
    dim: int,
    variant: str,
    use_ml: bool,
    seed: int,
    rescue_policy: str | None = None,
) -> str:
    tag = variant if variant == "PSO" else f"GP-{variant}"
    if variant != "PSO":
        if rescue_policy == "learned" or (rescue_policy is None and use_ml):
            tag += "+ML"
        elif rescue_policy and rescue_policy != "none":
            tag += f"+{rescue_policy.upper()}"
    safe_tag = tag.replace("+", "_plus_")
    return f"trace_{func_name}_dim{dim}_{safe_tag}_seed{seed}.npz"
