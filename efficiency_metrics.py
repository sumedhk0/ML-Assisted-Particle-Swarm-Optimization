"""Helpers for efficiency-oriented benchmark instrumentation."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def parse_target_values(spec: str | None) -> list[float]:
    """Parse a comma-delimited target list into sorted unique floats."""
    if spec is None:
        return []
    raw_parts = [part.strip() for part in spec.split(",")]
    parts = [part for part in raw_parts if part]
    if not parts:
        return []
    values = sorted({float(part) for part in parts})
    return values


def target_label(value: float) -> str:
    return format(float(value), ".12g")


def curve_file_name(
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
    return f"curve_{func_name}_dim{dim}_{safe_tag}_seed{seed}.npz"


def compute_target_hits(
    *,
    best_values: list[float],
    iteration_history: list[int],
    eval_history: list[int],
    elapsed_history_sec: list[float],
    targets: list[float],
) -> dict[str, dict[str, float | int | bool]]:
    """Compute first-hit metrics for one or more objective thresholds."""
    hits: dict[str, dict[str, float | int | bool]] = {}
    for target in targets:
        label = target_label(target)
        hit_record: dict[str, float | int | bool] = {
            "target_value": float(target),
            "hit": False,
        }
        for idx, best_value in enumerate(best_values):
            if best_value <= target:
                hit_record.update({
                    "hit": True,
                    "best_value": float(best_value),
                    "iteration": int(iteration_history[idx]),
                    "eval_count": int(eval_history[idx]),
                    "elapsed_sec": float(elapsed_history_sec[idx]),
                })
                break
        hits[label] = hit_record
    return hits


def save_convergence_curve(
    out_path: str | Path,
    *,
    func_name: str,
    dim: int,
    n_particles: int,
    variant: str,
    use_ml: bool,
    rescue_policy: str,
    seed: int,
    stop_reason: str,
    max_evals: int,
    max_wall_time_sec: float | None,
    time_gp_fit_sec: float,
    time_feature_sec: float,
    time_inference_sec: float,
    time_acquisition_sec: float,
    time_rescue_reset_sec: float,
    n_rescue_events: int,
    n_particles_rescued: int,
    iteration_history: list[int],
    eval_history: list[int],
    elapsed_history_sec: list[float],
    best_values: list[float],
) -> None:
    """Write lightweight best-so-far convergence data to a compressed NPZ."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        func_name=func_name,
        dim=int(dim),
        n_particles=int(n_particles),
        variant=variant,
        use_ml=bool(use_ml),
        rescue_policy=rescue_policy,
        seed=int(seed),
        stop_reason=stop_reason,
        max_evals=int(max_evals),
        max_wall_time_sec=(
            np.nan if max_wall_time_sec is None else float(max_wall_time_sec)
        ),
        time_gp_fit_sec=float(time_gp_fit_sec),
        time_feature_sec=float(time_feature_sec),
        time_inference_sec=float(time_inference_sec),
        time_acquisition_sec=float(time_acquisition_sec),
        time_rescue_reset_sec=float(time_rescue_reset_sec),
        n_rescue_events=int(n_rescue_events),
        n_particles_rescued=int(n_particles_rescued),
        iterations=np.asarray(iteration_history, dtype=np.int32),
        eval_counts=np.asarray(eval_history, dtype=np.int32),
        elapsed_times_sec=np.asarray(elapsed_history_sec, dtype=np.float32),
        best_values=np.asarray(best_values, dtype=np.float32),
    )
