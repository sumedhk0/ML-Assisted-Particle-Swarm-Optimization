"""Render a saved PSO trace NPZ into a GIF animation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


def _load_scalar(data, key: str):
    value = data[key]
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def _project_positions(positions: np.ndarray, gbest_positions: np.ndarray, projection: str):
    dim = positions.shape[-1]
    if projection == "auto":
        projection = "raw" if dim <= 2 else "pca"

    if projection == "raw":
        if dim == 1:
            pos_2d = np.stack([positions[..., 0], np.zeros_like(positions[..., 0])], axis=-1)
            gbest_2d = np.stack([gbest_positions[..., 0], np.zeros_like(gbest_positions[..., 0])], axis=-1)
            return pos_2d, gbest_2d, "x", "0"
        if dim == 2:
            return positions, gbest_positions, "x", "y"
        raise ValueError("raw projection only supports traces with dim <= 2")

    if projection != "pca":
        raise ValueError("projection must be one of: auto, raw, pca")

    flat_positions = positions.reshape(-1, dim).astype(np.float64, copy=False)
    mean = flat_positions.mean(axis=0, keepdims=True)
    centered = flat_positions - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].T

    pos_2d = (positions - mean) @ components
    gbest_2d = (gbest_positions - mean) @ components
    return pos_2d, gbest_2d, "PC1", "PC2"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_file", type=str)
    parser.add_argument("--out", type=str, default=None, help="Defaults to <trace_stem>.gif")
    parser.add_argument("--projection", choices=["auto", "raw", "pca"], default="auto")
    parser.add_argument("--fps", type=int, default=6)
    args = parser.parse_args()

    trace_path = Path(args.trace_file)
    out_path = Path(args.out) if args.out else trace_path.with_suffix(".gif")

    with np.load(trace_path) as data:
        positions = data["positions"]
        gbest_positions = data["gbest_positions"]
        iterations = data["iterations"]
        gbest_values = data["gbest_values"]
        func_name = _load_scalar(data, "func_name")
        variant = _load_scalar(data, "variant")
        use_ml = bool(_load_scalar(data, "use_ml"))
        seed = int(_load_scalar(data, "seed"))

    pos_2d, gbest_2d, xlabel, ylabel = _project_positions(
        positions, gbest_positions, args.projection
    )

    x_min = min(pos_2d[..., 0].min(), gbest_2d[..., 0].min())
    x_max = max(pos_2d[..., 0].max(), gbest_2d[..., 0].max())
    y_min = min(pos_2d[..., 1].min(), gbest_2d[..., 1].min())
    y_max = max(pos_2d[..., 1].max(), gbest_2d[..., 1].max())

    x_pad = max(1e-6, 0.05 * (x_max - x_min if x_max > x_min else 1.0))
    y_pad = max(1e-6, 0.05 * (y_max - y_min if y_max > y_min else 1.0))

    fig, ax = plt.subplots(figsize=(6, 6))
    particles = ax.scatter([], [], s=18, c="#1f77b4", alpha=0.8, label="particles")
    gbest = ax.scatter([], [], s=120, c="#d62728", marker="*", label="global best")

    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right")

    variant_tag = variant if variant == "PSO" else f"GP-{variant}"
    if use_ml and variant != "PSO":
        variant_tag += "+ML"

    def update(frame_idx: int):
        particles.set_offsets(pos_2d[frame_idx])
        gbest.set_offsets(gbest_2d[frame_idx][None, :])
        ax.set_title(
            f"{variant_tag} | {func_name} | seed {seed} | "
            f"iter {iterations[frame_idx]} | best {gbest_values[frame_idx]:.4e}"
        )
        return particles, gbest

    anim = FuncAnimation(fig, update, frames=len(iterations), interval=1000 / args.fps, blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"saved animation to {out_path}")


if __name__ == "__main__":
    main()
