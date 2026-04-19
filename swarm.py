"""Tensorized particle swarm.

Replaces the per-object Particle model with one Swarm holding all state as
(N, D) tensors on a single device. All steps are single-batch GPU ops.
"""
import torch

from function import Function


class Swarm:
    def __init__(self, n_particles: int, dim: int, function: Function,
                 device: str = "cpu", history_len: int = 5, seed: int | None = None):
        self.n = n_particles
        self.dim = dim
        self.function = function
        self.device = device
        self.bounds = function.bounds

        if seed is not None:
            gen = torch.Generator(device=device).manual_seed(seed)
        else:
            gen = None

        low, high = self.bounds
        self.positions = torch.empty(n_particles, dim, device=device).uniform_(low, high, generator=gen)
        self.velocities = torch.zeros(n_particles, dim, device=device)

        values = self.function(self.positions)
        self.pbest = self.positions.clone()
        self.pbest_values = values.clone()

        best_idx = int(torch.argmin(values))
        self.global_best = self.pbest[best_idx].clone()
        self.global_best_value = float(values[best_idx])

        # Per-particle history for ML features
        self.pbest_plateau = torch.zeros(n_particles, device=device, dtype=torch.long)
        self.value_history = values.unsqueeze(1).repeat(1, history_len)  # (N, history_len)
        self._history_idx = 0

        # Last evaluation values (kept for feature extraction without re-evaluating)
        self.last_values = values.clone()

    # ----- movement primitives -----

    def step_standard(self, omega: float, phi_p: float, phi_g: float, constriction: float = 0.729):
        r1 = torch.rand(self.n, self.dim, device=self.device)
        r2 = torch.rand(self.n, self.dim, device=self.device)
        self.velocities = constriction * (
            omega * self.velocities
            + phi_p * r1 * (self.pbest - self.positions)
            + phi_g * r2 * (self.global_best - self.positions)
        )
        self.positions = self.positions + self.velocities
        self._post_move_update()

    def step_variant_a(self, omega: float, phi_p: float, phi_g: float,
                       phi_h: float, h: torch.Tensor):
        r1 = torch.rand(self.n, self.dim, device=self.device)
        r2 = torch.rand(self.n, self.dim, device=self.device)
        r3 = torch.rand(self.n, self.dim, device=self.device)
        self.velocities = (
            omega * self.velocities
            + phi_p * r1 * (self.pbest - self.positions)
            + phi_g * r2 * (self.global_best - self.positions)
            + phi_h * r3 * (h - self.positions)
        )
        self.positions = self.positions + self.velocities
        self._post_move_update()

    def reset_particles(self, indices: torch.Tensor, new_positions: torch.Tensor):
        """Teleport selected particles to new positions. Velocity re-initialized N(0, 1)."""
        self.positions[indices] = new_positions
        self.velocities[indices] = torch.randn(len(indices), self.dim, device=self.device)

        new_values = self.function(new_positions)
        improved_mask = new_values < self.pbest_values[indices]
        if improved_mask.any():
            improved_idx = indices[improved_mask]
            self.pbest[improved_idx] = new_positions[improved_mask]
            self.pbest_values[improved_idx] = new_values[improved_mask]

        # Fresh plateau counter after relocation
        self.pbest_plateau[indices] = 0

        # Update global best if any new point beats it
        new_best = float(torch.min(new_values))
        if new_best < self.global_best_value:
            local_idx = int(torch.argmin(new_values))
            self.global_best = new_positions[local_idx].clone()
            self.global_best_value = new_best

        # Write the new values into last_values / history at those indices
        self.last_values[indices] = new_values

    def worst_particle_index(self) -> int:
        return int(torch.argmax(self.last_values))

    # ----- internals -----

    def _post_move_update(self):
        values = self.function(self.positions)
        self.last_values = values

        # Update personal bests
        improved = values < self.pbest_values
        if improved.any():
            self.pbest[improved] = self.positions[improved]
            self.pbest_values[improved] = values[improved]

        # Plateau counter: +1 for all, 0 for improved
        self.pbest_plateau = self.pbest_plateau + 1
        self.pbest_plateau[improved] = 0

        # Ring buffer of recent values
        self.value_history[:, self._history_idx] = values
        self._history_idx = (self._history_idx + 1) % self.value_history.shape[1]

        # Global best
        best_local = int(torch.argmin(values))
        if float(values[best_local]) < self.global_best_value:
            self.global_best = self.positions[best_local].clone()
            self.global_best_value = float(values[best_local])


if __name__ == "__main__":
    # Convergence check: plain PSO on sphere should reach near-zero in 100 iters.
    from function import Sphere

    torch.manual_seed(0)
    f = Sphere(dim=2)
    swarm = Swarm(n_particles=10, dim=2, function=f, seed=0)

    omega = 0.7
    phi_p = 2.05
    phi_g = 2.05
    for _ in range(100):
        swarm.step_standard(omega=omega, phi_p=phi_p, phi_g=phi_g, constriction=0.729)

    print(f"global best after 100 iters: {swarm.global_best_value:.3e}")
    assert swarm.global_best_value < 1e-3, "sphere convergence failed"

    # reset_particles sanity
    idx = torch.tensor([0, 1, 2])
    new_pos = torch.zeros(3, 2)  # exact sphere optimum
    swarm.reset_particles(idx, new_pos)
    assert swarm.global_best_value < 1e-10
    print("swarm.py checks passed")
