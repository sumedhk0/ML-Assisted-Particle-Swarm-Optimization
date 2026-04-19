"""Batched torch benchmark functions.

Every callable accepts a tensor of shape (..., D) and returns shape (...).
All functions expose a known `global_optimum` (for labeling training data) and
default `bounds` matching the original CPU implementation.
"""
import math
import torch


class Function:
    name: str
    bounds: tuple        # (low, high)
    global_optimum: torch.Tensor  # shape (D,)

    def __init__(self, dim: int, device: str = "cpu"):
        self.dim = dim
        self.device = device

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def to(self, device: str) -> "Function":
        self.device = device
        self.global_optimum = self.global_optimum.to(device)
        return self

    def domain_diameter(self) -> float:
        low, high = self.bounds
        return math.sqrt(self.dim) * (high - low)


class Sphere(Function):
    name = "sphere"
    bounds = (-5.12, 5.12)

    def __init__(self, dim, device="cpu"):
        super().__init__(dim, device)
        self.global_optimum = torch.zeros(dim, device=device)

    def __call__(self, x):
        return (x ** 2).sum(dim=-1)


class Rastrigin(Function):
    name = "rastrigin"
    bounds = (-5.12, 5.12)

    def __init__(self, dim, device="cpu"):
        super().__init__(dim, device)
        self.global_optimum = torch.zeros(dim, device=device)

    def __call__(self, x):
        return 10.0 * self.dim + (x ** 2 - 10.0 * torch.cos(2.0 * math.pi * x)).sum(dim=-1)


class Ackley(Function):
    name = "ackley"
    bounds = (-5.0, 5.0)

    def __init__(self, dim, device="cpu"):
        super().__init__(dim, device)
        self.global_optimum = torch.zeros(dim, device=device)

    def __call__(self, x):
        d = self.dim
        sum_sq = (x ** 2).sum(dim=-1)
        sum_cos = torch.cos(2.0 * math.pi * x).sum(dim=-1)
        return -20.0 * torch.exp(-0.2 * torch.sqrt(sum_sq / d)) \
               - torch.exp(sum_cos / d) + math.e + 20.0


class Griewank(Function):
    name = "griewank"
    bounds = (-600.0, 600.0)

    def __init__(self, dim, device="cpu"):
        super().__init__(dim, device)
        self.global_optimum = torch.zeros(dim, device=device)
        self._sqrt_idx = torch.sqrt(torch.arange(1, dim + 1, device=device, dtype=torch.float32))

    def to(self, device):
        super().to(device)
        self._sqrt_idx = self._sqrt_idx.to(device)
        return self

    def __call__(self, x):
        sum_sq = (x ** 2).sum(dim=-1) / 4000.0
        prod_cos = torch.cos(x / self._sqrt_idx).prod(dim=-1)
        return 1.0 + sum_sq - prod_cos


class Rosenbrock(Function):
    name = "rosenbrock"
    bounds = (-5.0, 10.0)

    def __init__(self, dim, device="cpu"):
        super().__init__(dim, device)
        self.global_optimum = torch.ones(dim, device=device)

    def __call__(self, x):
        a = x[..., 1:] - x[..., :-1] ** 2
        b = 1.0 - x[..., :-1]
        return (100.0 * a ** 2 + b ** 2).sum(dim=-1)


class GaussianMixture(Function):
    """Multimodal synthetic landscape for training-data diversity.

    f(x) = -sum_k depth_k * exp(-||x - center_k||^2 / (2 * width_k^2))
    Mode 0 is deepest by construction, so its center is the known global minimum.
    """
    name = "gaussian_mixture"
    bounds = (-5.0, 5.0)

    def __init__(self, dim, seed, n_modes=8, device="cpu"):
        super().__init__(dim, device)
        low, high = self.bounds
        gen = torch.Generator(device="cpu").manual_seed(seed)

        centers = torch.empty(n_modes, dim).uniform_(low, high, generator=gen)
        depths = torch.empty(n_modes).uniform_(0.3, 1.0, generator=gen)
        depths[0] = 1.5   # guarantees mode 0 is the global minimum
        widths = torch.empty(n_modes).uniform_(0.5, 1.5, generator=gen)

        self.centers = centers.to(device)
        self.depths = depths.to(device)
        self.widths = widths.to(device)
        self.global_optimum = self.centers[0].clone()

    def to(self, device):
        super().to(device)
        self.centers = self.centers.to(device)
        self.depths = self.depths.to(device)
        self.widths = self.widths.to(device)
        return self

    def __call__(self, x):
        # x: (..., D). Broadcast against centers (K, D) -> (..., K, D) -> (..., K)
        diffs = x.unsqueeze(-2) - self.centers
        sq_dists = (diffs ** 2).sum(dim=-1)
        contributions = self.depths * torch.exp(-sq_dists / (2.0 * self.widths ** 2))
        return -contributions.sum(dim=-1)


_STANDARD = {
    "sphere": Sphere,
    "rastrigin": Rastrigin,
    "ackley": Ackley,
    "griewank": Griewank,
    "rosenbrock": Rosenbrock,
}


def get_function(name: str, dim: int, device: str = "cpu", **kwargs) -> Function:
    if name in _STANDARD:
        return _STANDARD[name](dim, device)
    if name == "gaussian_mixture":
        return GaussianMixture(dim, device=device, **kwargs)
    raise ValueError(f"Unknown function: {name}. Available: {list(_STANDARD) + ['gaussian_mixture']}")


if __name__ == "__main__":
    import numpy as np

    dim = 5
    for name in _STANDARD:
        f = get_function(name, dim)
        # Check: f(global_optimum) ≈ 0 for all standard benchmarks
        z = f(f.global_optimum).item()
        print(f"{name:12s}  f(optimum) = {z:.3e}")
        assert abs(z) < 1e-6, f"{name} optimum not zero"

    # Batched shape check
    f = get_function("rastrigin", 10)
    x = torch.randn(7, 10)
    out = f(x)
    assert out.shape == (7,), f"expected (7,) got {out.shape}"
    print("batched shape OK")

    # Compare against naive numpy for a few points
    def rastrigin_np(x):
        return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))
    x_np = np.random.randn(10)
    x_torch = torch.from_numpy(x_np).float()
    diff = abs(float(f(x_torch)) - rastrigin_np(x_np))
    print(f"rastrigin torch vs numpy diff = {diff:.3e}")
    assert diff < 1e-4

    # GaussianMixture sanity
    gm = GaussianMixture(dim=3, seed=42)
    at_opt = gm(gm.global_optimum).item()
    elsewhere = gm(torch.zeros(3) + 10.0).item()
    print(f"GM  f(optimum)={at_opt:.3f}  f(far)={elsewhere:.3f}")
    assert at_opt < elsewhere, "global optimum should be lower than distant point"
    print("all function.py checks passed")
