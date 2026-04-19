"""Memory of informative GP observations with chi-selection filtering.

Only points that surprise the current GP (outside rho * sigma of the predicted
mean) get added. A soft cap drops the oldest points to keep GP fit O(n^3) bounded.
"""
import torch


class MemoryManager:
    def __init__(self, rho: float = 1.15, cap: int = 2000):
        self.rho = rho
        self.cap = cap
        self.X: torch.Tensor | None = None
        self.y: torch.Tensor | None = None

    def initialize(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X.clone()
        self.y = y.clone()
        self._enforce_cap()

    def update(self, gp, X_new: torch.Tensor, y_new: torch.Tensor):
        mean, std = gp.predict(X_new, return_std=True)
        mask = (y_new - mean).abs() > self.rho * std
        if bool(mask.any()):
            self.X = torch.cat([self.X, X_new[mask]])
            self.y = torch.cat([self.y, y_new[mask]])
            self._enforce_cap()

    def get_training_data(self):
        return self.X, self.y

    @property
    def size(self) -> int:
        return 0 if self.y is None else int(self.y.shape[0])

    def _enforce_cap(self):
        if self.cap is not None and self.size > self.cap:
            self.X = self.X[-self.cap:]
            self.y = self.y[-self.cap:]
