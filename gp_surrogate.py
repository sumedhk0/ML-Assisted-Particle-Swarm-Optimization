"""GPyTorch-based Gaussian Process surrogate.

Uses an ExactGP with ARD-RBF kernel (one length-scale per dimension) plus a
learnable noise term. Hyperparameters are optimized via Adam on the marginal
log-likelihood. Acquisition searches run as batched multi-start Adam with
gradients flowing through GPyTorch.
"""
import torch
import gpytorch


class _ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, dim):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=dim)
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


class GPSurrogate:
    def __init__(self, dim: int, device: str = "cpu",
                 fit_iters: int = 100, fit_lr: float = 0.1):
        self.dim = dim
        self.device = device
        self.fit_iters = fit_iters
        self.fit_lr = fit_lr
        self.model = None
        self.likelihood = None
        self._y_mean = 0.0
        self._y_std = 1.0

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        X = X.to(self.device).float()
        y = y.to(self.device).float()

        self._y_mean = float(y.mean())
        self._y_std = float(y.std()) + 1e-8
        y_norm = (y - self._y_mean) / self._y_std

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.model = _ExactGPModel(X, y_norm, self.likelihood, self.dim).to(self.device)

        self.model.train()
        self.likelihood.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=self.fit_lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        prev_loss = float("inf")
        patience = 0
        for _ in range(self.fit_iters):
            opt.zero_grad()
            loss = -mll(self.model(X), y_norm)
            loss.backward()
            opt.step()
            if abs(prev_loss - loss.item()) < 1e-4:
                patience += 1
                if patience >= 10:
                    break
            else:
                patience = 0
            prev_loss = loss.item()

        self.model.eval()
        self.likelihood.eval()

    def _posterior(self, X: torch.Tensor):
        """Posterior mean & std WITH gradients (for acquisition)."""
        pred = self.likelihood(self.model(X))
        mean = pred.mean * self._y_std + self._y_mean
        std = pred.stddev * self._y_std
        return mean, std

    def predict(self, X: torch.Tensor, return_std: bool = True):
        X = X.to(self.device).float()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            mean, std = self._posterior(X)
        return (mean, std) if return_std else mean

    def _acquisition_opt(self, acq_fn, bounds, n_starts, iters, lr=0.05):
        low, high = bounds
        x = torch.empty(n_starts, self.dim, device=self.device).uniform_(low, high)
        x = x.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([x], lr=lr * (high - low))

        for _ in range(iters):
            opt.zero_grad()
            with gpytorch.settings.fast_pred_var():
                vals = acq_fn(x)
            loss = vals.sum()
            loss.backward()
            opt.step()
            with torch.no_grad():
                x.data.clamp_(low, high)

        with torch.no_grad():
            vals = acq_fn(x)
            best = int(torch.argmin(vals))
            return x[best].detach().clone(), float(vals[best])

    def find_minimum(self, bounds, n_starts: int = 50, iters: int = 60):
        def acq(x):
            return self._posterior(x)[0]
        return self._acquisition_opt(acq, bounds, n_starts, iters)

    def find_lcb_minimum(self, bounds, kappa: float = 1.6,
                         n_starts: int = 50, iters: int = 60):
        def acq(x):
            m, s = self._posterior(x)
            return m - kappa * s
        return self._acquisition_opt(acq, bounds, n_starts, iters)

    def find_max_uncertainty(self, bounds, n_starts: int = 50, iters: int = 60):
        def acq(x):
            _, s = self._posterior(x)
            return -s
        x, neg = self._acquisition_opt(acq, bounds, n_starts, iters)
        return x, -neg


if __name__ == "__main__":
    # Toy 2D: fit on a quadratic, confirm low uncertainty at training points
    torch.manual_seed(0)
    X = torch.randn(20, 2)
    y = (X ** 2).sum(dim=-1)

    gp = GPSurrogate(dim=2, fit_iters=150)
    gp.fit(X, y)

    mean, std = gp.predict(X)
    print(f"mean train std: {std.mean():.3e}  (should be small)")

    X_test = torch.randn(10, 2) * 3  # far from training
    _, std_test = gp.predict(X_test)
    print(f"mean test std:  {std_test.mean():.3e}  (should be larger)")
    assert std_test.mean() > std.mean()

    # Acquisition smoke test
    x_min, v = gp.find_minimum(bounds=(-3.0, 3.0), n_starts=20, iters=40)
    print(f"GP argmin: {x_min.tolist()}  predicted val: {v:.3f}")
    assert x_min.norm() < 1.5, "argmin should be near origin for quadratic"
    print("gp_surrogate.py checks passed")
