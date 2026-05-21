"""Fixed-design linear-model helpers for Figure 3."""

from __future__ import annotations

import numpy as np
from scipy import linalg


def make_design(n: int, p: int, par: float | str | np.ndarray = 0.95, seed: int | None = None) -> np.ndarray:
    """Create an n by p Gaussian design with diagonal covariance."""

    rng = np.random.default_rng(seed)
    if isinstance(par, str):
        if par != "log":
            raise ValueError("string par must be 'log'")
        eig_vals = 1 / np.log(np.arange(1, p + 1) + 1)
    elif np.isscalar(par):
        eig_vals = float(par) ** np.arange(p)
    else:
        eig_vals = np.asarray(par, dtype=float)
        if eig_vals.shape != (p,):
            raise ValueError("explicit eigenvalue vector must have length p")
    return rng.normal(size=(n, p)) * np.sqrt(eig_vals)


def make_beta(p: int, decay_type: str = "power", power: float = 1.0) -> np.ndarray:
    """Generate a dense signal vector."""

    idx = np.arange(1, p + 1, dtype=float)
    if decay_type == "power":
        return idx ** (-power)
    if decay_type == "log":
        return 1 / np.log(idx + 1)
    raise ValueError("decay_type must be 'power' or 'log'")


def cache_svd(X: np.ndarray) -> dict[str, np.ndarray | int]:
    """SVD of X/sqrt(n), matching the manuscript linear-model scaling."""

    u, d, vt = linalg.svd(X / np.sqrt(X.shape[0]), full_matrices=False)
    return {"U": u, "V": vt.T, "d": d, "r": len(d)}


def translate_theta_lambda(svd_obj: dict[str, np.ndarray | int], beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Translate beta into theta/lambda coordinates."""

    d = np.asarray(svd_obj["d"], dtype=float)
    V = np.asarray(svd_obj["V"], dtype=float)
    theta = d * (V.T @ beta)
    lambda_vals = d**2
    return theta, lambda_vals


def build_A_diag_exp(p: int, alpha: float) -> np.ndarray:
    """Smooth exponential diagonal stretch with t_i in [-1/2, 1/2]."""

    t_vec = np.linspace(-0.5, 0.5, p)
    return np.diag(np.exp(alpha * t_vec))


def apply_alignment(X: np.ndarray, beta: np.ndarray, A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply X -> XA and beta -> A^{-1} beta."""

    return X @ A, linalg.solve(A, beta)


def pcr_fit_all(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return beta_hat for k=1,...,rank as columns."""

    u, d, vt = linalg.svd(X, full_matrices=False)
    proj = (u.T @ y) / d
    contrib = vt.T * proj[np.newaxis, :]
    return np.cumsum(contrib, axis=1)


def prediction_mse_by_column(X: np.ndarray, beta_mat: np.ndarray, beta_true: np.ndarray) -> np.ndarray:
    """Prediction MSE for every beta_hat column."""

    pred_err = X @ (beta_mat - beta_true[:, np.newaxis])
    return np.mean(pred_err**2, axis=0)

