"""RKHS helpers for Figures 4 and 5."""

from __future__ import annotations

import numpy as np

from esd_experiments.core.esd import as_numpy_1d, compute_esd


def cosine_features(x: np.ndarray, J: int) -> np.ndarray:
    """Periodic cosine basis phi_j(x)=sqrt(2) cos(2 pi j x), j=1,...,J."""

    x_arr = np.asarray(x, dtype=float)
    j = np.arange(1, J + 1, dtype=float)
    return np.sqrt(2.0) * np.cos(2.0 * np.pi * x_arr[:, np.newaxis] * j[np.newaxis, :])


def estimate_sup_norm_from_cosine_coeffs(theta: np.ndarray, num_grid: int = 10000) -> float:
    """Grid estimate of ||f^*||_infty for cosine coefficients."""

    x_grid = np.linspace(0.0, 1.0, num_grid)
    Phi_grid = cosine_features(x_grid, len(theta))
    return float(np.max(np.abs(Phi_grid @ np.asarray(theta, dtype=float))))


def effective_noise_from_sup_norm(sigma0_sq: float, f_sup_norm_sq: float, n: int) -> float:
    """sigma_eff^2 = (sigma0^2 + ||f^*||_infty^2) / n."""

    if n <= 0:
        raise ValueError("n must be positive")
    if sigma0_sq < 0 or f_sup_norm_sq < 0:
        raise ValueError("variance terms must be nonnegative")
    return (float(sigma0_sq) + float(f_sup_norm_sq)) / n


def pathwise_esd_from_eigendecomposition(theta_coeffs: object, eigenvalues: object, sigma2: float) -> int:
    """Pathwise ESD wrapper used by Figure 5.

    This function accepts NumPy arrays or torch tensors and delegates the ESD
    calculation to the shared NumPy helper.
    """

    theta_np = as_numpy_1d(theta_coeffs, "theta_coeffs")
    eigen_np = as_numpy_1d(eigenvalues, "eigenvalues")
    return compute_esd(theta_np, eigen_np, sigma2)


def truncation_loss_matrix(theta_hat_sorted: np.ndarray, theta_sorted: np.ndarray) -> np.ndarray:
    """Loss for all truncation levels k=1,...,J without building a dense matrix."""

    squared = (theta_hat_sorted - theta_sorted) ** 2
    prefix_est_error = np.cumsum(squared)
    theta_tail_sq = np.cumsum(theta_sorted[::-1] ** 2)[::-1]
    tail_after_k = np.empty_like(theta_tail_sq)
    tail_after_k[:-1] = theta_tail_sq[1:]
    tail_after_k[-1] = 0.0
    return prefix_est_error + tail_after_k

