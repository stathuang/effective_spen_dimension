"""Effective Span Dimension utilities.

The public functions in this module implement the paper definition

    H(k) = (1 / k) * sum_{i=k+1}^d theta_{pi_i}^2,  k = 1, ..., d,

where ``pi`` orders coordinates by decreasing eigenvalue.  Callers pass the
unsquared signal coefficients; this module performs the squaring exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ESDResult:
    """Container for an ESD value and the H-profile that produced it."""

    d_dagger: int
    h_values: np.ndarray
    order: np.ndarray


def _as_1d_float_array(values: Iterable[float], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if arr.size == 0:
        raise ValueError(f"{name} must be nonempty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def spectrum_order(lambda_vals: Iterable[float]) -> np.ndarray:
    """Return a deterministic decreasing-spectrum order.

    Ties are broken by the original coordinate index.  This implements the
    deterministic tie-breaking convention used for reproducible experiments.
    """

    lambdas = _as_1d_float_array(lambda_vals, "lambda_vals")
    return np.lexsort((np.arange(lambdas.size), -lambdas))


def compute_H(theta: Iterable[float], lambda_vals: Iterable[float]) -> np.ndarray:
    """Compute the paper's trade-off function H(k), k=1,...,d.

    Parameters
    ----------
    theta:
        Unsquared signal coefficients.
    lambda_vals:
        Eigenvalues used only to order coordinates.
    """

    theta_arr = _as_1d_float_array(theta, "theta")
    lambda_arr = _as_1d_float_array(lambda_vals, "lambda_vals")
    if theta_arr.size != lambda_arr.size:
        raise ValueError("theta and lambda_vals must have the same length")

    ordered_theta = theta_arr[spectrum_order(lambda_arr)]
    suffix = np.cumsum(ordered_theta[::-1] ** 2)[::-1]
    tail_after_k = np.empty_like(suffix)
    tail_after_k[:-1] = suffix[1:]
    tail_after_k[-1] = 0.0
    return tail_after_k / np.arange(1, theta_arr.size + 1, dtype=float)


def compute_esd(theta: Iterable[float], lambda_vals: Iterable[float], sigma2: float) -> int:
    """Return the smallest k such that H(k) <= sigma2."""

    if not np.isfinite(sigma2) or sigma2 < 0:
        raise ValueError("sigma2 must be a finite nonnegative value")
    h_values = compute_H(theta, lambda_vals)
    feasible = np.nonzero(h_values <= sigma2)[0]
    if feasible.size:
        return int(feasible[0] + 1)
    return int(h_values.size)


def compute_esd_result(
    theta: Iterable[float], lambda_vals: Iterable[float], sigma2: float
) -> ESDResult:
    """Return ESD plus H-profile and ordering for diagnostics."""

    lambda_arr = _as_1d_float_array(lambda_vals, "lambda_vals")
    order = spectrum_order(lambda_arr)
    h_values = compute_H(theta, lambda_arr)
    feasible = np.nonzero(h_values <= sigma2)[0]
    d_dagger = int(feasible[0] + 1) if feasible.size else int(h_values.size)
    return ESDResult(d_dagger=d_dagger, h_values=h_values, order=order)


def pc_oracle_error(
    y: Iterable[float],
    theta_star: Iterable[float],
    lambda_vals: Iterable[float],
    sigma2: float,
) -> tuple[float, int]:
    """Squared error of the ESD-tuned principal-coordinate estimator."""

    y_arr = _as_1d_float_array(y, "y")
    theta_arr = _as_1d_float_array(theta_star, "theta_star")
    lambda_arr = _as_1d_float_array(lambda_vals, "lambda_vals")
    if y_arr.size != theta_arr.size or theta_arr.size != lambda_arr.size:
        raise ValueError("y, theta_star, and lambda_vals must have the same length")

    d_dagger = compute_esd(theta_arr, lambda_arr, sigma2)
    theta_hat = np.zeros_like(theta_arr)
    theta_hat[spectrum_order(lambda_arr)[:d_dagger]] = y_arr[spectrum_order(lambda_arr)[:d_dagger]]
    return float(np.sum((theta_hat - theta_arr) ** 2)), int(d_dagger)


def as_numpy_1d(values: object, name: str = "values") -> np.ndarray:
    """Convert NumPy arrays or torch tensors to a one-dimensional NumPy array."""

    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return _as_1d_float_array(values, name)

