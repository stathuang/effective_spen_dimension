"""Sequence-model generators used by Figures 1 and 2."""

from __future__ import annotations

import numpy as np


def generate_lambda(d: int, gamma: float) -> np.ndarray:
    """Generate eigenvalues lambda_j = j^{-gamma}, j=1,...,d."""

    if d <= 0:
        raise ValueError("d must be positive")
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    indices = np.arange(1, d + 1, dtype=float)
    return indices ** (-gamma)


def generate_sparse_misaligned_theta(
    q: float,
    p: float,
    J: int,
    d: int,
    *,
    C: float = 5.0,
    background: float = 0.0,
) -> np.ndarray:
    """Generate the sparse q-misaligned signal used in the manuscript.

    Signal coordinates are at ``floor(j**q) - 1`` for ``j=1,...,J``.  Off-support
    entries are exactly ``background``; the manuscript contract uses
    ``background=0``.
    """

    if q < 1:
        raise ValueError("q must be at least 1")
    if p <= 0:
        raise ValueError("p must be positive")
    if J <= 0 or d <= 0:
        raise ValueError("J and d must be positive")

    theta = np.full(d, float(background), dtype=float)
    indices = support_indices(q, J, d)
    for j, idx in enumerate(indices, start=1):
        theta[idx] = C * j ** (-(p + 1) / 2)
    return theta


def support_indices(q: float, J: int, d: int) -> np.ndarray:
    """Return zero-based sparse support indices floor(j**q)-1 for j=1,...,J."""

    if q < 1:
        raise ValueError("q must be at least 1")
    if J <= 0 or d <= 0:
        raise ValueError("J and d must be positive")
    indices = np.array([int(j**q) - 1 for j in range(1, J + 1)], dtype=int)
    if np.any(indices < 0) or np.any(indices >= d):
        raise ValueError(f"q={q:g} and J={J} require d >= floor(J^q)")
    if len(set(indices.tolist())) != J:
        raise ValueError("q and J produced duplicate rounded signal indices")
    return indices


def validate_zero_off_support(theta: np.ndarray, support: np.ndarray) -> None:
    """Validate that all coordinates outside support are exactly zero."""

    theta_arr = np.asarray(theta, dtype=float)
    support_arr = np.asarray(support, dtype=int)
    if theta_arr.ndim != 1:
        raise ValueError("theta must be one-dimensional")
    if np.any(support_arr < 0) or np.any(support_arr >= theta_arr.size):
        raise ValueError("support contains indices outside theta")
    mask = np.ones(theta_arr.size, dtype=bool)
    mask[support_arr] = False
    if np.any(theta_arr[mask] != 0.0):
        raise ValueError("sequence generator returned nonzero off-support entries")


def gaussian_observation(
    theta: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate z = theta + N(0, sigma^2 I)."""

    if sigma < 0:
        raise ValueError("sigma must be nonnegative")
    return np.asarray(theta, dtype=float) + rng.normal(0.0, sigma, size=len(theta))
