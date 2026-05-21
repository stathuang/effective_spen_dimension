"""Reusable over-parameterized gradient-flow helpers for Figures 1 and 2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OPGFTrajectory:
    iterations: np.ndarray
    lambda_path: np.ndarray


def compute_gradients(a: np.ndarray, b: np.ndarray, beta: np.ndarray, y: np.ndarray, D: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gradient of 0.5 ||y - a b^D beta||^2."""

    theta = a * (b**D) * beta
    error = y - theta
    grad_beta = -error * a * (b**D)
    grad_a = -error * (b**D) * beta
    grad_b = -error * a * D * (b ** (D - 1)) * beta if D > 0 else np.zeros_like(b)
    return grad_a, grad_b, grad_beta


def run_opgf(
    *,
    y: np.ndarray,
    lambda0: np.ndarray,
    n: int,
    D: int,
    learning_rate: float,
    num_iterations: int,
    record_iterations: np.ndarray,
) -> OPGFTrajectory:
    """Run the diagonal OP-GF dynamics for a fixed observation vector.

    This is the core trajectory API used by the Figure 1 and Figure 2 scripts.
    It records the learned eigenvalue sequence at the requested iterations,
    including iteration 0 before any gradient step when requested.
    """

    y_arr = np.asarray(y, dtype=float)
    lambda_arr = np.asarray(lambda0, dtype=float)
    if y_arr.ndim != 1 or lambda_arr.ndim != 1 or y_arr.size != lambda_arr.size:
        raise ValueError("y and lambda0 must be one-dimensional arrays with the same length")
    if n <= 0 or D < 0 or learning_rate <= 0 or num_iterations <= 0:
        raise ValueError("invalid OP-GF run parameters")

    record_points = np.unique(np.asarray(record_iterations, dtype=int))
    record_points = record_points[(record_points >= 0) & (record_points <= num_iterations)]
    if record_points.size == 0:
        raise ValueError("record_iterations must contain at least one valid iteration")
    record_set = set(int(x) for x in record_points)

    a_est = np.sqrt(lambda_arr)
    b_est = np.full(y_arr.size, n ** (-1 / (2 * (D + 2))))
    beta_est = np.zeros(y_arr.size)
    iterations: list[int] = []
    lambda_path: list[np.ndarray] = []

    for iteration in range(num_iterations + 1):
        if iteration in record_set:
            iterations.append(iteration)
            lambda_path.append((a_est * (b_est**D)) ** 2)
        if iteration == num_iterations:
            break
        grad_a, grad_b, grad_beta = compute_gradients(a_est, b_est, beta_est, y_arr, D)
        a_est -= learning_rate * grad_a
        b_est = np.abs(b_est - learning_rate * grad_b)
        beta_est -= learning_rate * grad_beta

    return OPGFTrajectory(
        iterations=np.asarray(iterations, dtype=int),
        lambda_path=np.asarray(lambda_path),
    )
