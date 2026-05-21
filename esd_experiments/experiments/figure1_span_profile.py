"""Figure 1 span-profile experiment logic."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from esd_experiments.core.esd import compute_esd
from esd_experiments.core.opgf import run_opgf
from esd_experiments.core.sequence import (
    generate_lambda,
    generate_sparse_misaligned_theta,
    support_indices,
    validate_zero_off_support,
)


DEFAULT_Q_VALUES = (1.0, 1.5, 2.0, 3.0)
SIGNAL_SCALE = 5.0


@dataclass(frozen=True)
class Figure1Config:
    seed: int = 2026
    fast: bool = False
    n: int = 10_000
    sigma0: float = 1.0
    d: int = 5_000
    J: int = 15
    p: float = 2.5
    gamma: float = 1.0
    D: int = 0
    q_values: tuple[float, ...] = DEFAULT_Q_VALUES
    learning_rate: float = 1e-2
    num_iterations: int | None = None
    snapshot_stride: int = 2_000
    tau_min: float = 1e-7
    tau_max: float = 1e-2
    num_tau: int = 20


def default_num_iterations(n: int, D: int, learning_rate: float) -> int:
    return int(n ** ((D + 1) / (D + 2)) / learning_rate)


def build_config(
    *,
    seed: int = 2026,
    fast: bool = False,
    n: int = 10_000,
    sigma0: float = 1.0,
    d: int = 5_000,
    J: int = 15,
    p: float = 2.5,
    gamma: float = 1.0,
    D: int = 0,
    q_values: tuple[float, ...] = DEFAULT_Q_VALUES,
    learning_rate: float = 1e-2,
    num_iterations: int | None = None,
    snapshot_stride: int = 2_000,
    tau_min: float = 1e-7,
    tau_max: float = 1e-2,
    num_tau: int = 20,
) -> Figure1Config:
    if fast:
        n = min(n, 1_000)
        d = min(d, 800)
        J = min(J, 8)
        snapshot_stride = min(snapshot_stride, 500)
        num_tau = min(num_tau, 12)
    if num_iterations is None:
        num_iterations = default_num_iterations(n, D, learning_rate)
    if fast:
        num_iterations = min(num_iterations, 3_000)
    return Figure1Config(
        seed=seed,
        fast=fast,
        n=n,
        sigma0=sigma0,
        d=d,
        J=J,
        p=p,
        gamma=gamma,
        D=D,
        q_values=tuple(q_values),
        learning_rate=learning_rate,
        num_iterations=num_iterations,
        snapshot_stride=snapshot_stride,
        tau_min=tau_min,
        tau_max=tau_max,
        num_tau=num_tau,
    )


def snapshot_iterations(num_iterations: int, stride: int) -> np.ndarray:
    if num_iterations <= 0:
        raise ValueError("num_iterations must be positive")
    if stride <= 0:
        raise ValueError("snapshot_stride must be positive")
    points = np.arange(0, num_iterations, stride, dtype=int)
    if points.size == 0:
        points = np.array([0], dtype=int)
    return points


def compute_profiles(
    theta: np.ndarray, lambda_path: np.ndarray, tau_grid: np.ndarray
) -> np.ndarray:
    profiles = np.empty((lambda_path.shape[0], tau_grid.shape[0]), dtype=float)
    for row, lambdas in enumerate(lambda_path):
        profiles[row, :] = [compute_esd(theta, lambdas, tau) for tau in tau_grid]
    return profiles


def run_experiment(config: Figure1Config) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    sigma = config.sigma0 / np.sqrt(config.n)
    tau_grid = np.logspace(np.log10(config.tau_min), np.log10(config.tau_max), config.num_tau)
    record_iterations = snapshot_iterations(config.num_iterations, config.snapshot_stride)
    rng = np.random.default_rng(config.seed)
    lambda0 = generate_lambda(config.d, config.gamma)

    q_values = np.asarray(config.q_values, dtype=float)
    profiles_by_q = []
    lambda_paths_by_q = []
    theta_by_q = []
    y_by_q = []
    iterations_ref = None

    for q in q_values:
        theta = generate_sparse_misaligned_theta(
            q=q, p=config.p, J=config.J, d=config.d, C=SIGNAL_SCALE
        )
        validate_zero_off_support(theta, support_indices(q, config.J, config.d))
        y = theta + rng.normal(0.0, sigma, size=config.d)
        trajectory = run_opgf(
            y=y,
            lambda0=lambda0,
            n=config.n,
            D=config.D,
            learning_rate=config.learning_rate,
            num_iterations=config.num_iterations,
            record_iterations=record_iterations,
        )
        if iterations_ref is None:
            iterations_ref = trajectory.iterations
        elif not np.array_equal(iterations_ref, trajectory.iterations):
            raise RuntimeError("OPGF core returned inconsistent Figure 1 time grids")
        profiles = compute_profiles(theta, trajectory.lambda_path, tau_grid)

        profiles_by_q.append(profiles)
        lambda_paths_by_q.append(trajectory.lambda_path)
        theta_by_q.append(theta)
        y_by_q.append(y)

    iterations_arr = np.asarray(iterations_ref, dtype=int)
    arrays = {
        "q_values": q_values,
        "tau_grid": tau_grid,
        "iterations": iterations_arr,
        "time_values": iterations_arr * config.learning_rate,
        "profiles": np.asarray(profiles_by_q, dtype=float),
        "lambda_paths": np.asarray(lambda_paths_by_q, dtype=float),
        "theta": np.asarray(theta_by_q, dtype=float),
        "observations": np.asarray(y_by_q, dtype=float),
        "lambda0": lambda0,
    }
    metadata = {
        "figure": "figure1_span_profile",
        "seed": config.seed,
        "n": config.n,
        "sigma0": config.sigma0,
        "d": config.d,
        "J": config.J,
        "p": config.p,
        "gamma": config.gamma,
        "D": config.D,
        "C": SIGNAL_SCALE,
        "learning_rate": config.learning_rate,
        "num_iterations": config.num_iterations,
        "snapshot_stride": config.snapshot_stride,
        "tau_min": config.tau_min,
        "tau_max": config.tau_max,
        "num_tau": config.num_tau,
        "fast": config.fast,
    }
    return arrays, metadata
