"""Figure 2 OP-GF error and ESD experiment logic."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np

from esd_experiments.core.esd import pc_oracle_error
from esd_experiments.core.opgf import run_opgf
from esd_experiments.core.results import load_result, stable_parameter_key
from esd_experiments.core.sequence import (
    generate_lambda,
    generate_sparse_misaligned_theta,
    support_indices,
    validate_zero_off_support,
)


DEFAULT_D_VALUES = (0, 1, 3)
SIGNAL_SCALE = 5.0


@dataclass(frozen=True)
class Figure2Config:
    seed: int = 2026
    fast: bool = False
    num_experiments: int = 20
    n: int = 10_000
    sigma0: float = 1.0
    d: int = 5_000
    J: int = 15
    p: float = 2.5
    q: float = 2.5
    gamma: float = 1.0
    D_values: tuple[int, ...] = DEFAULT_D_VALUES
    learning_rate: float = 1e-2
    num_iterations: int | None = None
    log_start: int = 200
    log_step: float = 0.05


def default_num_iterations(n: int, learning_rate: float) -> int:
    return int(n / learning_rate)


def build_config(
    *,
    seed: int = 2026,
    fast: bool = False,
    num_experiments: int = 20,
    n: int = 10_000,
    sigma0: float = 1.0,
    d: int = 5_000,
    J: int = 15,
    p: float = 2.5,
    q: float = 2.5,
    gamma: float = 1.0,
    D_values: tuple[int, ...] = DEFAULT_D_VALUES,
    learning_rate: float = 1e-2,
    num_iterations: int | None = None,
    log_start: int = 200,
    log_step: float = 0.05,
) -> Figure2Config:
    if fast:
        n = min(n, 1_000)
        d = min(d, 800)
        J = min(J, 8)
        num_experiments = min(num_experiments, 3)
        log_start = min(log_start, 20)
        log_step = max(log_step, 0.2)
    if num_iterations is None:
        num_iterations = default_num_iterations(n, learning_rate)
    if fast:
        num_iterations = min(num_iterations, 3_000)
    return Figure2Config(
        seed=seed,
        fast=fast,
        num_experiments=num_experiments,
        n=n,
        sigma0=sigma0,
        d=d,
        J=J,
        p=p,
        q=q,
        gamma=gamma,
        D_values=tuple(int(value) for value in D_values),
        learning_rate=learning_rate,
        num_iterations=num_iterations,
        log_start=log_start,
        log_step=log_step,
    )


def log_record_iterations(num_iterations: int, log_start: int, log_step: float) -> np.ndarray:
    if num_iterations <= 0:
        raise ValueError("num_iterations must be positive")
    if log_start <= 0:
        raise ValueError("log_start must be positive for a logarithmic x-axis")
    if log_step <= 0:
        raise ValueError("log_step must be positive")
    start = min(log_start, num_iterations)
    if start == num_iterations:
        return np.array([num_iterations], dtype=int)
    log_points = np.arange(np.log10(start), np.log10(num_iterations), log_step)
    points = np.unique(np.round(10**log_points).astype(int))
    points = points[(points >= 1) & (points <= num_iterations)]
    if points.size == 0 or points[-1] != num_iterations:
        points = np.unique(np.append(points, num_iterations))
    return points.astype(int)


def cache_key_payload(config: Figure2Config, record_iterations: np.ndarray) -> dict[str, object]:
    return {
        "n": config.n,
        "sigma0": config.sigma0,
        "d": config.d,
        "J": config.J,
        "p": config.p,
        "q": config.q,
        "gamma": config.gamma,
        "D_values": list(config.D_values),
        "C": SIGNAL_SCALE,
        "num_experiments": config.num_experiments,
        "seed": config.seed,
        "learning_rate": config.learning_rate,
        "num_iterations": config.num_iterations,
        "record_iterations": record_iterations.tolist(),
    }


def parameter_key(config: Figure2Config, record_iterations: np.ndarray) -> str:
    return stable_parameter_key(cache_key_payload(config, record_iterations))


def cache_file(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"figure2_opgf_error_esd_{key}.npz"


def result_matches_parameter_key(path: Path, key: str) -> bool:
    if not path.exists():
        return False
    try:
        _, metadata = load_result(path)
    except (OSError, ValueError, KeyError):
        return False
    return metadata.get("parameter_key") == key


def oracle_error_and_esd(
    y: np.ndarray, theta: np.ndarray, lambdas: np.ndarray, sigma2: float
) -> tuple[float, float]:
    sq_error, d_dagger = pc_oracle_error(y, theta, lambdas, sigma2)
    return float(sq_error), float(d_dagger)


def run_single_experiment(
    *,
    config: Figure2Config,
    D: int,
    experiment: int,
    record_iterations: np.ndarray,
    theta: np.ndarray,
    lambda0: np.ndarray,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(config.seed + experiment)
    y = theta + rng.normal(0.0, sigma, size=config.d)
    trajectory = run_opgf(
        y=y,
        lambda0=lambda0,
        n=config.n,
        D=D,
        learning_rate=config.learning_rate,
        num_iterations=config.num_iterations,
        record_iterations=record_iterations,
    )
    sq_errors = np.empty(trajectory.lambda_path.shape[0], dtype=float)
    d_daggers = np.empty(trajectory.lambda_path.shape[0], dtype=float)
    for row, lambdas in enumerate(trajectory.lambda_path):
        sq_errors[row], d_daggers[row] = oracle_error_and_esd(
            y, theta, lambdas, sigma2=sigma**2
        )
    return trajectory.iterations, sq_errors, d_daggers


def _run_single_task(
    task: tuple[
        Figure2Config,
        int,
        int,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
    ]
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray, float]:
    config, D, experiment, record_iterations, theta, lambda0, sigma = task
    start_time = time.perf_counter()
    iterations, sq_errors, d_daggers = run_single_experiment(
        config=config,
        D=D,
        experiment=experiment,
        record_iterations=record_iterations,
        theta=theta,
        lambda0=lambda0,
        sigma=sigma,
    )
    elapsed = time.perf_counter() - start_time
    return D, experiment, iterations, sq_errors, d_daggers, elapsed


def run_experiment(
    config: Figure2Config,
    record_iterations: np.ndarray,
    *,
    workers: int = 1,
    verbose: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    if config.num_experiments <= 0:
        raise ValueError("num_experiments must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")

    sigma = config.sigma0 / np.sqrt(config.n)
    theta = generate_sparse_misaligned_theta(
        q=config.q, p=config.p, J=config.J, d=config.d, C=SIGNAL_SCALE
    )
    validate_zero_off_support(theta, support_indices(config.q, config.J, config.d))
    lambda0 = generate_lambda(config.d, config.gamma)

    d_values = np.asarray(config.D_values, dtype=int)
    all_time_points = []
    sq_errors_by_D = []
    d_daggers_by_D = []
    results: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    if workers == 1:
        for D in d_values:
            for experiment in range(config.num_experiments):
                start_time = time.perf_counter()
                if verbose:
                    print(
                        f"Figure 2 D={int(D)} experiment "
                        f"{experiment + 1}/{config.num_experiments} started",
                        flush=True,
                    )
                iterations, sq_errors, d_daggers = run_single_experiment(
                    config=config,
                    D=int(D),
                    experiment=experiment,
                    record_iterations=record_iterations,
                    theta=theta,
                    lambda0=lambda0,
                    sigma=sigma,
                )
                results[(int(D), experiment)] = (iterations, sq_errors, d_daggers)
                if verbose:
                    elapsed = time.perf_counter() - start_time
                    print(
                        f"Figure 2 D={int(D)} experiment "
                        f"{experiment + 1}/{config.num_experiments} finished "
                        f"in {elapsed:.1f}s",
                        flush=True,
                    )
    else:
        tasks = [
            (
                config,
                int(D),
                experiment,
                record_iterations,
                theta,
                lambda0,
                sigma,
            )
            for D in d_values
            for experiment in range(config.num_experiments)
        ]
        if verbose:
            print(
                f"Figure 2 parallel run: {len(tasks)} trajectories "
                f"with {workers} workers",
                flush=True,
            )
        completed = 0
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_run_single_task, task) for task in tasks]
            for future in as_completed(futures):
                D, experiment, iterations, sq_errors, d_daggers, elapsed = future.result()
                results[(int(D), experiment)] = (iterations, sq_errors, d_daggers)
                completed += 1
                if verbose:
                    print(
                        f"Figure 2 D={int(D)} experiment "
                        f"{experiment + 1}/{config.num_experiments} finished "
                        f"in {elapsed:.1f}s ({completed}/{len(tasks)})",
                        flush=True,
                    )

    for D in d_values:
        errors_for_D = []
        esd_for_D = []
        iterations_for_D = None
        for experiment in range(config.num_experiments):
            iterations, sq_errors, d_daggers = results[(int(D), experiment)]
            if iterations_for_D is None:
                iterations_for_D = iterations
            elif not np.array_equal(iterations_for_D, iterations):
                raise RuntimeError("OPGF core returned inconsistent time grids")
            errors_for_D.append(sq_errors)
            esd_for_D.append(d_daggers)

        all_time_points.append(iterations_for_D * config.learning_rate)
        sq_errors_by_D.append(np.asarray(errors_for_D, dtype=float))
        d_daggers_by_D.append(np.asarray(esd_for_D, dtype=float))

    payload = cache_key_payload(config, record_iterations)
    key = stable_parameter_key(payload)
    arrays = {
        "D_values": d_values,
        "time_points": np.asarray(all_time_points, dtype=float),
        "sq_errors": np.asarray(sq_errors_by_D, dtype=float),
        "d_daggers": np.asarray(d_daggers_by_D, dtype=float),
        "theta": theta,
        "lambda0": lambda0,
        "record_iterations": record_iterations,
    }
    metadata = {
        "figure": "figure2_opgf_error_esd",
        "parameter_key": key,
        "cache_key_payload": payload,
        "seed": config.seed,
        "n": config.n,
        "sigma0": config.sigma0,
        "d": config.d,
        "J": config.J,
        "p": config.p,
        "q": config.q,
        "gamma": config.gamma,
        "D_values": [int(value) for value in d_values],
        "C": SIGNAL_SCALE,
        "num_experiments": config.num_experiments,
        "learning_rate": config.learning_rate,
        "num_iterations": config.num_iterations,
        "log_start": config.log_start,
        "log_step": config.log_step,
        "fast": config.fast,
        "error_bars": "standard_error",
    }
    return arrays, metadata
