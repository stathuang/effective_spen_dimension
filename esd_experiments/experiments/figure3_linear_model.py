"""Figure 3 linear-model experiment logic."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from esd_experiments.core.esd import compute_esd
from esd_experiments.core.linear import (
    apply_alignment,
    build_A_diag_exp,
    cache_svd,
    make_beta,
    make_design,
    prediction_mse_by_column,
    translate_theta_lambda,
)
from esd_experiments.core.results import records_to_arrays


@dataclass(frozen=True)
class LinearCaseConfig:
    """Configuration for one panel of Figure 3."""

    case_id: int
    n: int
    p: int
    sigma0_sq: float
    replicates: int
    design_decay: float | str
    beta_decay: str
    beta_power: float | None
    alpha_grid: np.ndarray
    output_name: str


def _standard_error(values: np.ndarray, axis: int) -> np.ndarray:
    if values.shape[axis] <= 1:
        shape = list(values.shape)
        del shape[axis]
        return np.zeros(shape, dtype=float)
    return np.std(values, axis=axis, ddof=1) / np.sqrt(values.shape[axis])


def _pcr_fit_all_from_svd(u: np.ndarray, d: np.ndarray, vt: np.ndarray, y: np.ndarray) -> np.ndarray:
    proj = (u.T @ y) / d
    contrib = vt.T * proj[np.newaxis, :]
    return np.cumsum(contrib, axis=1)


def build_case_config(case_id: int, fast: bool) -> LinearCaseConfig:
    n = 300
    p = 400
    replicates = 20

    if fast:
        n = 120
        p = 160
        replicates = 4

    if case_id == 1:
        alpha_count = 5 if fast else 10
        return LinearCaseConfig(
            case_id=1,
            n=n,
            p=p,
            sigma0_sq=1.0,
            replicates=replicates,
            design_decay=0.95,
            beta_decay="power",
            beta_power=0.2,
            alpha_grid=np.linspace(0.0, 30.0, alpha_count),
            output_name="risk_vs_alpha_case_1.pdf",
        )

    if case_id == 2:
        alpha_count = 5 if fast else 10
        return LinearCaseConfig(
            case_id=2,
            n=n,
            p=p,
            sigma0_sq=1.0,
            replicates=replicates,
            design_decay="log",
            beta_decay="log",
            beta_power=None,
            alpha_grid=np.linspace(0.0, 10.0, alpha_count),
            output_name="risk_vs_alpha_case_2.pdf",
        )

    raise ValueError(f"Unknown Figure 3 case: {case_id}")


def run_case(config: LinearCaseConfig, seed: int) -> tuple[list[dict[str, float]], dict[str, np.ndarray]]:
    design_seed = seed + 1009 * config.case_id
    noise_rng = np.random.default_rng(seed + 2003 * config.case_id)

    X0 = make_design(config.n, config.p, par=config.design_decay, seed=design_seed)
    beta_power = 1.0 if config.beta_power is None else config.beta_power
    beta_star = make_beta(
        config.p,
        decay_type=config.beta_decay,
        power=beta_power,
    )
    eps_mat = noise_rng.normal(
        0.0, np.sqrt(config.sigma0_sq), size=(config.n, config.replicates)
    )

    records: list[dict[str, float]] = []
    loss_by_alpha = []
    risk_by_k_by_alpha = []
    risk_se_by_k_by_alpha = []
    for alpha in config.alpha_grid:
        A = build_A_diag_exp(config.p, alpha)
        X_A, beta_A = apply_alignment(X0, beta_star, A)

        svd_A = cache_svd(X_A)
        theta, lambda_vals = translate_theta_lambda(svd_A, beta_A)
        d_dagger = compute_esd(theta, lambda_vals, config.sigma0_sq / config.n)

        u, d, vt = np.linalg.svd(X_A, full_matrices=False)
        rank = len(d)
        loss_mat = np.full((rank, config.replicates), np.nan)

        for b in range(config.replicates):
            y_b = X_A @ beta_A + eps_mat[:, b]
            beta_mat = _pcr_fit_all_from_svd(u, d, vt, y_b)
            loss_mat[:, b] = prediction_mse_by_column(X_A, beta_mat, beta_A)

        risk_by_k = np.mean(loss_mat, axis=1)
        se_by_k = _standard_error(loss_mat, axis=1)
        k_star = int(np.argmin(risk_by_k))
        loss_by_alpha.append(loss_mat)
        risk_by_k_by_alpha.append(risk_by_k)
        risk_se_by_k_by_alpha.append(se_by_k)
        records.append(
            {
                "alpha": float(alpha),
                "esd": float(d_dagger),
                "k_opt": float(k_star + 1),
                "risk": float(risk_by_k[k_star]),
                "risk_se": float(se_by_k[k_star]),
            }
        )

    raw_arrays = {
        "alpha_grid": np.asarray(config.alpha_grid, dtype=float),
        "loss_by_alpha_k_replication": np.asarray(loss_by_alpha, dtype=float),
        "risk_by_alpha_k": np.asarray(risk_by_k_by_alpha, dtype=float),
        "risk_se_by_alpha_k": np.asarray(risk_se_by_k_by_alpha, dtype=float),
        "beta_star": beta_star,
    }
    return records, raw_arrays


def metadata_for_case(config: LinearCaseConfig, seed: int, fast: bool) -> dict[str, object]:
    return {
        "figure": "figure3_linear_model",
        "case_id": config.case_id,
        "seed": seed,
        "n": config.n,
        "p": config.p,
        "sigma0_sq": config.sigma0_sq,
        "replicates": config.replicates,
        "design_decay": config.design_decay,
        "beta_decay": config.beta_decay,
        "beta_power": config.beta_power,
        "fast": fast,
        "error_bars": "standard_error",
    }


def arrays_for_case(records: list[dict[str, float]], raw_arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    arrays = records_to_arrays(records)
    arrays.update(raw_arrays)
    return arrays

