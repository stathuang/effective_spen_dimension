"""Figure 4 RKHS KPCPE experiment logic."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from esd_experiments.core.esd import compute_esd
from esd_experiments.core.results import records_to_arrays
from esd_experiments.core.rkhs import (
    cosine_features,
    effective_noise_from_sup_norm,
    estimate_sup_norm_from_cosine_coeffs,
    truncation_loss_matrix,
)


@dataclass(frozen=True)
class RKHSConfig:
    """Configuration for the Appendix C Figure 4 RKHS simulation."""

    n: int
    J: int
    D: int
    replicates: int
    sigma0_sq: float
    spectrum_decay: float
    signal_decay: float
    alpha_grid: np.ndarray
    sup_norm_grid_size: int
    output_name: str


def _standard_error(values: np.ndarray, axis: int) -> np.ndarray:
    if values.shape[axis] <= 1:
        shape = list(values.shape)
        del shape[axis]
        return np.zeros(shape, dtype=float)
    return np.std(values, axis=axis, ddof=1) / np.sqrt(values.shape[axis])


def build_config(fast: bool) -> RKHSConfig:
    if fast:
        return RKHSConfig(
            n=120,
            J=180,
            D=30,
            replicates=3,
            sigma0_sq=1.0,
            spectrum_decay=1.1,
            signal_decay=4.0,
            alpha_grid=np.linspace(0.0, 30.0, 5),
            sup_norm_grid_size=2000,
            output_name="risk_vs_alpha_rkhs.pdf",
        )

    return RKHSConfig(
        n=400,
        J=800,
        D=80,
        replicates=10,
        sigma0_sq=1.0,
        spectrum_decay=1.1,
        signal_decay=4.0,
        alpha_grid=np.linspace(0.0, 30.0, 11),
        sup_norm_grid_size=10000,
        output_name="risk_vs_alpha_rkhs.pdf",
    )


def make_alignment_weights(J: int, D: int) -> np.ndarray:
    if D > J:
        raise ValueError("D must be no larger than J.")
    return np.concatenate([np.linspace(0.0, 1.0, D), np.zeros(J - D)])


def estimate_sup_norm_sq(theta: np.ndarray, grid_size: int) -> float:
    f_sup_norm = estimate_sup_norm_from_cosine_coeffs(theta, num_grid=grid_size)
    return float(f_sup_norm**2)


def simulate(config: RKHSConfig, seed: int) -> tuple[list[dict[str, float]], float, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    noise_rng = np.random.default_rng(seed + 1)

    index = np.arange(1, config.J + 1)
    lambda0 = index.astype(float) ** (-config.spectrum_decay)
    theta0 = np.sqrt(config.sigma0_sq) * index.astype(float) ** (-config.signal_decay)
    alignment_weights = make_alignment_weights(config.J, config.D)

    x = rng.uniform(0.0, 1.0, config.n)
    Phi = cosine_features(x, config.J)
    f_true = Phi @ theta0

    f_sup_norm_sq = estimate_sup_norm_sq(theta0, config.sup_norm_grid_size)
    sigma_eff2 = effective_noise_from_sup_norm(
        config.sigma0_sq,
        f_sup_norm_sq,
        config.n,
    )
    eps_mat = noise_rng.normal(
        0.0, np.sqrt(config.sigma0_sq), size=(config.n, config.replicates)
    )

    records: list[dict[str, float]] = []
    loss_by_alpha = []
    risk_by_k_by_alpha = []
    risk_se_by_k_by_alpha = []
    for alpha in config.alpha_grid:
        lambda_alpha = lambda0 * np.exp(alpha * alignment_weights)
        order = np.argsort(lambda_alpha)[::-1]
        theta_sorted = theta0[order]

        d_dagger = compute_esd(theta0, lambda_alpha, sigma_eff2)
        loss_mat = np.full((config.J, config.replicates), np.nan)

        for b in range(config.replicates):
            y_b = f_true + eps_mat[:, b]
            theta_hat = (Phi.T @ y_b) / config.n
            theta_hat_sorted = theta_hat[order]
            loss_mat[:, b] = truncation_loss_matrix(theta_hat_sorted, theta_sorted)

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
                "lower_bound": float((d_dagger - 1) * config.sigma0_sq / config.n),
                "upper_bound": float(2.0 * d_dagger * sigma_eff2),
            }
        )

    raw_arrays = {
        "alpha_grid": np.asarray(config.alpha_grid, dtype=float),
        "lambda0": lambda0,
        "theta0": theta0,
        "alignment_weights": alignment_weights,
        "x": x,
        "f_true": f_true,
        "loss_by_alpha_k_replication": np.asarray(loss_by_alpha, dtype=float),
        "risk_by_alpha_k": np.asarray(risk_by_k_by_alpha, dtype=float),
        "risk_se_by_alpha_k": np.asarray(risk_se_by_k_by_alpha, dtype=float),
    }
    return records, f_sup_norm_sq, raw_arrays


def metadata_for_config(
    config: RKHSConfig, seed: int, fast: bool, f_sup_norm_sq: float
) -> dict[str, object]:
    sigma_eff2 = effective_noise_from_sup_norm(
        config.sigma0_sq,
        f_sup_norm_sq,
        config.n,
    )
    return {
        "figure": "figure4_rkhs_kpcpe",
        "seed": seed,
        "n": config.n,
        "J": config.J,
        "D": config.D,
        "replicates": config.replicates,
        "sigma0_sq": config.sigma0_sq,
        "spectrum_decay": config.spectrum_decay,
        "signal_decay": config.signal_decay,
        "sup_norm_grid_size": config.sup_norm_grid_size,
        "f_sup_norm_sq": f_sup_norm_sq,
        "sigma_eff2": sigma_eff2,
        "fast": fast,
        "error_bars": "standard_error",
    }


def arrays_for_records(records: list[dict[str, float]], raw_arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    arrays = records_to_arrays(records)
    arrays.update(raw_arrays)
    return arrays

