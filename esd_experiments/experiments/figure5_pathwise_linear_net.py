"""Figure 5 pathwise linear-network experiment logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from esd_experiments.core.rkhs import pathwise_esd_from_eigendecomposition


class MissingDependencyError(RuntimeError):
    """Raised when an optional runtime dependency is unavailable."""


@dataclass(frozen=True)
class ExperimentConfig:
    n_samples: int
    n_features: int
    n_layers: int
    n_nonzero: int
    noise_std: float
    n_epochs: int
    learning_rate: float
    log_interval: int
    teacher_decay: float = 1.1
    init_noise_std: float = 0.01


def full_config() -> ExperimentConfig:
    return ExperimentConfig(
        n_samples=1000,
        n_features=900,
        n_layers=4,
        n_nonzero=200,
        noise_std=0.1,
        n_epochs=500,
        learning_rate=1e-4,
        log_interval=20,
    )


def fast_config() -> ExperimentConfig:
    return ExperimentConfig(
        n_samples=120,
        n_features=90,
        n_layers=4,
        n_nonzero=30,
        noise_std=0.1,
        n_epochs=20,
        learning_rate=1e-4,
        log_interval=5,
    )


def import_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise MissingDependencyError(
                "Figure 5 requires PyTorch. Install torch before running "
                "esd_experiments/scripts/figure5_pathwise_linear_net.py."
            ) from exc
        raise
    return torch


def make_teacher_coefficients(config: ExperimentConfig) -> np.ndarray:
    indices = np.arange(1, config.n_features + 1, dtype=np.float64)
    beta_star = indices ** (-config.teacher_decay)
    beta_star[min(config.n_nonzero, config.n_features) :] = 0.0
    return beta_star.astype(np.float32)


def make_training_data(
    config: ExperimentConfig, beta_star: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_train = (2 * rng.integers(0, 2, size=(config.n_samples, config.n_features)) - 1).astype(
        np.float32
    )
    noise = config.noise_std * rng.standard_normal(config.n_samples)
    y_train = x_train @ beta_star + noise
    return x_train, y_train.astype(np.float32)


def make_deep_linear_network(torch: Any, config: ExperimentConfig) -> Any:
    nn = torch.nn

    class DeepLinearNetwork(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            hidden_dim = config.n_features
            layers = [nn.Linear(config.n_features, hidden_dim, bias=False)]
            for _ in range(config.n_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            self.layers = nn.Sequential(*layers)
            self.final_layer = nn.Linear(hidden_dim, 1, bias=False)
            self.initialize_weights()

        def initialize_weights(self) -> None:
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    eye = torch.eye(
                        module.out_features,
                        module.in_features,
                        dtype=module.weight.dtype,
                        device=module.weight.device,
                    )
                    with torch.no_grad():
                        module.weight.copy_(
                            eye + config.init_noise_std * torch.randn_like(module.weight)
                        )

        def forward_representation(self, x: Any) -> Any:
            return self.layers(x)

        def forward(self, x: Any) -> Any:
            return self.final_layer(self.forward_representation(x))

    return DeepLinearNetwork()


def representation_matrix(torch: Any, model: Any, n_features: int) -> Any:
    first_weight = model.layers[0].weight
    matrix = torch.eye(n_features, dtype=first_weight.dtype, device=first_weight.device)
    for layer in model.layers:
        matrix = layer.weight @ matrix
    return matrix


def _as_numpy_1d(array: Any) -> np.ndarray:
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    return np.asarray(array, dtype=np.float64).reshape(-1)


def _as_numpy_2d(array: Any) -> np.ndarray:
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    return np.asarray(array, dtype=np.float64)


def pathwise_esd_dimension(
    eigenvalues: Any,
    eigenvectors: Any,
    beta_star: Any,
    sigma2: float,
) -> int:
    eigenvalues_np = _as_numpy_1d(eigenvalues)
    eigenvectors_np = _as_numpy_2d(eigenvectors)
    beta_star_np = _as_numpy_1d(beta_star)
    theta_coeffs = eigenvectors_np.T @ beta_star_np

    return int(pathwise_esd_from_eigendecomposition(theta_coeffs, eigenvalues_np, float(sigma2)))


def record_metrics(
    torch: Any,
    model: Any,
    beta_star_torch: Any,
    sigma_eff2: float,
) -> tuple[float, int]:
    with torch.no_grad():
        representation = representation_matrix(torch, model, beta_star_torch.numel())
        gram_matrix = representation.T @ representation
        eigenvalues, eigenvectors = torch.linalg.eigh(gram_matrix)
        esd_value = pathwise_esd_dimension(eigenvalues, eigenvectors, beta_star_torch, sigma_eff2)

        final_weight = model.final_layer.weight.detach().reshape(-1)
        effective_predictor = representation.T @ final_weight
        risk = torch.sum((effective_predictor - beta_star_torch) ** 2).item()

    return float(risk), int(esd_value)


def run_experiment(config: ExperimentConfig, seed: int, verbose: bool = True) -> dict[str, list[float]]:
    torch = import_torch()
    torch.manual_seed(seed)

    beta_star = make_teacher_coefficients(config)
    x_train, y_train = make_training_data(config, beta_star, seed)

    x_train_torch = torch.tensor(x_train, dtype=torch.float32)
    y_train_torch = torch.tensor(y_train, dtype=torch.float32)
    beta_star_torch = torch.tensor(beta_star, dtype=torch.float32)

    sigma_eff2 = (config.noise_std**2 + np.linalg.norm(beta_star, ord=1) ** 2) / config.n_samples
    model = make_deep_linear_network(torch, config)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    history: dict[str, list[float]] = {"epoch": [], "loss": [], "esd": [], "risk": []}

    if verbose:
        print("Starting Figure 5 pathwise linear-network experiment...")

    for epoch in range(config.n_epochs + 1):
        prediction = model(x_train_torch).squeeze(-1)
        loss = criterion(prediction, y_train_torch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if epoch % config.log_interval == 0 or epoch == config.n_epochs:
            risk, esd_value = record_metrics(torch, model, beta_star_torch, sigma_eff2)
            history["epoch"].append(float(epoch))
            history["loss"].append(float(loss.item()))
            history["esd"].append(float(esd_value))
            history["risk"].append(float(risk))
            if verbose:
                print(
                    f"Epoch {epoch:4d} | loss {loss.item():.6f} | "
                    f"ESD {esd_value:4d} | risk {risk:.6f}"
                )

    return history


def history_to_arrays(history: dict[str, list[float]]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value, dtype=float) for key, value in history.items()}


def metadata_for_config(config: ExperimentConfig, seed: int, fast: bool) -> dict[str, object]:
    return {
        "figure": "figure5_pathwise_linear_net",
        "seed": seed,
        "n_samples": config.n_samples,
        "n_features": config.n_features,
        "n_layers": config.n_layers,
        "n_nonzero": config.n_nonzero,
        "noise_std": config.noise_std,
        "n_epochs": config.n_epochs,
        "learning_rate": config.learning_rate,
        "log_interval": config.log_interval,
        "teacher_decay": config.teacher_decay,
        "init_noise_std": config.init_noise_std,
        "fast": fast,
        "risk": "target_function_error",
    }
