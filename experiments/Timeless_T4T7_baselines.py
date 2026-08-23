"""Same-information-budget standard baselines for T4-T7."""

from __future__ import annotations

import numpy as np
from scipy.linalg import logm

from timeless_t4t7_common import has_directed_cycle, topological_order, transitive_closure


def causal_reachability_baseline(response: np.ndarray, threshold: float = 0.5) -> dict:
    adjacency = np.asarray(response) > threshold
    np.fill_diagonal(adjacency, False)
    cyclic = has_directed_cycle(adjacency)
    return {
        "adjacency": adjacency,
        "reachability": transitive_closure(adjacency),
        "cyclic": cyclic,
        "topological_order": None if cyclic else topological_order(adjacency),
    }


def spectral_seriation_baseline(response: np.ndarray) -> list[int]:
    score = np.asarray(response).sum(axis=1) - np.asarray(response).sum(axis=0)
    return np.argsort(-score, kind="stable").astype(int).tolist()


def matrix_log_duration_baseline(operators: list[np.ndarray]) -> np.ndarray:
    """Recover a monotone stable-flow coordinate using the trace of log(M)."""
    values = []
    for matrix in operators:
        generator_coordinate = -float(np.real(np.trace(logm(matrix)))) / matrix.shape[0]
        values.append(generator_coordinate)
    return np.asarray(values)


def determinant_duration_estimator(operators: list[np.ndarray]) -> np.ndarray:
    values = []
    for matrix in operators:
        sign, logabsdet = np.linalg.slogdet(matrix)
        values.append(-float(logabsdet) / matrix.shape[0] if sign != 0 else np.nan)
    return np.asarray(values)


def ordinal_spectral_radius_baseline(operators: list[np.ndarray]) -> np.ndarray:
    values = []
    for matrix in operators:
        radius = max(float(np.max(np.abs(np.linalg.eigvals(matrix)))), 1e-15)
        values.append(-np.log(radius))
    return np.asarray(values)


def probability_current_score(current: np.ndarray) -> float:
    n = current.shape[0]
    return float(sum(current[i, (i + 1) % n] for i in range(n)))


def entropy_production(transition: np.ndarray, stationary: np.ndarray) -> float:
    flux = stationary[:, None] * transition
    reverse_flux = flux.T
    mask = (flux > 0) & (reverse_flux > 0)
    return float(np.sum(flux[mask] * np.log(flux[mask] / reverse_flux[mask])))


def detailed_balance_residual(transition: np.ndarray, stationary: np.ndarray) -> float:
    flux = stationary[:, None] * transition
    return float(np.linalg.norm(flux - flux.T))

