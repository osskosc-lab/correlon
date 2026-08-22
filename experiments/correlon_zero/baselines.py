"""Frozen conventional baseline scores for the Correlon Zero battery."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import coherence


def _center(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float) - np.mean(x, axis=0, keepdims=True)


def pearson_summary(x: np.ndarray, y: np.ndarray) -> float:
    xx = _center(x)
    yy = _center(y)
    xx /= np.maximum(np.std(xx, axis=0, ddof=1, keepdims=True), 1e-12)
    yy /= np.maximum(np.std(yy, axis=0, ddof=1, keepdims=True), 1e-12)
    cross = xx.T @ yy / (len(xx) - 1)
    return float(np.sqrt(np.mean(np.square(np.clip(cross, -1.0, 1.0)))))


def gaussian_mutual_information(x: np.ndarray, y: np.ndarray, regularizer: float) -> float:
    xx = _center(x)
    yy = _center(y)
    joint = np.column_stack([xx, yy])
    cxx = xx.T @ xx / (len(xx) - 1) + regularizer * np.eye(xx.shape[1])
    cyy = yy.T @ yy / (len(yy) - 1) + regularizer * np.eye(yy.shape[1])
    cjoint = joint.T @ joint / (len(joint) - 1) + regularizer * np.eye(joint.shape[1])
    sign_x, log_x = np.linalg.slogdet(cxx)
    sign_y, log_y = np.linalg.slogdet(cyy)
    sign_joint, log_joint = np.linalg.slogdet(cjoint)
    if min(sign_x, sign_y, sign_joint) <= 0:
        raise FloatingPointError("regularized covariance is not positive definite for mutual information")
    information = max(0.0, 0.5 * (log_x + log_y - log_joint))
    return float(1.0 - np.exp(-information / min(x.shape[1], y.shape[1])))


def covariance_summaries(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    joint = _center(np.column_stack([x, y]))
    covariance = joint.T @ joint / (len(joint) - 1)
    values = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    largest = float(values[-1])
    explained = float(largest / max(float(np.sum(values)), 1e-12))
    return largest, explained


def _first_pc(x: np.ndarray) -> np.ndarray:
    centered = _center(x)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[0]


def spectral_coherence_score(x: np.ndarray, y: np.ndarray, nperseg: int) -> float:
    x_pc = _first_pc(x)
    y_pc = _first_pc(y)
    segment = min(int(nperseg), len(x_pc))
    _, values = coherence(x_pc, y_pc, nperseg=segment, detrend="constant")
    return float(np.clip(np.mean(values), 0.0, 1.0))


def autocorrelation_persistence(x: np.ndarray, y: np.ndarray) -> float:
    joint = _center(np.column_stack([x, y]))
    numerator = np.sum(joint[:-1] * joint[1:], axis=0)
    denominator = np.sqrt(np.sum(joint[:-1] ** 2, axis=0) * np.sum(joint[1:] ** 2, axis=0))
    values = np.divide(numerator, np.maximum(denominator, 1e-12))
    return float(np.mean(np.abs(np.clip(values, -1.0, 1.0))))


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    xx = _center(x)
    yy = _center(y)
    numerator = float(np.linalg.norm(xx.T @ yy, ord="fro") ** 2)
    denominator = float(np.linalg.norm(xx.T @ xx, ord="fro") * np.linalg.norm(yy.T @ yy, ord="fro"))
    return float(np.clip(numerator / max(denominator, 1e-12), 0.0, 1.0))


def _lag_design(a: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    target = a[lags:]
    predictors = np.column_stack([a[lags - lag : len(a) - lag] for lag in range(1, lags + 1)])
    return target, predictors


def _ridge_sse(design: np.ndarray, target: np.ndarray, ridge: float) -> float:
    design_centered = _center(design)
    target_centered = _center(target)
    gram = design_centered.T @ design_centered + ridge * np.eye(design_centered.shape[1])
    coefficients = np.linalg.solve(gram, design_centered.T @ target_centered)
    residual = target_centered - design_centered @ coefficients
    return float(np.sum(residual**2))


def _predictive_gain(source: np.ndarray, target_space: np.ndarray, lags: int, ridge: float) -> float:
    target, target_history = _lag_design(target_space, lags)
    _, source_history = _lag_design(source, lags)
    base_sse = _ridge_sse(target_history, target, ridge)
    full_sse = _ridge_sse(np.column_stack([target_history, source_history]), target, ridge)
    return float(np.clip(1.0 - full_sse / max(base_sse, 1e-12), 0.0, 1.0))


def var_arx_predictive(x: np.ndarray, y: np.ndarray, lags: int, ridge: float) -> float:
    return float(0.5 * (_predictive_gain(x, y, lags, ridge) + _predictive_gain(y, x, lags, ridge)))


def baseline_scores(x: np.ndarray, y: np.ndarray, config: dict[str, Any]) -> dict[str, float]:
    params = config["baseline_parameters"]
    largest, explained = covariance_summaries(x, y)
    scores = {
        "pearson": pearson_summary(x, y),
        "mutual_information": gaussian_mutual_information(x, y, float(params["covariance_regularizer"])),
        "largest_covariance_eigenvalue": largest,
        "pca_rank1_explained_variance": explained,
        "spectral_coherence": spectral_coherence_score(x, y, int(params["coherence_nperseg"])),
        "autocorrelation_persistence": autocorrelation_persistence(x, y),
        "linear_CKA": linear_cka(x, y),
        "VAR_ARX_predictive": var_arx_predictive(
            x, y, int(params["var_lags"]), float(params["ridge"])
        ),
    }
    for name, value in scores.items():
        if not np.isfinite(value):
            raise FloatingPointError(f"baseline {name} produced a non-finite score")
    return scores

