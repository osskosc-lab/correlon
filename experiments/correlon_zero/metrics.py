"""Frozen Phase-4 operator and the single preregistered Correlon Zero v1 score."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _invsqrt(covariance: np.ndarray, epsilon: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(covariance)
    return (vectors * (1.0 / np.sqrt(np.maximum(values, 0.0) + epsilon))) @ vectors.T


def _window_modes(
    x: np.ndarray,
    y: np.ndarray,
    window: int,
    step: int,
    epsilon: float,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
        raise ValueError("X and Y must be two-dimensional with equal time length")
    if x.shape[1] < 2 or y.shape[1] < 2:
        raise ValueError("the frozen spectral-gap readout requires at least two channels per space")
    if len(x) < window + step:
        raise ValueError(f"at least two windows are required: T={len(x)}, window={window}, step={step}")
    output: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for start in range(0, len(x) - window + 1, step):
        xx = x[start : start + window] - np.mean(x[start : start + window], axis=0, keepdims=True)
        yy = y[start : start + window] - np.mean(y[start : start + window], axis=0, keepdims=True)
        cxx = xx.T @ xx / (window - 1)
        cyy = yy.T @ yy / (window - 1)
        cxy = xx.T @ yy / (window - 1)
        operator = _invsqrt(cxx, epsilon) @ cxy @ _invsqrt(cyy, epsilon)
        u, singular, vt = np.linalg.svd(operator, full_matrices=False)
        output.append((singular, u[:, 0], vt.T[:, 0]))
    if len(output) < 2:
        raise ValueError("fewer than two sliding windows were produced")
    return output


def _readouts(
    modes: list[tuple[np.ndarray, np.ndarray, np.ndarray]], max_lag: int
) -> dict[str, float]:
    s1 = np.asarray([mode[0][0] for mode in modes], dtype=float)
    s2 = np.asarray([mode[0][1] for mode in modes], dtype=float)
    lag_persistence: list[float] = []
    for lag in range(1, min(max_lag, len(modes) - 1) + 1):
        x_overlap = np.mean(
            [float(np.dot(modes[i][1], modes[i + lag][1]) ** 2) for i in range(len(modes) - lag)]
        )
        y_overlap = np.mean(
            [float(np.dot(modes[i][2], modes[i + lag][2]) ** 2) for i in range(len(modes) - lag)]
        )
        lag_persistence.append(math.sqrt(max(x_overlap * y_overlap, 0.0)))
    joint = np.asarray(
        [
            math.sqrt(
                max(
                    float(np.dot(modes[i][1], modes[i + 1][1]) ** 2)
                    * float(np.dot(modes[i][2], modes[i + 1][2]) ** 2),
                    0.0,
                )
            )
            for i in range(len(modes) - 1)
        ],
        dtype=float,
    )
    return {
        "strength": float(np.median(s1)),
        "gap": float(np.median(s1 - s2)),
        "rel_gap": float(np.median((s1 - s2) / np.maximum(s1, 1e-15))),
        "persistence_mean": float(np.mean(lag_persistence)),
        "persistence_q10": float(np.quantile(joint, 0.10, method="linear")),
        "persistence_min": float(np.min(joint)),
        "window_count": float(len(modes)),
    }


def correlon_components(
    x: np.ndarray,
    y: np.ndarray,
    metric_config: dict[str, Any],
    null_seed: int,
) -> dict[str, float]:
    window = int(metric_config["window"])
    step = int(metric_config["step"])
    epsilon = float(metric_config["epsilon"])
    max_lag = int(metric_config["max_lag"])
    null_replicates = int(metric_config["matched_null_replicates"])
    observed = _readouts(_window_modes(x, y, window, step, epsilon), max_lag)
    rng = np.random.default_rng(null_seed)
    shifts = rng.integers(len(y) // 4, 3 * len(y) // 4, size=null_replicates)
    null_values = [
        _readouts(_window_modes(x, np.roll(y, int(shift), axis=0), window, step, epsilon), max_lag)
        for shift in shifts
    ]
    null_medians = {
        key: float(np.median([row[key] for row in null_values]))
        for key in ("strength", "gap", "rel_gap", "persistence_mean", "persistence_q10", "persistence_min")
    }
    return {
        **{f"observed_{key}": float(value) for key, value in observed.items()},
        **{f"null_{key}": float(value) for key, value in null_medians.items()},
        "T_iso": float(observed["gap"] - null_medians["gap"]),
        "T_pers": float(observed["persistence_mean"] - null_medians["persistence_mean"]),
        "T_floor": float(observed["persistence_q10"] - null_medians["persistence_q10"]),
    }


def correlon_score(
    x: np.ndarray,
    y: np.ndarray,
    metric_config: dict[str, Any],
    null_seed: int,
) -> tuple[float, dict[str, float]]:
    components = correlon_components(x, y, metric_config, null_seed)
    isolation = float(np.clip(components["T_iso"], 0.0, 1.0))
    continuity = float(np.clip(components["T_floor"], 0.0, 1.0))
    score = math.sqrt(isolation * continuity)
    if not np.isfinite(score):
        raise FloatingPointError("CorrelonZero_v1 produced a non-finite score")
    return float(score), {**components, "isolation_clipped": isolation, "continuity_clipped": continuity}


def retention(original: float, transformed: float, epsilon: float = 1e-12) -> float:
    if not np.isfinite(original) or not np.isfinite(transformed):
        raise FloatingPointError("retention received a non-finite score")
    return float(np.clip(transformed / max(original, epsilon), 0.0, 1.0))

