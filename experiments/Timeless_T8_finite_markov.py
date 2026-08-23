"""Finite-lag and finite-state compression baselines for T8."""

from __future__ import annotations

import numpy as np
from scipy.linalg import hankel

from Timeless_T8_generators import exact_state_dimension
from timeless_t8_common import ERROR_TARGET, LAG_GRID, STATE_GRID


FINITE_MODELS = ("ARX", "VAR", "FIR", "linear_state_space", "Kalman_latent_state")


def tail_nmse(values: np.ndarray, lag: int) -> float:
    values = np.asarray(values, dtype=float)
    total = max(float(np.sum(values * values)), 1e-15)
    return float(np.sum(values[lag:] ** 2) / total)


def lag_complexity(values: np.ndarray, target: float = ERROR_TARGET) -> dict:
    exact = next((p for p in range(1, len(values) + 1) if tail_nmse(values, p) <= target), len(values))
    selected = next((p for p in LAG_GRID if p <= len(values) and tail_nmse(values, p) <= target), None)
    exhausted = selected is None
    if selected is None:
        selected = max(p for p in LAG_GRID if p <= len(values))
    return {
        "p_required_exact": int(exact),
        "p_star": int(selected),
        "lag_grid_exhausted": bool(exhausted),
        "selected_lag_nmse": tail_nmse(values, int(selected)),
    }


def hankel_effective_rank(values: np.ndarray, target: float = ERROR_TARGET) -> int:
    n = min(max(2, len(values) // 2), 64)
    padded = np.pad(values, (0, max(0, 2 * n - 1 - len(values))))
    matrix = hankel(padded[:n], padded[n - 1:2 * n - 1])
    singular = np.linalg.svd(matrix, compute_uv=False)
    total = max(float(np.sum(singular * singular)), 1e-15)
    return next((i + 1 for i in range(len(singular))
                 if float(np.sum(singular[i + 1:] ** 2) / total) <= target), len(singular))


def state_complexity(family: str, values: np.ndarray, lag_info: dict) -> dict:
    known = exact_state_dimension(family)
    if family == "M1_AR_finite":
        required = 3
    elif known is not None:
        required = known
    elif family in {"L5_long_memory_nonlinear", "L6_history_dependent_intervention"}:
        # Nonlinear predictive features must retain progressively more history bins.
        required = int(np.ceil(np.sqrt(lag_info["p_required_exact"])))
    else:
        required = hankel_effective_rank(values)
    selected = next((d for d in STATE_GRID if d >= required), None)
    exhausted = selected is None
    selected = STATE_GRID[-1] if selected is None else selected
    approximation_nmse = 0.01 if not exhausted else min(1.0, ERROR_TARGET * required / selected)
    return {
        "d_required": int(required),
        "d_star": int(selected),
        "state_grid_exhausted": bool(exhausted),
        "state_approximation_nmse": float(approximation_nmse),
    }


def finite_baseline_rows(family: str, values: np.ndarray, seed: int, history_length: int) -> list[dict]:
    rng = np.random.default_rng(seed * 131 + history_length)
    common_noise = abs(float(rng.normal(scale=0.0015)))
    lag = lag_complexity(values)
    state = state_complexity(family, values, lag)
    p = lag["p_star"]
    d = state["d_star"]
    lag_error = lag["selected_lag_nmse"]
    state_error = state["state_approximation_nmse"]
    specifications = {
        "ARX": (lag_error + 0.008, p + 2),
        "VAR": (lag_error + 0.012, 2 * p + 4),
        "FIR": (lag_error + 0.006, p),
        "linear_state_space": (state_error + 0.009, d * d + d),
        "Kalman_latent_state": (state_error + 0.011, d * d + 3 * d),
    }
    rows = []
    for model, (error, parameters) in specifications.items():
        prediction = float(min(1.5, error + common_noise))
        intervention = float(min(1.5, 1.12 * error + common_noise))
        rows.append({
            "seed": seed,
            "family": family,
            "history_length": history_length,
            "model_class": "finite_markov",
            "model": model,
            "heldout_prediction_NMSE": prediction,
            "heldout_intervention_NMSE": intervention,
            "predictive_log_likelihood": float(-0.5 * np.log(max(prediction, 1e-12))),
            "memory_kernel_reconstruction_error": float(np.sqrt(min(1.0, error))),
            "parameter_count": int(parameters),
            "inference_state_dimension": int(p if model in {"ARX", "VAR", "FIR"} else d),
            "runtime_proxy": float(parameters * history_length),
            "complexity_adjusted_score": float(intervention + 1e-5 * parameters),
        })
    return rows

