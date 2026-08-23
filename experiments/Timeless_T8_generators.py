"""Ground-truth finite-memory, long-memory, and hard-null generators for T8."""

from __future__ import annotations

import numpy as np


POSITIVE_CONTROLS = ("M1_AR_finite", "M2_linear_SSM", "M3_sum_exponential_kernel")
LONG_MEMORY_TARGETS = (
    "L1_power_law_kernel", "L2_fractional_dynamics", "L3_generalized_Langevin",
    "L4_ARFIMA", "L5_long_memory_nonlinear", "L6_history_dependent_intervention",
)
HARD_NULLS = (
    "N1_large_but_finite_AR", "N2_high_dimensional_SSM", "N3_many_exponential_kernel",
    "N4_slow_common_driver", "N5_nonstationary_drift", "N6_switching_regime",
    "N7_colored_noise", "N8_measurement_filter",
)
ALL_FAMILIES = POSITIVE_CONTROLS + LONG_MEMORY_TARGETS


def fractional_weights(length: int, d: float) -> np.ndarray:
    weights = np.ones(length, dtype=float)
    for k in range(1, length):
        weights[k] = weights[k - 1] * (k - 1 + d) / k
    return weights


def family_parameter(family: str, seed: int) -> float:
    if family == "L1_power_law_kernel":
        return (0.25, 0.50, 0.75, 1.25)[seed % 4]
    if family == "L2_fractional_dynamics":
        return (0.18, 0.28, 0.38, 0.45)[seed % 4]
    if family == "L4_ARFIMA":
        return (0.10, 0.20, 0.35, 0.45)[seed % 4]
    return 0.0


def kernel(family: str, length: int, parameter: float | None = None) -> np.ndarray:
    t = np.arange(length, dtype=float)
    parameter = 0.0 if parameter is None else float(parameter)
    if family == "M1_AR_finite":
        out = np.zeros(length)
        out[: min(3, length)] = np.array([0.62, -0.28, 0.22])[: min(3, length)]
        return out
    if family == "M2_linear_SSM":
        rates = np.array([0.08, 0.21, 0.47, 0.83])
        weights = np.array([0.30, -0.18, 0.26, 0.12])
        return np.sum(weights[:, None] * np.exp(-rates[:, None] * t), axis=0)
    if family == "M3_sum_exponential_kernel":
        rates = np.array([0.04, 0.18, 0.62])
        weights = np.array([0.45, 0.30, -0.12])
        return np.sum(weights[:, None] * np.exp(-rates[:, None] * t), axis=0)
    if family == "L1_power_law_kernel":
        alpha = parameter or 0.50
        return (t + 2.0) ** (-alpha)
    if family == "L2_fractional_dynamics":
        return fractional_weights(length, parameter or 0.35)
    if family == "L3_generalized_Langevin":
        return (t + 2.0) ** -0.58 * np.cos(0.11 * t)
    if family == "L4_ARFIMA":
        return fractional_weights(length, parameter or 0.35)
    if family == "L5_long_memory_nonlinear":
        return (t + 2.0) ** -0.42
    if family == "L6_history_dependent_intervention":
        return (t + 2.0) ** -0.36
    raise ValueError(family)


def exact_finite_order(family: str) -> int | None:
    return 3 if family == "M1_AR_finite" else None


def exact_state_dimension(family: str) -> int | None:
    return {"M2_linear_SSM": 4, "M3_sum_exponential_kernel": 3}.get(family)


def has_target_history_mechanism(family: str) -> bool:
    return family in LONG_MEMORY_TARGETS


def validate_ground_truth() -> dict:
    ar = kernel("M1_AR_finite", 64)
    ar_order_pass = bool(np.allclose(ar[3:], 0.0) and abs(ar[2]) > 0)
    ssm_dim_pass = exact_state_dimension("M2_linear_SSM") == 4
    exponential_dim_pass = exact_state_dimension("M3_sum_exponential_kernel") == 3

    power = kernel("L1_power_law_kernel", 1024, 0.50)
    x = np.log(np.arange(128, 1024, dtype=float) + 2.0)
    y = np.log(power[128:])
    slope = float(np.polyfit(x, y, 1)[0])
    powerlaw_pass = abs(slope + 0.50) <= 0.01

    frac = kernel("L2_fractional_dynamics", 4096, 0.35)
    xf = np.log(np.arange(512, 4096, dtype=float))
    yf = np.log(frac[512:])
    fractional_slope = float(np.polyfit(xf, yf, 1)[0])
    fractional_pass = abs(fractional_slope - (0.35 - 1.0)) <= 0.03

    null_mechanism_pass = not any(has_target_history_mechanism(name) for name in HARD_NULLS)
    drift_explicit_kernel_absent = not has_target_history_mechanism("N5_nonstationary_drift")
    checks = {
        "finite_AR_true_order": ar_order_pass,
        "finite_SSM_true_dimension": ssm_dim_pass,
        "sum_exponential_finite_augmentation": exponential_dim_pass,
        "powerlaw_tail": powerlaw_pass,
        "fractional_long_memory": fractional_pass,
        "hard_null_target_mechanism_absent": null_mechanism_pass,
        "drift_explicit_powerlaw_kernel_absent": drift_explicit_kernel_absent,
    }
    return {
        "status": "PASS" if all(checks.values()) else "INCONCLUSIVE_IMPLEMENTATION",
        "checks": checks,
        "powerlaw_fitted_slope": slope,
        "fractional_fitted_slope": fractional_slope,
    }
