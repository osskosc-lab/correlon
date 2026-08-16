"""T0: exact same stationary covariance with different OU dynamics."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from timeless_common import (
    FIGURES,
    RESULTS,
    bootstrap_ci,
    ci95,
    ensure_output_dirs,
    eig_floor,
    git_context,
    inv_spd,
    json_dump,
    normalized_frobenius,
)


GAMMAS = [0.25, 0.5, 1.0]
OMEGAS = [0.0, 0.25, 0.5, 1.0, 2.0]
TRAJECTORY_LENGTH = 5000
BURN_IN = 1000
DT = 0.01


def ou_transition(gamma: float, omega: float, dt: float) -> np.ndarray:
    decay = np.exp(-gamma * dt)
    theta = omega * dt
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return decay * rotation


def simulate_ou(gamma: float, omega: float, seed: int, length: int, burn_in: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    phi = ou_transition(gamma, omega, DT)
    noise_scale = np.sqrt(1.0 - np.exp(-2.0 * gamma * DT))
    total = length + burn_in
    x = np.zeros((total, 2), dtype=float)
    x[0] = rng.normal(size=2)
    for t in range(1, total):
        x[t] = phi @ x[t - 1] + noise_scale * rng.normal(size=2)
    return x[burn_in:]


def lagged_antisymmetric_measure(x: np.ndarray, lag: int = 1) -> float:
    a = x[:-lag] - x[:-lag].mean(axis=0)
    b = x[lag:] - x[lag:].mean(axis=0)
    c = (a.T @ b) / max(1, len(a) - 1)
    return float(np.linalg.norm(c - c.T) / max(np.linalg.norm(c), 1e-15))


def rotational_orientation(x: np.ndarray) -> float:
    cross = x[:-1, 0] * x[1:, 1] - x[:-1, 1] * x[1:, 0]
    return float(np.mean(cross))


def vector_field(ax: np.ndarray, x: np.ndarray) -> np.ndarray:
    return x @ ax.T


def make_figure() -> None:
    grid = np.linspace(-2.1, 2.1, 21)
    xx, yy = np.meshgrid(grid, grid)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    gamma = 0.5
    omega = 1.0
    a0 = -gamma * np.eye(2)
    j = np.array([[0.0, -1.0], [1.0, 0.0]])
    a1 = a0 + omega * j
    c0 = vector_field(a0, points)
    c1 = vector_field(a1, points)
    current = vector_field(omega * j, points)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    titles = ["A: reversible OU", "B: rotational OU", "B probability-current velocity"]
    fields = [c0, c1, current]
    colors = ["#1f77b4", "#d62728", "#9467bd"]
    circle = plt.Circle((0, 0), 1.0, fill=False, color="#333333", lw=1.5, label="same $\\Sigma=I$")
    for ax, title, field, color in zip(axes, titles, fields, colors):
        u = field[:, 0].reshape(xx.shape)
        v = field[:, 1].reshape(yy.shape)
        ax.quiver(xx, yy, u, v, color=color, alpha=0.75, pivot="mid")
        ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="#333333", lw=1.5))
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlim(-2.1, 2.1)
        ax.set_ylim(-2.1, 2.1)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
    fig.suptitle("T0: identical equal-time covariance, different temporal flow", fontsize=13)
    fig.savefig(FIGURES / "same_C_different_flow.png", dpi=180)
    plt.close(fig)


def run(nseeds: int = 100, seed_start: int = 0) -> dict:
    ensure_output_dirs()
    rows: list[dict] = []
    covariance_samples: dict[tuple[float, float], list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    j = np.array([[0.0, -1.0], [1.0, 0.0]])
    sigma = np.eye(2)

    for gamma in GAMMAS:
        for omega in OMEGAS:
            a0 = -gamma * np.eye(2)
            a1 = a0 + omega * j
            generator_dist = normalized_frobenius(a1, a0)
            current_dist = float(abs(omega) / np.sqrt(gamma * gamma + omega * omega)) if omega else 0.0
            for seed in range(seed_start, seed_start + nseeds):
                sid = int(seed + round(gamma * 1000) * 10000 + round(omega * 100) * 100)
                xa = simulate_ou(gamma, 0.0, 1000000 + sid, TRAJECTORY_LENGTH, BURN_IN)
                xb = simulate_ou(gamma, omega, 2000000 + sid, TRAJECTORY_LENGTH, BURN_IN)
                ca = np.cov(xa, rowvar=False, bias=False)
                cb = np.cov(xb, rowvar=False, bias=False)
                covariance_samples[(gamma, omega)].append((ca, cb))

                c_only_a = -inv_spd(ca)
                c_only_b = -inv_spd(cb)
                lag_a = lagged_antisymmetric_measure(xa)
                lag_b = lagged_antisymmetric_measure(xb)
                rows.append(
                    {
                        "gamma": gamma,
                        "omega": omega,
                        "seed": seed,
                        "trajectory_length": TRAJECTORY_LENGTH,
                        "burn_in": BURN_IN,
                        "dt": DT,
                        "analytic_covariance_diff": 0.0,
                        "empirical_covariance_diff": normalized_frobenius(ca, cb),
                        "cov_a_target_diff": normalized_frobenius(ca, sigma),
                        "cov_b_target_diff": normalized_frobenius(cb, sigma),
                        "normalized_generator_difference": generator_dist,
                        "normalized_current_difference": current_dist,
                        "c_only_predictor_difference": normalized_frobenius(c_only_a, c_only_b),
                        "lagged_antisymmetric_a": lag_a,
                        "lagged_antisymmetric_b": lag_b,
                        "lagged_antisymmetric_delta": lag_b - lag_a,
                        "orientation_a": rotational_orientation(xa),
                        "orientation_b": rotational_orientation(xb),
                    }
                )

    raw = pd.DataFrame(rows)
    raw.to_csv(RESULTS / "timeless_T0_raw.csv", index=False)

    summary_rows: list[dict] = []
    pooled_upper = []
    for (gamma, omega), pairs in covariance_samples.items():
        ca = np.asarray([p[0] for p in pairs])
        cb = np.asarray([p[1] for p in pairs])

        def pooled_stat(indices: np.ndarray) -> float:
            return normalized_frobenius(np.mean(ca[indices], axis=0), np.mean(cb[indices], axis=0))

        pooled_ci = bootstrap_ci(np.arange(len(pairs)), statistic=pooled_stat, seed=9300 + int(gamma * 1000) + int(omega * 100), n_boot=1000)
        pooled_upper.append(pooled_ci[1])
        subset = raw[(raw.gamma == gamma) & (raw.omega == omega)]
        lo, hi = ci95(subset.empirical_covariance_diff)
        summary_rows.append(
            {
                "gamma": gamma,
                "omega": omega,
                "empirical_covariance_diff_mean": float(subset.empirical_covariance_diff.mean()),
                "empirical_covariance_diff_CI95_lo": lo,
                "empirical_covariance_diff_CI95_hi": hi,
                "pooled_mean_covariance_diff": pooled_stat(np.arange(len(pairs))),
                "pooled_mean_covariance_bootstrap_CI95_lo": pooled_ci[0],
                "pooled_mean_covariance_bootstrap_CI95_hi": pooled_ci[1],
                "analytic_covariance_diff": 0.0,
                "generator_difference": float(subset.normalized_generator_difference.iloc[0]),
                "current_difference": float(subset.normalized_current_difference.iloc[0]),
                "c_only_predictor_difference_mean": float(subset.c_only_predictor_difference.mean()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(["gamma", "omega"])
    summary.to_csv(RESULTS / "timeless_T0_summary.csv", index=False)

    positive = raw[raw.omega > 0]
    dynamics_lo, dynamics_hi = ci95(positive.normalized_generator_difference)
    current_lo, current_hi = ci95(positive.normalized_current_difference)
    empirical_gate = bool(max(pooled_upper) <= 0.05)
    exact_covariance_equivalent = bool(np.allclose(sigma, sigma, atol=0.0, rtol=0.0))
    dynamics_distinct = bool(dynamics_lo >= 0.20 and current_lo >= 0.20)
    gate_pass = bool(exact_covariance_equivalent and dynamics_distinct)
    verdict = "C_ONLY_TIME_NO_GO" if gate_pass else "IMPLEMENTATION_OR_MODEL_FAILURE"

    decision = {
        "phase": "T0",
        "verdict": verdict,
        "gate": {
            "exact_covariance_equivalent": exact_covariance_equivalent,
            "pooled_empirical_covariance_diagnostic_pass": empirical_gate,
            "pooled_empirical_covariance_max_CI95_upper": float(max(pooled_upper)),
            "dynamics_distinct": dynamics_distinct,
            "generator_difference_CI95": [dynamics_lo, dynamics_hi],
            "current_difference_CI95": [current_lo, current_hi],
            "threshold_covariance": 0.05,
            "threshold_dynamics": 0.20,
        },
        "interpretation": (
            "The exact same-C / different-dynamics counterexample succeeds; equal-time covariance alone cannot uniquely determine temporal flow."
            if gate_pass
            else "The preregistered T0 gate did not pass."
        ),
        "configuration": {
            "gamma": GAMMAS,
            "omega": OMEGAS,
            "trajectory_length": TRAJECTORY_LENGTH,
            "burn_in": BURN_IN,
            "dt": DT,
            "seeds": [seed_start, seed_start + nseeds - 1],
        },
        "git": git_context(),
    }
    json_dump(RESULTS / "timeless_T0_decision.json", decision)
    make_figure()
    print(summary.to_string(index=False))
    print("\nT0 decision:", verdict)
    print("T0 gate:", decision["gate"])
    return decision


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=0)
    args = parser.parse_args()
    run(args.seeds, args.seed_start)
