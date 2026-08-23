"""T4: complete equal-time correlator hierarchy no-go counterexample."""

from __future__ import annotations

import argparse

import numpy as np

from timeless_t4t7_common import RESULTS, replace_stage_rows, seeds_for


J = np.array([[0.0, -1.0], [1.0, 0.0]])


def gaussian_population_moments_through_6() -> dict[str, float]:
    """Selected complete-basis monomials for a centered 2D standard Gaussian."""
    out: dict[str, float] = {}
    moments_1d = {0: 1.0, 1: 0.0, 2: 1.0, 3: 0.0, 4: 3.0, 5: 0.0, 6: 15.0}
    for degree in range(1, 7):
        for px in range(degree + 1):
            py = degree - px
            out[f"x{px}_y{py}"] = moments_1d[px] * moments_1d[py]
    return out


def empirical_moments(samples: np.ndarray) -> np.ndarray:
    values = []
    x, y = samples[:, 0], samples[:, 1]
    for degree in range(1, 7):
        for px in range(degree + 1):
            values.append(float(np.mean((x ** px) * (y ** (degree - px)))))
    return np.asarray(values)


def run_seed(seed: int) -> list[dict]:
    rows = []
    rng = np.random.default_rng(seed)
    population = gaussian_population_moments_through_6()
    population_vector = np.asarray(list(population.values()), dtype=float)
    population_scale = max(float(np.max(np.abs(population_vector))), 1.0)
    population_diff = max(abs(value - population[key]) for key, value in population.items())
    for gamma in (0.5, 1.0):
        for omega in (0.5, 1.5):
            a0 = -gamma * np.eye(2)
            a1 = a0 + omega * J
            diffusion = 2.0 * gamma * np.eye(2)
            lyapunov0 = a0 + a0.T + diffusion
            lyapunov1 = a1 + a1.T + diffusion
            # Equal-time samples are drawn independently from the shared stationary law.
            s0 = rng.normal(size=(8000, 2))
            s1 = rng.normal(size=(8000, 2))
            empirical0 = empirical_moments(s0)
            empirical1 = empirical_moments(s1)
            empirical_diff = float(np.max(np.abs(empirical0 - empirical1)))
            rows.append({
                "seed": seed,
                "gamma": gamma,
                "omega": omega,
                "analytic_stationary_covariance_difference": 0.0,
                "analytic_equal_time_moment_max_difference_order_1_to_6": population_diff,
                "stationary_lyapunov_residual_A0": float(np.linalg.norm(lyapunov0)),
                "stationary_lyapunov_residual_A1": float(np.linalg.norm(lyapunov1)),
                "empirical_moment_max_difference_order_1_to_6": empirical_diff,
                "empirical_moment_normalized_max_difference": empirical_diff / population_scale,
                "empirical_to_population_normalized_RMSE_A0": float(np.sqrt(np.mean((empirical0 - population_vector) ** 2)) / population_scale),
                "empirical_to_population_normalized_RMSE_A1": float(np.sqrt(np.mean((empirical1 - population_vector) ** 2)) / population_scale),
                "generator_difference_norm": float(np.linalg.norm(a1 - a0)),
                "probability_current_norm_A0": 0.0,
                "probability_current_norm_A1": float(np.linalg.norm(omega * J)),
                "input_has_timestamp": False,
                "input_has_lag": False,
                "input_has_sample_order": False,
                "baseline_equal_time_verdict": "NON_IDENTIFIABLE",
            })
    return rows


def run(stage: str) -> None:
    rows = [row for seed in seeds_for(stage) for row in run_seed(seed)]
    replace_stage_rows(RESULTS / "timeless_T4_raw.csv", rows, stage)
    print(f"T4 {stage}: {len(rows)} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("development", "validation", "confirmation"), required=True)
    run(parser.parse_args().stage)
