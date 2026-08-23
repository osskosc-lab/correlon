"""T6: relative-duration identifiability and its hard aliases."""

from __future__ import annotations

import argparse

import numpy as np
from scipy.linalg import expm

from Timeless_T4T7_baselines import (
    determinant_duration_estimator,
    matrix_log_duration_baseline,
    ordinal_spectral_radius_baseline,
)
from timeless_t4t7_common import (
    RESULTS,
    pairwise_rank_accuracy,
    positive_affine_error,
    replace_stage_rows,
    seeds_for,
)


WORLDS = (
    "stable_linear_flow", "non_normal_linear_flow", "damped_rotation",
    "multi_rate_system", "periodic_rotation", "nearly_degenerate_generator",
    "multi_clock_product_system",
)
IDENTIFIABLE = set(WORLDS) - {"periodic_rotation", "multi_clock_product_system"}
DURATIONS = np.array([0.25, 0.5, 1.0, 2.0])


def generator_for(world: str, seed: int) -> np.ndarray:
    jitter = (seed % 17) * 1e-4
    if world == "stable_linear_flow":
        return np.diag([-0.4 - jitter, -1.0])
    if world == "non_normal_linear_flow":
        return np.array([[-0.5 - jitter, 1.8], [0.0, -1.2]])
    if world == "damped_rotation":
        return np.array([[-0.3 - jitter, -1.1], [1.1, -0.3 - jitter]])
    if world == "multi_rate_system":
        return np.diag([-0.15 - jitter, -0.7, -1.8])
    if world == "periodic_rotation":
        return 2.0 * np.pi * np.array([[0.0, -1.0], [1.0, 0.0]])
    if world == "nearly_degenerate_generator":
        return np.diag([-0.0100 - jitter, -0.0101 - jitter])
    raise ValueError(world)


def semigroup_residual(operators: list[np.ndarray]) -> float:
    residuals = [
        np.linalg.norm(operators[2] - operators[1] @ operators[1]) / max(np.linalg.norm(operators[2]), 1e-15),
        np.linalg.norm(operators[3] - operators[2] @ operators[2]) / max(np.linalg.norm(operators[3]), 1e-15),
    ]
    return float(max(residuals))


def multi_clock_operators(seed: int) -> tuple[list[np.ndarray], np.ndarray, float]:
    jitter = (seed % 13) * 0.002
    clock_pairs = np.array([
        [0.2, 0.2 + jitter],
        [0.5, 1.2 + jitter],
        [1.0, 0.55 + jitter],
        [2.0, 2.5 + jitter],
    ])
    operators = [np.diag(np.exp(-pair)) for pair in clock_pairs]
    centered = clock_pairs - clock_pairs.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    residual = float(singular[1] / max(np.linalg.norm(singular), 1e-15))
    return operators, clock_pairs[:, 0], residual


def run_case(seed: int, world: str) -> dict:
    rng = np.random.default_rng(seed * 101 + WORLDS.index(world))
    if world == "periodic_rotation":
        truth = np.array([0.10, 0.35, 1.10, 1.35])
        generator = generator_for(world, seed)
        operators = [expm(generator * tau) for tau in truth]
        alias_error = max(
            np.linalg.norm(operators[0] - operators[2]),
            np.linalg.norm(operators[1] - operators[3]),
        )
        scalar_residual = 0.0
        composition = np.nan
    elif world == "multi_clock_product_system":
        operators, truth, scalar_residual = multi_clock_operators(seed)
        alias_error = np.nan
        composition = float(
            np.linalg.norm(operators[2] - operators[1] @ operators[1])
            / max(np.linalg.norm(operators[2]), 1e-15)
        )
    else:
        truth = DURATIONS.copy()
        generator = generator_for(world, seed)
        operators = [expm(generator * tau) for tau in truth]
        alias_error = np.nan
        scalar_residual = 0.0
        composition = semigroup_residual(operators)

    storage_permutation = rng.permutation(len(operators))
    serialized = [operators[i] for i in storage_permutation]
    hidden_truth = truth[storage_permutation]
    estimate = determinant_duration_estimator(serialized)
    matrix_log = matrix_log_duration_baseline(serialized)
    ordinal = ordinal_spectral_radius_baseline(serialized)
    estimator_error = positive_affine_error(estimate, hidden_truth)
    baseline_error = positive_affine_error(matrix_log, hidden_truth)
    rank_accuracy = pairwise_rank_accuracy(estimate, hidden_truth)
    ordinal_rank = pairwise_rank_accuracy(ordinal, hidden_truth)
    return {
        "seed": seed,
        "world": world,
        "n_operators": len(operators),
        "scale_aligned_duration_error": estimator_error,
        "rank_order_accuracy": rank_accuracy,
        "composition_residual": composition,
        "matrix_log_baseline_error": baseline_error,
        "ordinal_embedding_rank_accuracy": ordinal_rank,
        "periodic_alias_operator_difference": alias_error,
        "periodic_alias_ground_truth": bool(world == "periodic_rotation" and alias_error < 1e-10),
        "generator_branch_ambiguity": world == "periodic_rotation",
        "multiclock_1D_embedding_residual": scalar_residual,
        "scalar_time_rejected": bool(scalar_residual > 0.20),
        "same_budget_baseline_equal_or_better": bool(baseline_error <= estimator_error + 1e-12),
        "duration_labels_hidden": True,
        "timestamps_hidden": True,
        "operator_storage_order_randomized": True,
        "input_feature_count": int(sum(matrix.size for matrix in serialized)),
    }


def run(stage: str) -> None:
    rows = [run_case(seed, world) for seed in seeds_for(stage) for world in WORLDS]
    replace_stage_rows(RESULTS / "timeless_T6_raw.csv", rows, stage)
    print(f"T6 {stage}: {len(rows)} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("development", "validation", "confirmation"), required=True)
    run(parser.parse_args().stage)

