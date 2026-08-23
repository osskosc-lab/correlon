"""T7: orientation is unavailable from symmetric relations and needs asymmetry."""

from __future__ import annotations

import argparse

import numpy as np

from Timeless_T4T7_baselines import (
    detailed_balance_residual,
    entropy_production,
    probability_current_score,
)
from timeless_t4t7_common import RESULTS, auc_score, replace_stage_rows, seeds_for


def forward_chain(seed: int, n: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """A doubly stochastic chain with a signed cycle current."""
    rng = np.random.default_rng(seed)
    off = 0.055 + 0.005 * rng.random()
    transition = np.full((n, n), off)
    np.fill_diagonal(transition, 1.0 - off * (n - 1))
    epsilon = 0.025 + 0.005 * rng.random()
    for i in range(n):
        j = (i + 1) % n
        transition[i, j] += epsilon
        transition[j, i] -= epsilon
    stationary = np.full(n, 1.0 / n)
    return transition, stationary


def reverse_chain(transition: np.ndarray, stationary: np.ndarray) -> np.ndarray:
    return (stationary[None, :] * transition.T) / stationary[:, None]


def probability_current(transition: np.ndarray, stationary: np.ndarray) -> np.ndarray:
    flux = stationary[:, None] * transition
    return flux - flux.T


def symmetric_features(transition: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (transition + transition.T)
    return np.sort(np.linalg.eigvalsh(symmetric))


def run_seed(seed: int) -> dict:
    forward, stationary = forward_chain(seed)
    reverse = reverse_chain(forward, stationary)
    current_f = probability_current(forward, stationary)
    current_r = probability_current(reverse, stationary)
    sym_f = symmetric_features(forward)
    sym_r = symmetric_features(reverse)

    labels = [1, 0]
    symmetric_scores = [float(np.sum(sym_f)), float(np.sum(sym_r))]
    asymmetric_scores = [probability_current_score(current_f), probability_current_score(current_r)]
    ep_scores = [entropy_production(forward, stationary), entropy_production(reverse, stationary)]
    sym_auc = auc_score(labels, symmetric_scores)
    asym_auc = auc_score(labels, asymmetric_scores)
    ep_auc = auc_score(labels, ep_scores)
    sym_accuracy = 0.5 if np.allclose(symmetric_scores[0], symmetric_scores[1]) else float(sym_auc >= 0.5)
    asym_accuracy = float(asymmetric_scores[0] > 0.0 and asymmetric_scores[1] < 0.0)

    reversible = 0.5 * (forward + forward.T)
    reversible /= reversible.sum(axis=1, keepdims=True)
    reversible_current = probability_current(reversible, stationary)
    return {
        "seed": seed,
        "forward_reverse_symmetric_feature_difference": float(np.linalg.norm(sym_f - sym_r)),
        "forward_reverse_transition_difference": float(np.linalg.norm(forward - reverse)),
        "current_sign_reversal_error": float(np.linalg.norm(current_f + current_r)),
        "symmetric_information_accuracy": sym_accuracy,
        "symmetric_information_AUC": sym_auc,
        "asymmetric_current_accuracy": asym_accuracy,
        "asymmetric_current_AUC": asym_auc,
        "entropy_production_AUC": ep_auc,
        "entropy_production_forward": ep_scores[0],
        "entropy_production_reverse": ep_scores[1],
        "detailed_balance_residual_forward": detailed_balance_residual(forward, stationary),
        "detailed_balance_residual_reverse": detailed_balance_residual(reverse, stationary),
        "reversible_negative_control_current_norm": float(np.linalg.norm(reversible_current)),
        "probability_current_baseline_accuracy": asym_accuracy,
        "input_A_has_asymmetric_information": False,
        "input_B_has_explicit_current": True,
        "timestamps_hidden": True,
        "forward_reverse_metadata_stripped": True,
    }


def run(stage: str) -> None:
    rows = [run_seed(seed) for seed in seeds_for(stage)]
    replace_stage_rows(RESULTS / "timeless_T7_raw.csv", rows, stage)
    print(f"T7 {stage}: {len(rows)} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("development", "validation", "confirmation"), required=True)
    run(parser.parse_args().stage)

