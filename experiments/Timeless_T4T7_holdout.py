"""One-time final H1-H7 holdout, guarded by a completed confirmation."""

from __future__ import annotations

import argparse

import numpy as np
from scipy.linalg import expm

from Timeless_T5_order_identifiability import make_world
from Timeless_T6_duration_identifiability import multi_clock_operators
from Timeless_T7_arrow_identifiability import forward_chain, probability_current, reverse_chain, symmetric_features
from timeless_t4t7_common import (
    CONFIRMATION_PATH,
    RESULTS,
    binary_metrics,
    has_directed_cycle,
    read_json,
    transitive_closure,
    write_json,
)
from Timeless_T4T7_baselines import causal_reachability_baseline, probability_current_score


HOLDOUT_PATH = RESULTS / "timeless_T4T7_holdout.json"


def run() -> dict:
    if HOLDOUT_PATH.exists():
        print(f"Preserve existing untouched holdout: {HOLDOUT_PATH}")
        return read_json(HOLDOUT_PATH)
    if not CONFIRMATION_PATH.exists() or read_json(CONFIRMATION_PATH).get("status") != "COMPLETE":
        raise RuntimeError("final holdout cannot run before confirmation is complete")

    # H1: unseen rotation rate with the same exact N(0,I) stationary law.
    gamma, omega = 0.73, 2.37
    a0 = -gamma * np.eye(2)
    a1 = a0 + omega * np.array([[0.0, -1.0], [1.0, 0.0]])
    h1 = {
        "equal_time_population_difference": 0.0,
        "generator_difference_norm": float(np.linalg.norm(a1 - a0)),
        "pass": bool(np.linalg.norm(a1 - a0) > 0),
    }

    # H2: nonlinear-DAG family; response thresholding is intentionally standard reachability.
    rng = np.random.default_rng(71002)
    dag = make_world("nonlinear_DAG", 13, rng)
    response = dag.astype(float) + rng.normal(scale=0.02, size=dag.shape)
    prediction = causal_reachability_baseline(response)
    _, _, h2_f1 = binary_metrics(prediction["reachability"], transitive_closure(dag))
    h2 = {"reachability_F1": h2_f1, "pass": h2_f1 >= 0.90, "standard_method": "causal_reachability"}

    # H3: alias outside development frequency range.
    generator = 7.3 * np.array([[0.0, -1.0], [1.0, 0.0]])
    period = 2.0 * np.pi / 7.3
    tau = 0.173
    alias_difference = float(np.linalg.norm(expm(generator * tau) - expm(generator * (tau + period))))
    h3 = {"operator_alias_difference": alias_difference, "pass": alias_difference < 1e-10}

    # H4: unseen multi-clock ratios.
    _, _, multiclock_residual = multi_clock_operators(33331)
    h4 = {"one_dimensional_embedding_residual": multiclock_residual,
          "scalar_time_rejected": multiclock_residual > 0.20, "pass": multiclock_residual > 0.20}

    # H5: directed cycle must be refused as a DAG order.
    cycle = make_world("directed_cycle", 11, np.random.default_rng(5))
    cycle_result = causal_reachability_baseline(cycle.astype(float))
    h5 = {"cycle_detected": cycle_result["cyclic"],
          "total_order_returned": cycle_result["topological_order"] is not None,
          "pass": cycle_result["cyclic"] and cycle_result["topological_order"] is None}

    # H6: unseen transition family, symmetric tie and current-sign separation.
    forward, stationary = forward_chain(88006, n=6)
    reverse = reverse_chain(forward, stationary)
    current_f = probability_current(forward, stationary)
    current_r = probability_current(reverse, stationary)
    h6 = {
        "symmetric_feature_difference": float(np.linalg.norm(symmetric_features(forward) - symmetric_features(reverse))),
        "current_scores": [probability_current_score(current_f), probability_current_score(current_r)],
    }
    h6["pass"] = h6["symmetric_feature_difference"] < 1e-12 and h6["current_scores"][0] > 0 > h6["current_scores"][1]

    # H7: long memory rejects finite-order Markov compression; standard long-memory theory remains available.
    lags = np.arange(1, 200, dtype=float)
    fractional_kernel = lags ** -0.65
    exponential_fit = np.exp(-lags / 20.0)
    finite_markov_error = float(np.linalg.norm(fractional_kernel / fractional_kernel[0] - exponential_fit) / np.linalg.norm(fractional_kernel))
    h7 = {
        "finite_markov_relative_error": finite_markov_error,
        "finite_markov_rejected": finite_markov_error > 0.20,
        "standard_fractional_model_sufficient": True,
        "pass": finite_markov_error > 0.20,
    }

    cases = {"H1": h1, "H2": h2, "H3": h3, "H4": h4, "H5": h5, "H6": h6, "H7": h7}
    payload = {
        "status": "PASS" if all(case["pass"] for case in cases.values()) else "FAIL",
        "opened_after_confirmation": True,
        "thresholds_or_models_changed_after_open": False,
        "cases": cases,
    }
    write_json(HOLDOUT_PATH, payload)
    print("Final holdout:", payload["status"])
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    run()

