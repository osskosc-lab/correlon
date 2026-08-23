"""T5: partial-order identifiability from directed response relations."""

from __future__ import annotations

import argparse

import numpy as np

from Timeless_T4T7_baselines import causal_reachability_baseline, spectral_seriation_baseline
from timeless_t4t7_common import (
    RESULTS,
    binary_metrics,
    has_directed_cycle,
    kendall_tau,
    replace_stage_rows,
    seeds_for,
    topological_order,
    transitive_closure,
)


WORLDS = (
    "linear_DAG", "nonlinear_DAG", "branching_DAG", "mediated_chain",
    "common_driver", "disconnected_components", "feedback_pair", "directed_cycle",
)


def make_world(world: str, n: int, rng: np.random.Generator) -> np.ndarray:
    adjacency = np.zeros((n, n), dtype=bool)
    if world in {"linear_DAG", "mediated_chain"}:
        adjacency[np.arange(n - 1), np.arange(1, n)] = True
    elif world == "nonlinear_DAG":
        adjacency[np.arange(n - 1), np.arange(1, n)] = True
        for i in range(n):
            for j in range(i + 2, n):
                adjacency[i, j] = rng.random() < 0.15
    elif world == "branching_DAG":
        for child in range(1, n):
            adjacency[(child - 1) // 2, child] = True
    elif world == "common_driver":
        adjacency[0, 1:] = True
    elif world == "disconnected_components":
        split = n // 2
        adjacency[np.arange(split - 1), np.arange(1, split)] = True
        if n - split > 1:
            adjacency[np.arange(split, n - 1), np.arange(split + 1, n)] = True
    elif world == "feedback_pair":
        adjacency[0, 1] = adjacency[1, 0] = True
        if n > 2:
            adjacency[1, 2] = True
    elif world == "directed_cycle":
        adjacency[np.arange(n), np.roll(np.arange(n), -1)] = True
    else:
        raise ValueError(world)
    return adjacency


def permute_graph(adjacency: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    return adjacency[np.ix_(permutation, permutation)]


def run_case(seed: int, world: str, n: int) -> dict:
    rng = np.random.default_rng(seed * 97 + WORLDS.index(world))
    original = make_world(world, n, rng)
    permutation = rng.permutation(n)
    truth = permute_graph(original, permutation)
    response = truth.astype(float) + rng.normal(scale=0.02, size=(n, n))
    np.fill_diagonal(response, 0.0)
    # A second, irrelevant serialization permutation never enters matrix semantics.
    serialization = rng.permutation(n * n)
    serialized_response = response.ravel()[serialization]
    restored = np.empty_like(serialized_response)
    restored[serialization] = serialized_response
    estimator = causal_reachability_baseline(restored.reshape(n, n))
    truth_reach = transitive_closure(truth)
    precision, recall, f1 = binary_metrics(estimator["reachability"], truth_reach)
    cyclic = has_directed_cycle(truth)
    predicted_cyclic = bool(estimator["cyclic"])
    false_total_order = bool(cyclic and estimator["topological_order"] is not None)
    tau = np.nan
    if world in {"linear_DAG", "mediated_chain"} and not cyclic:
        truth_order_original = list(range(n))
        inverse = {old: new for new, old in enumerate(permutation.tolist())}
        truth_order = [inverse[node] for node in truth_order_original]
        tau = kendall_tau(estimator["topological_order"], truth_order)
    spectral_order = spectral_seriation_baseline(response)
    standard = causal_reachability_baseline(response)
    _, _, baseline_f1 = binary_metrics(standard["reachability"], truth_reach)
    return {
        "seed": seed,
        "world": world,
        "n_nodes": n,
        "reachability_precision": precision,
        "reachability_recall": recall,
        "reachability_F1": f1,
        "Kendall_tau_when_total_order_exists": tau,
        "true_cycle": cyclic,
        "cycle_detected": predicted_cyclic,
        "cycle_detection_correct": predicted_cyclic == cyclic,
        "false_total_order": false_total_order,
        "causal_reachability_baseline_F1": baseline_f1,
        "spectral_seriation_returned_order": len(spectral_order) == n,
        "node_labels_randomized": True,
        "storage_order_randomized": True,
        "input_has_timestamp": False,
        "input_has_sample_index": False,
        "input_has_true_topological_order": False,
        "estimator_equals_standard_reachability": abs(f1 - baseline_f1) <= 1e-12,
    }


def run(stage: str) -> None:
    rows = []
    sizes = (4, 8, 16, 32)
    for seed in seeds_for(stage):
        n = sizes[seed % len(sizes)]
        rows.extend(run_case(seed, world, n) for world in WORLDS)
    replace_stage_rows(RESULTS / "timeless_T5_raw.csv", rows, stage)
    print(f"T5 {stage}: {len(rows)} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("development", "validation", "confirmation"), required=True)
    run(parser.parse_args().stage)

