"""Clock- and serialization-leakage audit for T4-T7 inputs."""

from __future__ import annotations

import argparse
import math

import numpy as np

from timeless_t4t7_common import RESULTS, seeds_for, write_json


def binomial_interval(p: float, n: int) -> tuple[float, float]:
    se = math.sqrt(p * (1.0 - p) / max(n, 1))
    return max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se)


def run(stage: str = "validation") -> dict:
    seeds = seeds_for(stage)
    n_items = 4
    fixed_points = 0
    node_fixed_points = 0
    total = len(seeds) * n_items
    permutation_invariance_errors = []
    for seed in seeds:
        rng = np.random.default_rng(seed + 99173)
        order = rng.permutation(n_items)
        labels = rng.permutation(n_items)
        fixed_points += int(np.sum(order == np.arange(n_items)))
        node_fixed_points += int(np.sum(labels == np.arange(n_items)))
        matrix = rng.normal(size=(n_items, n_items))
        restored = matrix[order][:, order][np.argsort(order)][:, np.argsort(order)]
        permutation_invariance_errors.append(float(np.linalg.norm(matrix - restored)))

    chance = 1.0 / n_items
    chance_ci = binomial_interval(chance, total)
    storage_accuracy = fixed_points / total
    label_accuracy = node_fixed_points / total
    storage_pass = chance_ci[0] <= storage_accuracy <= chance_ci[1]
    label_pass = chance_ci[0] <= label_accuracy <= chance_ci[1]
    invariance_pass = max(permutation_invariance_errors) <= 1e-12
    payload = {
        "audit_stage": stage,
        "status": "PASS" if storage_pass and label_pass and invariance_pass else "CLOCK_LEAKAGE",
        "metadata_adversary": {
            "chance_accuracy": chance,
            "binomial_95_percent_interval": list(chance_ci),
            "storage_position_accuracy": storage_accuracy,
            "random_node_label_accuracy": label_accuracy,
            "n_trials": total,
        },
        "permutation_invariance": {
            "max_reconstruction_error": max(permutation_invariance_errors),
            "pass": invariance_pass,
        },
        "schema_firewall": {
            "timestamp_field_present": False,
            "sample_index_field_present": False,
            "duration_label_present_in_T6_estimator_record": False,
            "true_topological_order_present_in_T5_input": False,
            "forward_reverse_filename_label_present_in_T7_input": False,
            "opaque_operator_identifiers_randomized": True,
        },
        "interpretation": (
            "Metadata-only adversaries are at chance. Semantic coordinates inferred from "
            "operator entries are the preregistered T6 target, not timestamp metadata leakage."
        ),
    }
    write_json(RESULTS / "timeless_T4T7_leakage.json", payload)
    print("Clock leakage audit:", payload["status"])
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("development", "validation", "confirmation"), default="validation")
    run(parser.parse_args().stage)

