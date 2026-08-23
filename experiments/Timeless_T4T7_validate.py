"""Independent raw-file validator and validation-freeze writer for T4-T7."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from timeless_t4t7_common import (
    CONFIG_PATH,
    CONFIRMATION_PATH,
    RESULTS,
    VALIDATION_PATH,
    assert_confirmation_freeze,
    canonical_json_hash,
    current_git,
    read_json,
    summary,
    write_json,
)


CONFIG = {
    "program": "Timeless Correlator T4-T7",
    "version": "1.0",
    "source_HEAD_SHA": "a727f11e95e40d7b0919dc87dd9730bfc16abc69",
    "seed_protocol": {
        "development": [0, 29], "validation": [1000, 1049], "confirmation": [20000, 20299],
        "bootstrap": 881122,
    },
    "thresholds": {
        "T4_population_equal_time_max_difference": 0.0,
        "T5_median_reachability_F1": 0.90,
        "T5_cycle_detection_accuracy": 0.95,
        "T5_false_total_order_rate": 0.05,
        "T6_median_scale_aligned_error": 0.10,
        "T6_median_rank_accuracy": 0.90,
        "T6_alias_operator_difference": 1e-10,
        "T6_multiclock_1D_residual": 0.20,
        "T7_symmetric_AUC_interval": [0.45, 0.55],
        "T7_asymmetric_accuracy": 0.90,
        "T7_asymmetric_AUC": 0.90,
    },
    "worlds": {
        "T4": ["stationary_OU_reversible", "stationary_OU_rotational_current"],
        "T5": ["linear_DAG", "nonlinear_DAG", "branching_DAG", "mediated_chain", "common_driver", "disconnected_components", "feedback_pair", "directed_cycle"],
        "T6": ["stable_linear_flow", "non_normal_linear_flow", "damped_rotation", "multi_rate_system", "periodic_rotation", "nearly_degenerate_generator", "multi_clock_product_system"],
        "T7": ["forward_reverse_twin", "reversible_negative_control"],
    },
    "baselines": {
        "order": ["causal_reachability", "topological_sort", "spectral_seriation"],
        "duration": ["matrix_logarithm", "semigroup_fit", "ordinal_embedding"],
        "arrow": ["probability_current", "detailed_balance", "entropy_production"],
    },
    "verdict_priority": [
        "CLOCK_LEAKAGE", "INCONCLUSIVE_IMPLEMENTATION", "FULL_EQUAL_TIME_NO_GO",
        "SCALAR_TIME_MODEL_REJECTED", "DURATION_ALIASING", "GENERATOR_AMBIGUITY",
        "THERMODYNAMIC_ARROW_SUFFICIENT", "ARROW_REQUIRES_ASYMMETRY",
        "CAUSAL_ORDER_ONLY", "STANDARD_THEORY_SUFFICIENT", "RELATIONAL_TIME_CANDIDATE",
    ],
    "holdout_definitions": ["H1", "H2", "H3", "H4", "H5", "H6", "H7"],
}


def stage_rows(filename: str, stage: str) -> pd.DataFrame:
    frame = pd.read_csv(RESULTS / filename)
    selected = frame[frame.stage == stage].copy()
    if selected.empty:
        raise RuntimeError(f"no {stage} rows in {filename}")
    return selected


def evaluate(stage: str) -> dict:
    t4 = stage_rows("timeless_T4_raw.csv", stage)
    t5 = stage_rows("timeless_T5_raw.csv", stage)
    t6 = stage_rows("timeless_T6_raw.csv", stage)
    t7 = stage_rows("timeless_T7_raw.csv", stage)

    t4_pass = bool(
        (t4.analytic_equal_time_moment_max_difference_order_1_to_6 == 0.0).all()
        and (t4.generator_difference_norm > 0.0).all()
        and (t4.probability_current_norm_A1 > 0.0).all()
    )
    dag = t5[~t5.true_cycle.astype(bool)]
    cycles = t5[t5.true_cycle.astype(bool)]
    t5_f1 = float(dag.reachability_F1.median())
    cycle_accuracy = float(cycles.cycle_detection_correct.mean())
    false_total_rate = float(cycles.false_total_order.mean())
    t5_pass = t5_f1 >= 0.90 and cycle_accuracy >= 0.95 and false_total_rate <= 0.05
    causal_only = bool(t5.estimator_equals_standard_reachability.astype(bool).all())

    identifiable = t6[t6.world.isin([
        "stable_linear_flow", "non_normal_linear_flow", "damped_rotation",
        "multi_rate_system", "nearly_degenerate_generator",
    ])]
    duration_error = float(identifiable.scale_aligned_duration_error.median())
    duration_rank = float(identifiable.rank_order_accuracy.median())
    identifiable_pass = duration_error <= 0.10 and duration_rank >= 0.90
    aliases = t6[t6.world == "periodic_rotation"]
    alias_pass = bool(aliases.periodic_alias_ground_truth.astype(bool).all())
    multiclock = t6[t6.world == "multi_clock_product_system"]
    scalar_rejected = bool(multiclock.scalar_time_rejected.astype(bool).all())
    duration_baseline_sufficient = bool(t6.same_budget_baseline_equal_or_better.astype(bool).all())

    symmetric_accuracy = float(t7.symmetric_information_accuracy.mean())
    symmetric_auc = float(t7.symmetric_information_AUC.mean())
    asymmetric_accuracy = float(t7.asymmetric_current_accuracy.mean())
    asymmetric_auc = float(t7.asymmetric_current_AUC.mean())
    t7_pass = (
        0.45 <= symmetric_auc <= 0.55
        and asymmetric_accuracy >= 0.90
        and asymmetric_auc >= 0.90
        and float(t7.current_sign_reversal_error.max()) <= 1e-12
    )

    labels = {
        "T4": "FULL_EQUAL_TIME_NO_GO" if t4_pass else "INCONCLUSIVE_IMPLEMENTATION",
        "T5": "CAUSAL_ORDER_ONLY" if t5_pass and causal_only else "ORDER_ONLY_SUPPORTED" if t5_pass else "ORDER_NOT_IDENTIFIABLE",
        "T6": "DURATION_ALIASING" if alias_pass else "RELATIVE_DURATION_IDENTIFIABLE" if identifiable_pass else "INCONCLUSIVE_POWER",
        "T7": "ARROW_REQUIRES_ASYMMETRY" if t7_pass else "INCONCLUSIVE_IMPLEMENTATION",
    }
    return {
        "stage": stage,
        "T4": {
            "verdict": labels["T4"],
            "population_equal_time_match": t4_pass,
            "empirical_moment_difference": summary(t4.empirical_moment_max_difference_order_1_to_6),
            "generator_difference": summary(t4.generator_difference_norm),
            "current_difference": summary(t4.probability_current_norm_A1),
        },
        "T5": {
            "verdict": labels["T5"],
            "reachability_F1": summary(dag.reachability_F1, dag.reachability_F1 < 0.90),
            "cycle_detection_accuracy": cycle_accuracy,
            "false_total_order_rate": false_total_rate,
            "standard_causal_reachability_equal": causal_only,
        },
        "T6": {
            "verdict": labels["T6"],
            "identifiable_subset_pass": identifiable_pass,
            "scale_aligned_duration_error": summary(identifiable.scale_aligned_duration_error, identifiable.scale_aligned_duration_error > 0.10),
            "rank_order_accuracy": summary(identifiable.rank_order_accuracy, identifiable.rank_order_accuracy < 0.90),
            "periodic_aliasing": alias_pass,
            "generator_ambiguity": bool(aliases.generator_branch_ambiguity.astype(bool).all()),
            "scalar_time_model_rejected": scalar_rejected,
            "multiclock_embedding_residual": summary(multiclock.multiclock_1D_embedding_residual),
            "matrix_log_same_budget_sufficient": duration_baseline_sufficient,
        },
        "T7": {
            "verdict": labels["T7"],
            "symmetric_accuracy": summary(t7.symmetric_information_accuracy),
            "symmetric_AUC": summary(t7.symmetric_information_AUC),
            "asymmetric_accuracy": summary(t7.asymmetric_current_accuracy, t7.asymmetric_current_accuracy < 0.90),
            "asymmetric_AUC": summary(t7.asymmetric_current_AUC),
            "entropy_production_AUC": summary(t7.entropy_production_AUC),
            "standard_probability_current_equal": bool(np.allclose(t7.probability_current_baseline_accuracy, t7.asymmetric_current_accuracy)),
        },
        "labels": labels,
        "standard_theory_sufficient": causal_only and duration_baseline_sufficient and t7_pass,
        "relational_time_candidate": False,
    }


def freeze_validation() -> dict:
    metrics = evaluate("validation")
    write_json(CONFIG_PATH, CONFIG)
    config_hash = canonical_json_hash(CONFIG)
    leakage = read_json(RESULTS / "timeless_T4T7_leakage.json")
    payload = {
        "status": "FROZEN" if leakage.get("status") == "PASS" else "CLOCK_LEAKAGE",
        "validation_config_SHA256": config_hash,
        "configuration": str(CONFIG_PATH.relative_to(RESULTS.parent)),
        "metrics": metrics,
        "clock_leakage_status": leakage.get("status"),
        "git": current_git(),
    }
    write_json(VALIDATION_PATH, payload)
    print("Validation freeze SHA-256:", config_hash)
    return payload


def validate_confirmation() -> dict:
    _, freeze = assert_confirmation_freeze()
    metrics = evaluate("confirmation")
    payload = {
        "status": "COMPLETE",
        "validation_config_SHA256": freeze["validation_config_SHA256"],
        "independent_raw_csv_recomputation": True,
        "metrics": metrics,
        "git": current_git(),
    }
    write_json(CONFIRMATION_PATH, payload)
    decision = {
        "program": "Timeless Correlator T4-T7",
        "T4_verdict": metrics["labels"]["T4"],
        "T5_verdict": metrics["labels"]["T5"],
        "T6_verdict": metrics["labels"]["T6"],
        "T6_secondary": ["GENERATOR_AMBIGUITY", "SCALAR_TIME_MODEL_REJECTED"],
        "T7_verdict": metrics["labels"]["T7"],
        "strongest_standard_baseline": "causal reachability / matrix logarithm / probability current (same-budget ties)",
        "overall_scientific_verdict": "STANDARD_THEORY_SUFFICIENT",
        "RELATIONAL_TIME_CANDIDATE": False,
        "claim_firewall": "No physical-time, causality-emergence, or Correlon-specific novelty claim is supported.",
        "validation_config_SHA256": freeze["validation_config_SHA256"],
    }
    write_json(RESULTS / "timeless_T4T7_decision.json", decision)
    print("Independent confirmation validation complete")
    for key in ("T4_verdict", "T5_verdict", "T6_verdict", "T7_verdict"):
        print(key, decision[key])
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("freeze", "confirmation"), required=True)
    args = parser.parse_args()
    freeze_validation() if args.stage == "freeze" else validate_confirmation()

