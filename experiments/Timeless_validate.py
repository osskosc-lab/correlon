"""Independent gate recomputation from Timeless Correlon raw CSV files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from timeless_common import RESULTS, bootstrap_ci, ci95, git_context, json_dump, json_load, sha256_json


def validate_t0() -> dict:
    raw = pd.read_csv(RESULTS / "timeless_T0_raw.csv")
    positive = raw[raw.omega > 0]
    gen_ci = ci95(positive.normalized_generator_difference)
    current_ci = ci95(positive.normalized_current_difference)
    exact_cov = bool(np.allclose(raw.analytic_covariance_diff.to_numpy(), 0.0, atol=0.0, rtol=0.0))
    dynamics = bool(gen_ci[0] >= 0.20 and current_ci[0] >= 0.20)
    empirical_case_ci = []
    for (_, _), group in raw.groupby(["gamma", "omega"]):
        empirical_case_ci.append(ci95(group.empirical_covariance_diff)[1])
    return {
        "verdict": "C_ONLY_TIME_NO_GO" if exact_cov and dynamics else "IMPLEMENTATION_OR_MODEL_FAILURE",
        "exact_covariance_equivalent": exact_cov,
        "generator_difference_CI95": list(gen_ci),
        "current_difference_CI95": list(current_ci),
        "dynamics_distinct": dynamics,
        "empirical_per_seed_covariance_CI95_max_upper": float(np.nanmax(empirical_case_ci)),
        "empirical_covariance_is_diagnostic_only": True,
        "thresholds": {"covariance": 0.05, "dynamics": 0.20},
    }


def validate_t1() -> dict:
    raw = pd.read_csv(RESULTS / "timeless_T1_raw.csv")
    controls = pd.read_csv(RESULTS / "timeless_T1_controls.csv")
    target = raw[(raw.sample_size >= 2000) & (raw.condition <= 10.0)]
    median_error = float(target.generator_error.median())
    median_cosine = float(target.vector_field_cosine.median())
    medians = raw[raw.condition <= 10.0].groupby("sample_size").generator_error.median().sort_index()
    monotonic = bool(medians.loc[500] >= medians.loc[2000] >= medians.loc[10000])
    positive = bool(median_error <= 0.10 and median_cosine >= 0.95 and monotonic)
    ambiguity = controls[controls.control == "unknown_symplectic_structure"].generator_error.to_numpy()
    amb_ci = ci95(ambiguity)
    return {
        "verdict": "RESTRICTED_GIBBS_GAUSSIAN_RECOVERY" if positive else "IMPLEMENTATION_OR_MODEL_FAILURE",
        "median_generator_error_N_ge_2000_cond_le_10": median_error,
        "median_vector_field_cosine_N_ge_2000_cond_le_10": median_cosine,
        "median_generator_error_by_sample_size_cond_le_10": {str(int(k)): float(v) for k, v in medians.items()},
        "monotonicity": monotonic,
        "positive_control_pass": positive,
        "unknown_symplectic_generator_error_CI95": list(amb_ci),
        "unknown_symplectic_structure_pass": bool(amb_ci[0] >= 0.20),
        "thresholds": {"generator_error": 0.10, "vector_field_cosine": 0.95, "unknown_J_lower": 0.20},
    }


def validate_t2() -> dict:
    raw = pd.read_csv(RESULTS / "timeless_T2_raw.csv")
    max_gen = float(raw.generator_error.max())
    max_flow = float(raw.flow_frobenius_error.max())
    passed = bool(max_gen <= 1e-8 and max_flow <= 1e-6)
    return {
        "verdict": "MODULAR_FLOW_CONTROL_PASS" if passed else "IMPLEMENTATION_FAILURE",
        "max_generator_error": max_gen,
        "max_flow_error": max_flow,
        "generator_threshold": 1e-8,
        "flow_threshold": 1e-6,
        "pass": passed,
        "claim_limit": "known modular-flow mathematics only; not Correlon-specific evidence",
    }


def validate_t3() -> dict:
    selection = json_load(RESULTS / "timeless_T3_selection.json")
    config = json_load(RESULTS / "timeless_correlon_config.json")
    config_hash = sha256_json(config)
    hash_pass = config_hash == selection["config_sha256"]
    raw = pd.read_csv(RESULTS / "timeless_T3_confirmation_raw.csv")
    corr = raw[(raw.method == "correlon") & (raw["rank"] == int(selection["correlon_rank"]))].copy()
    base = raw[(raw.method == selection["best_baseline_method"]) & (raw["rank"] == int(selection["best_baseline_rank"]))].copy()
    if corr.empty or base.empty:
        raise RuntimeError("T3 confirmation selection produced no rows; rank/method filter is invalid")
    key = ["seed", "dimension", "true_rank", "strength"]
    paired = corr[key + ["generator_error", "cca_identity_diff"]].merge(
        base[key + ["generator_error"]], on=key, suffixes=("_correlon", "_baseline")
    )
    paired["relative_gain"] = 1.0 - paired.generator_error_correlon / np.maximum(paired.generator_error_baseline, 1e-15)
    gain_ci = bootstrap_ci(paired.relative_gain.to_numpy(), seed=881122, n_boot=5000)
    gain_mean = float(paired.relative_gain.mean())
    identity_max = float(max(raw.cca_identity_diff.max(), selection["cca_identity_max_diff"]))
    identity = bool(identity_max <= float(selection["identity_tolerance"]))
    specificity = bool(gain_ci[0] > 0.10)
    if not hash_pass:
        verdict = "IMPLEMENTATION_FAILURE"
    elif identity:
        verdict = "CORRELON_EQUALS_CCA_FOR_THIS_OPERATOR"
    elif specificity:
        verdict = "CORRELON_RELATION_TO_STATE_BRIDGE_PROVISIONALLY_SUPPORTED"
    else:
        verdict = "RELATION_TO_STATE_NO_GO"
    return {
        "verdict": verdict,
        "configuration_hash": config_hash,
        "configuration_hash_matches_selection": hash_pass,
        "selected_correlon_rank": int(selection["correlon_rank"]),
        "selected_baseline": selection["best_baseline_method"],
        "selected_baseline_rank": int(selection["best_baseline_rank"]),
        "n_paired_confirmation_cases": int(len(paired)),
        "correlon_error_mean": float(paired.generator_error_correlon.mean()),
        "baseline_error_mean": float(paired.generator_error_baseline.mean()),
        "relative_gain_mean": gain_mean,
        "relative_gain_CI95": list(gain_ci),
        "specificity_threshold": 0.10,
        "specificity_gate_pass": specificity,
        "cca_identity_max_diff": identity_max,
        "cca_identity_tolerance": float(selection["identity_tolerance"]),
        "cca_identity_gate": identity,
    }


def run() -> dict:
    t0 = validate_t0()
    t1 = validate_t1()
    t2 = validate_t2()
    t3 = validate_t3()
    decision = {
        "program": "Timeless Correlon",
        "validator": "independent_raw_csv_recomputation",
        "git": git_context(),
        "level_0": {
            "decision": "UNIVERSAL_C_TO_TIME_FALSIFIED" if t0["verdict"] == "C_ONLY_TIME_NO_GO" else "LEVEL_0_FAILURE",
            "details": t0,
        },
        "level_1": {
            "decision": "RESTRICTED_C_TO_FLOW_SUPPORTED" if t1["positive_control_pass"] else "LEVEL_1_FAILURE",
            "additional_structure": "C_REQUIRES_ADDITIONAL_ALGEBRAIC_STRUCTURE" if t1["unknown_symplectic_structure_pass"] else "UNRESOLVED",
            "details": t1,
        },
        "level_2": {"decision": t2["verdict"], "details": t2},
        "level_3": {"decision": t3["verdict"], "details": t3},
        "strongest_claim_status": "NOT_ESTABLISHED: algebraic structure and state are not both derived from the Correlon primitive",
        "previous_phase_firewall": {
            "phase5c_no_go_preserved": True,
            "direct_causation_claim_not_restored": True,
            "descriptive_relation_mode_interpretation_preserved": True,
        },
    }
    json_dump(RESULTS / "timeless_correlon_decision.json", decision)
    json_dump(RESULTS / "timeless_correlon_independent_validation.json", {"t0": t0, "t1": t1, "t2": t2, "t3": t3, "git": git_context()})
    print("Independent validation complete")
    print("T0:", t0["verdict"])
    print("T1:", t1["verdict"], "unknown-J:", t1["unknown_symplectic_structure_pass"])
    print("T2:", t2["verdict"])
    print("T3:", t3["verdict"], "relative gain CI:", t3["relative_gain_CI95"])
    return decision


if __name__ == "__main__":
    run()
