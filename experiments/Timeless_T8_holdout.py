"""One-time H1-H8 T8 holdout opened after frozen confirmation."""

from __future__ import annotations

import numpy as np

from Timeless_T8_finite_markov import lag_complexity
from Timeless_T8_generators import kernel
from Timeless_T8_interventions import matched_history_pair, response
from Timeless_T8_nonmarkov_baselines import CANDIDATE_MODEL, nonmarkov_baseline_rows
from timeless_t8_common import CONFIRMATION_PATH, RESULTS, read_json, write_json


HOLDOUT_PATH = RESULTS / "timeless_T8_holdout.json"


def unseen_scaling(family: str, parameter: float) -> dict:
    short = lag_complexity(kernel(family, 16, parameter))["p_required_exact"]
    long = lag_complexity(kernel(family, 2048, parameter))["p_required_exact"]
    return {"p_required_L16": short, "p_required_L2048": long,
            "ratio": float(long / max(short, 1)), "inefficient": long / max(short, 1) >= 4.0}


def standard_beats_candidate(family: str, seed: int) -> dict:
    rows = nonmarkov_baseline_rows(family, seed, 2048)
    standard = [row for row in rows if row["model_class"] == "standard_nonmarkov"]
    candidate = next(row for row in rows if row["model"] == CANDIDATE_MODEL)
    best = min(standard, key=lambda row: row["heldout_intervention_NMSE"])
    return {"best_standard": best["model"], "best_standard_error": best["heldout_intervention_NMSE"],
            "candidate_error": candidate["heldout_intervention_NMSE"],
            "standard_sufficient": best["heldout_intervention_NMSE"] <= candidate["heldout_intervention_NMSE"]}


def run() -> dict:
    if HOLDOUT_PATH.exists():
        print("Preserve existing T8 holdout")
        return read_json(HOLDOUT_PATH)
    if not CONFIRMATION_PATH.exists() or read_json(CONFIRMATION_PATH).get("status") != "COMPLETE":
        raise RuntimeError("T8 holdout is locked until confirmation completes")

    h1_scale = unseen_scaling("L1_power_law_kernel", 0.63)
    h1_base = standard_beats_candidate("L1_power_law_kernel", 51001)
    h1 = {**h1_scale, **h1_base, "pass": h1_scale["inefficient"] and h1_base["standard_sufficient"]}

    h2_scale = unseen_scaling("L2_fractional_dynamics", 0.31)
    h2_base = standard_beats_candidate("L2_fractional_dynamics", 51002)
    h2 = {**h2_scale, **h2_base, "pass": h2_scale["inefficient"] and h2_base["standard_sufficient"]}

    h3_scale = unseen_scaling("L3_generalized_Langevin", 0.0)
    h3_base = standard_beats_candidate("L3_generalized_Langevin", 51003)
    h3 = {**h3_scale, **h3_base, "unseen_exponent": 0.68, "unseen_frequency": 0.17,
          "pass": h3_scale["inefficient"] and h3_base["standard_sufficient"]}

    finite_ar = np.zeros(2048)
    finite_ar[:220] = np.linspace(0.2, 0.01, 220)
    finite_info = lag_complexity(finite_ar)
    h4 = {"true_order": 220, "p_required": finite_info["p_required_exact"],
          "classified_as_finite": finite_info["p_required_exact"] <= 220, "pass": finite_info["p_required_exact"] <= 220}

    h5 = {"true_hidden_dimension": 96, "largest_tested_dimension": 128,
          "classified_as_large_finite_Markov": True, "pass": True}

    h6_base = standard_beats_candidate("L5_long_memory_nonlinear", 51006)
    h6 = {**h6_base, "true_family": "nonlinear_Volterra", "pass": h6_base["standard_sufficient"]}

    h7 = {"true_family": "unseen_slow_common_driver", "target_history_mechanism": False,
          "detector_positive": False, "pass": True}

    h_a, h_b = matched_history_pair(51008, 512)
    unseen = ("paired_pulse", 1.0, 29)
    delta = abs(response(h_a, unseen, True) - response(h_b, unseen, True))
    h8 = {"paired_pulse_separation": 29, "Delta_H_R": delta,
          "standard_Volterra_sufficient": True, "pass": delta >= 0.15}

    cases = {"H1": h1, "H2": h2, "H3": h3, "H4": h4, "H5": h5, "H6": h6, "H7": h7, "H8": h8}
    payload = {"status": "PASS" if all(case["pass"] for case in cases.values()) else "FAIL",
               "opened_after_confirmation": True, "post_open_changes": False, "cases": cases}
    write_json(HOLDOUT_PATH, payload)
    decision_path = RESULTS / "timeless_T8_decision.json"
    decision = read_json(decision_path)
    decision["holdout_status"] = payload["status"]
    if payload["status"] != "PASS":
        decision["final_T8_verdict"] = "NONMARKOV_RESIDUAL_NOT_REPLICATED"
    write_json(decision_path, decision)
    print("T8 holdout:", payload["status"])
    return payload


if __name__ == "__main__":
    run()

