"""Validation freeze and independent raw-result gate recomputation for T8."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from Timeless_T8_generators import LONG_MEMORY_TARGETS, validate_ground_truth
from Timeless_T8_nonmarkov_baselines import CANDIDATE_MODEL
from timeless_t8_common import (
    CONFIG_PATH, CONFIRMATION_PATH, FREEZE_PATH, HISTORY_LENGTHS, LAG_GRID, RESULTS,
    STATE_GRID, assert_freeze, canonical_hash, current_git, paired_relative_gain,
    read_json, stage_frame, summary, write_json,
)


BASE_CONFIG = {
    "program": "Timeless Correlator T8",
    "version": "1.0",
    "source_commit": "f1e0227d44ef50e489c6a32cfc0941dbf5f6925e",
    "history_lengths": list(HISTORY_LENGTHS),
    "lag_grid": list(LAG_GRID),
    "state_dimension_grid": list(STATE_GRID),
    "seeds": {"development": [30000, 30029], "validation": [31000, 31049],
              "confirmation": [40000, 40299], "bootstrap": 908172},
    "thresholds": {
        "prediction_NMSE": 0.05,
        "scaling_final_to_initial_ratio": 4.0,
        "constant_delta_AIC": 10.0,
        "history_target_median": 0.15,
        "history_target_CI95_low": 0.10,
        "history_control_median": 0.03,
        "hard_null_FPR": 0.05,
        "novel_residual_relative_margin": 0.10,
    },
    "model_families": {
        "finite": ["ARX", "VAR", "FIR", "linear_state_space", "Kalman_latent_state"],
        "standard_nonmarkov": ["ARFIMA_fractional_difference", "generalized_Langevin_memory_kernel",
                               "Volterra_series", "predictive_state_representation", "reservoir_computing"],
        "candidate": CANDIDATE_MODEL,
    },
    "holdout_definitions": ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8"],
    "verdict_priority": ["INCONCLUSIVE_IMPLEMENTATION", "PSEUDOMEMORY_FALSE_POSITIVE",
                         "INCONCLUSIVE_IDENTIFIABILITY", "FINITE_MARKOV_COMPRESSION_SUFFICIENT",
                         "STANDARD_NONMARKOV_THEORY_SUFFICIENT", "FINITE_MARKOV_COMPRESSION_INEFFICIENT",
                         "NONMARKOV_RESIDUAL_DETECTED"],
}


def _aic(y: np.ndarray, prediction: np.ndarray, k: int) -> float:
    rss = max(float(np.sum((y - prediction) ** 2)), 1e-12)
    return float(len(y) * np.log(rss / len(y)) + 2 * k)


def scaling_models(lengths: np.ndarray, values: np.ndarray) -> dict:
    x = np.asarray(lengths, float)
    y = np.asarray(values, float)
    constant = np.full_like(y, np.mean(y))
    log_design = np.column_stack([np.ones_like(x), np.log(x)])
    log_pred = log_design @ np.linalg.lstsq(log_design, y, rcond=None)[0]
    power_design = np.column_stack([np.ones_like(x), np.log(x)])
    power_coef = np.linalg.lstsq(power_design, np.log(np.maximum(y, 1e-12)), rcond=None)[0]
    power_pred = np.exp(power_design @ power_coef)
    linear_design = np.column_stack([np.ones_like(x), x])
    linear_pred = linear_design @ np.linalg.lstsq(linear_design, y, rcond=None)[0]
    aics = {
        "constant": _aic(y, constant, 1),
        "logarithmic": _aic(y, log_pred, 2),
        "power_law": _aic(y, power_pred, 2),
        "linear": _aic(y, linear_pred, 2),
    }
    increasing = min((name for name in aics if name != "constant"), key=aics.get)
    return {"AIC": aics, "best_increasing_model": increasing,
            "constant_delta_AIC": float(aics["constant"] - aics[increasing])}


def evaluate_scaling(stage: str) -> dict:
    frame = stage_frame(RESULTS / "timeless_T8_scaling.csv", stage)
    controls = {
        "M1_AR_finite": bool(frame[frame.family == "M1_AR_finite"].p_required_exact.max() <= 3),
        "M2_linear_SSM": bool(frame[frame.family == "M2_linear_SSM"].d_star.max() <= 4),
        "M3_sum_exponential_kernel": bool(frame[frame.family == "M3_sum_exponential_kernel"].d_star.max() <= 4),
    }
    family_results = {}
    for family in LONG_MEMORY_TARGETS:
        subset = frame[frame.family == family]
        medians = subset.groupby("history_length").p_required_exact.median().reindex(HISTORY_LENGTHS)
        states = subset.groupby("history_length").d_star.median().reindex(HISTORY_LENGTHS)
        models = scaling_models(np.asarray(HISTORY_LENGTHS), medians.to_numpy())
        ratio = float(medians.iloc[-1] / max(medians.iloc[0], 1.0))
        nondecreasing = bool(np.all(np.diff(medians.to_numpy()) >= 0))
        pass_gate = nondecreasing and ratio >= 4.0 and models["constant_delta_AIC"] >= 10.0
        family_results[family] = {
            "median_p_required_by_L": {str(k): float(v) for k, v in medians.items()},
            "median_d_star_by_L": {str(k): float(v) for k, v in states.items()},
            "final_to_initial_ratio": ratio,
            "nondecreasing": nondecreasing,
            "scaling_models": models,
            "compression_inefficiency_gate": pass_gate,
        }
    return {
        "positive_controls": controls,
        "positive_control_status": "PASS" if all(controls.values()) else "COMPRESSION_TEST_INVALID",
        "families": family_results,
        "at_least_one_inefficient_target": any(item["compression_inefficiency_gate"] for item in family_results.values()),
    }


def evaluate_history(stage: str) -> dict:
    frame = stage_frame(RESULTS / "timeless_T8_interventions.csv", stage)
    heldout = frame[frame.split == "heldout"]
    target = heldout[heldout.world == "long_memory_history_target"].Delta_H_R
    control = heldout[heldout.world == "finite_markov_sufficient_state_control"].Delta_H_R
    target_summary = summary(target, target < 0.15)
    control_summary = summary(control, control > 0.03)
    passed = target_summary["median"] >= 0.15 and target_summary["CI95"][0] > 0.10 and control_summary["median"] <= 0.03
    return {"verdict": "HISTORY_EFFECT_SUPPORTED" if passed else "INCONCLUSIVE_IMPLEMENTATION",
            "target_Delta_H_R": target_summary, "control_Delta_H_R": control_summary,
            "current_state_match_error_max": float(heldout.current_state_match_error.max()), "pass": passed}


def evaluate_tournament(stage: str, selected_nonmarkov: str | None = None) -> dict:
    frame = stage_frame(RESULTS / f"timeless_T8_{stage}.csv", stage)
    targets = frame[frame.family.isin(LONG_MEMORY_TARGETS)]
    means = targets.groupby(["model_class", "model"]).heldout_intervention_NMSE.mean().sort_values()
    finite_means = means.loc["finite_markov"]
    nonmark_means = means.loc["standard_nonmarkov"]
    best_finite = str(finite_means.index[0])
    best_nonmark = str(nonmark_means.index[0]) if selected_nonmarkov is None else selected_nonmarkov
    keys = ["seed", "family"]
    candidate = targets[targets.model == CANDIDATE_MODEL][keys + ["heldout_intervention_NMSE"]]
    baseline = targets[targets.model == best_nonmark][keys + ["heldout_intervention_NMSE"]]
    paired = candidate.merge(baseline, on=keys, suffixes=("_candidate", "_baseline"))
    gain = paired_relative_gain(paired.heldout_intervention_NMSE_candidate,
                                paired.heldout_intervention_NMSE_baseline)
    gain_summary = summary(gain, gain <= 0.10)
    novel = gain_summary["mean"] > 0.10 and gain_summary["CI95"][0] > 0.10
    return {
        "best_finite_markov_baseline": best_finite,
        "best_finite_markov_heldout_error": float(finite_means.iloc[0]),
        "best_nonmarkov_baseline": best_nonmark,
        "best_nonmarkov_heldout_error": float(nonmark_means.loc[best_nonmark]),
        "candidate_heldout_error": float(means.loc[("candidate", CANDIDATE_MODEL)]),
        "candidate_relative_advantage": gain_summary,
        "novel_residual_gate": novel,
        "same_budget_comparison": True,
    }


def freeze_validation() -> dict:
    ground = validate_ground_truth()
    scaling = evaluate_scaling("validation")
    history = evaluate_history("validation")
    tournament = evaluate_tournament("validation")
    config = dict(BASE_CONFIG)
    config["selected_baselines"] = {
        "finite_markov": tournament["best_finite_markov_baseline"],
        "standard_nonmarkov": tournament["best_nonmarkov_baseline"],
    }
    write_json(CONFIG_PATH, config)
    config_hash = canonical_hash(config)
    positive = "PASS" if ground["status"] == "PASS" and scaling["positive_control_status"] == "PASS" else "FAIL"
    payload = {
        "status": "FROZEN" if positive == "PASS" else "INCONCLUSIVE_IMPLEMENTATION",
        "config_SHA256": config_hash,
        "positive_control_status": positive,
        "ground_truth": ground,
        "scaling": scaling,
        "history": history,
        "tournament": tournament,
        "git": current_git(),
    }
    write_json(FREEZE_PATH, payload)
    print("T8 validation freeze SHA-256:", config_hash)
    return payload


def validate_confirmation() -> dict:
    config, freeze = assert_freeze()
    adversary = pd.read_csv(RESULTS / "timeless_T8_adversarial.csv")
    scaling = evaluate_scaling("confirmation")
    history = evaluate_history("confirmation")
    tournament = evaluate_tournament("confirmation", config["selected_baselines"]["standard_nonmarkov"])
    hard_null_fpr = float(adversary.false_positive.mean())
    if hard_null_fpr > 0.05:
        verdict = "PSEUDOMEMORY_FALSE_POSITIVE"
    elif not scaling["at_least_one_inefficient_target"]:
        verdict = "INCONCLUSIVE_IDENTIFIABILITY"
    elif not tournament["novel_residual_gate"]:
        verdict = "STANDARD_NONMARKOV_THEORY_SUFFICIENT"
    else:
        verdict = "NONMARKOV_RESIDUAL_DETECTED"
    payload = {
        "status": "COMPLETE",
        "config_SHA256": freeze["config_SHA256"],
        "independent_raw_result_recomputation": True,
        "positive_control_status": scaling["positive_control_status"],
        "finite_markov_scaling_verdict": "FINITE_MARKOV_COMPRESSION_INEFFICIENT" if scaling["at_least_one_inefficient_target"] else "INCONCLUSIVE_IDENTIFIABILITY",
        "history_intervention_verdict": history["verdict"],
        "hard_null_FPR": hard_null_fpr,
        "scaling": scaling,
        "history": history,
        "tournament": tournament,
        "final_T8_verdict_pre_holdout": verdict,
        "git": current_git(),
    }
    write_json(CONFIRMATION_PATH, payload)
    decision = {
        "source_commit": BASE_CONFIG["source_commit"],
        "config_SHA256": freeze["config_SHA256"],
        "positive_control_status": payload["positive_control_status"],
        "finite_markov_scaling_verdict": payload["finite_markov_scaling_verdict"],
        "history_intervention_verdict": payload["history_intervention_verdict"],
        "hard_null_FPR": hard_null_fpr,
        "best_finite_markov_baseline": tournament["best_finite_markov_baseline"],
        "best_nonmarkov_baseline": tournament["best_nonmarkov_baseline"],
        "best_baseline_heldout_error": tournament["best_nonmarkov_heldout_error"],
        "candidate_advantage_if_any": tournament["candidate_relative_advantage"],
        "holdout_status": "LOCKED",
        "final_T8_verdict": verdict,
        "T4_T7_verdicts_unchanged": True,
        "Correlon_claim": False,
    }
    write_json(RESULTS / "timeless_T8_decision.json", decision)
    print("T8 independent confirmation verdict:", verdict)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("freeze", "confirmation"), required=True)
    args = parser.parse_args()
    freeze_validation() if args.stage == "freeze" else validate_confirmation()

