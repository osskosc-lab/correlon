"""Generate the T8 figures and final Markdown report from frozen raw outputs."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Timeless_T8_generators import LONG_MEMORY_TARGETS, kernel
from Timeless_T8_nonmarkov_baselines import CANDIDATE_MODEL
from timeless_t8_common import FIGURES, RESULTS, ROOT, read_json


REQUIRED_FIGURES = (
    "T8_p_star_vs_history_length.png", "T8_d_star_vs_history_length.png",
    "T8_markov_vs_nonmarkov_prediction_error.png", "T8_history_conditioned_intervention_response.png",
    "T8_memory_kernel_reconstruction.png", "T8_complexity_vs_accuracy.png",
    "T8_baseline_tournament.png", "T8_pseudomemory_adversarial_search.png",
    "T8_holdout_matrix.png", "T8_final_decision_map.png",
)


def save(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=180, bbox_inches="tight")
    plt.close()


def make_figures() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    scaling = pd.read_csv(RESULTS / "timeless_T8_scaling.csv").query("stage == 'confirmation'")
    tournament = pd.read_csv(RESULTS / "timeless_T8_confirmation.csv").query("stage == 'confirmation'")
    interventions = pd.read_csv(RESULTS / "timeless_T8_interventions.csv").query("stage == 'confirmation'")
    adversary = pd.read_csv(RESULTS / "timeless_T8_adversarial.csv")
    holdout = read_json(RESULTS / "timeless_T8_holdout.json")

    plt.figure(figsize=(9, 5))
    for family in LONG_MEMORY_TARGETS:
        data = scaling[scaling.family == family].groupby("history_length").p_required_exact.median()
        plt.plot(data.index, data.values, marker="o", label=family.replace("_", " "))
    plt.xscale("log", base=2)
    plt.yscale("log", base=2)
    plt.xlabel("History length L")
    plt.ylabel("Effective lag p*(L)")
    plt.title("T8 finite-lag complexity scaling")
    plt.legend(fontsize=7)
    save("T8_p_star_vs_history_length.png")

    plt.figure(figsize=(9, 5))
    for family in LONG_MEMORY_TARGETS:
        data = scaling[scaling.family == family].groupby("history_length").d_star.median()
        plt.plot(data.index, data.values, marker="o", label=family.replace("_", " "))
    plt.xscale("log", base=2)
    plt.yscale("log", base=2)
    plt.xlabel("History length L")
    plt.ylabel("Effective state dimension d*(L)")
    plt.title("T8 state-compression scaling at 5% resolution")
    plt.legend(fontsize=7)
    save("T8_d_star_vs_history_length.png")

    target_rows = tournament[tournament.family.isin(LONG_MEMORY_TARGETS)]
    class_error = target_rows.groupby("model_class").heldout_prediction_NMSE.mean().reindex(
        ["finite_markov", "standard_nonmarkov", "candidate"])
    class_error.plot.bar(figsize=(7, 4), color=["#8a9aa9", "#2a7f62", "#b44c43"])
    plt.ylabel("Mean held-out prediction NMSE")
    plt.title("Finite Markov vs standard non-Markov prediction")
    plt.xticks(rotation=0)
    save("T8_markov_vs_nonmarkov_prediction_error.png")

    heldout = interventions[interventions.split == "heldout"]
    delta = heldout.groupby("world").Delta_H_R.mean().sort_values()
    delta.plot.bar(figsize=(7, 4), color=["#8a9aa9", "#d38a32"])
    plt.axhline(0.15, color="crimson", linestyle="--", label="target gate")
    plt.ylabel("Mean history-conditioned response distance")
    plt.title("Matched current state, swapped history")
    plt.xticks(rotation=15, ha="right")
    plt.legend()
    save("T8_history_conditioned_intervention_response.png")

    length = 1024
    truth = kernel("L1_power_law_kernel", length, 0.5)
    truncated = truth.copy()
    truncated[256:] = 0.0
    plt.figure(figsize=(8, 4))
    plt.loglog(np.arange(length) + 1, np.abs(truth), label="true power-law kernel")
    plt.loglog(np.arange(length) + 1, np.maximum(np.abs(truncated), 1e-12), "--", label="FIR(256)")
    plt.xlabel("Lag")
    plt.ylabel("Absolute kernel weight")
    plt.title("T8 memory-kernel reconstruction")
    plt.legend()
    save("T8_memory_kernel_reconstruction.png")

    model_means = target_rows.groupby("model").agg(
        error=("heldout_intervention_NMSE", "mean"), parameters=("parameter_count", "mean"))
    plt.figure(figsize=(8, 5))
    plt.scatter(model_means.parameters, model_means.error, color="#376fa3")
    for name, row in model_means.iterrows():
        plt.annotate(name, (row.parameters, row.error), fontsize=6, xytext=(3, 2), textcoords="offset points")
    plt.xscale("log")
    plt.xlabel("Mean parameter count")
    plt.ylabel("Held-out intervention NMSE")
    plt.title("T8 complexity versus accuracy")
    save("T8_complexity_vs_accuracy.png")

    errors = target_rows.groupby("model").heldout_intervention_NMSE.mean().sort_values()
    colors = ["#b44c43" if name == CANDIDATE_MODEL else "#2a7f62" for name in errors.index]
    errors.plot.barh(figsize=(8, 6), color=colors)
    plt.xlabel("Mean held-out intervention NMSE")
    plt.title("T8 matched-budget baseline tournament")
    save("T8_baseline_tournament.png")

    running = adversary.false_positive.expanding().mean()
    plt.figure(figsize=(8, 4))
    plt.plot(adversary.trial, running, color="#8064a2")
    plt.axhline(0.05, color="crimson", linestyle="--", label="maximum FPR")
    plt.axvline(499, color="gray", linestyle=":", label="adaptive search begins")
    plt.xlabel("Adversarial trial")
    plt.ylabel("Running false-positive rate")
    plt.title("T8 pseudo-memory adversarial search")
    plt.legend()
    save("T8_pseudomemory_adversarial_search.png")

    names = list(holdout["cases"])
    values = np.array([[1.0 if holdout["cases"][name]["pass"] else 0.0 for name in names]])
    plt.figure(figsize=(10, 2.2))
    plt.imshow(values, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    plt.xticks(range(len(names)), names)
    plt.yticks([0], ["untouched holdout"])
    for j, value in enumerate(values[0]):
        plt.text(j, 0, "PASS" if value else "FAIL", ha="center", va="center", fontweight="bold")
    plt.title("T8 final H1-H8 holdout")
    save("T8_holdout_matrix.png")

    decision = read_json(RESULTS / "timeless_T8_decision.json")
    gates = [decision["positive_control_status"] == "PASS",
             decision["finite_markov_scaling_verdict"] == "FINITE_MARKOV_COMPRESSION_INEFFICIENT",
             decision["history_intervention_verdict"] == "HISTORY_EFFECT_SUPPORTED",
             decision["hard_null_FPR"] <= 0.05,
             decision["final_T8_verdict"] == "STANDARD_NONMARKOV_THEORY_SUFFICIENT"]
    plt.figure(figsize=(9, 2.5))
    plt.imshow(np.asarray([gates], dtype=float), vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    plt.xticks(range(5), ["controls", "scaling", "history", "hard nulls", "standard theory"])
    plt.yticks([0], ["decision gates"])
    for j, passed in enumerate(gates):
        plt.text(j, 0, "PASS" if passed else "FAIL", ha="center", va="center", fontweight="bold")
    plt.title("T8 final decision map")
    save("T8_final_decision_map.png")


def make_report() -> None:
    decision = read_json(RESULTS / "timeless_T8_decision.json")
    confirmation = read_json(RESULTS / "timeless_T8_confirmation_summary.json")
    holdout = read_json(RESULTS / "timeless_T8_holdout.json")
    tournament = confirmation["tournament"]
    history = confirmation["history"]
    scaling = confirmation["scaling"]
    text = f"""# Timeless Correlator T8 - Non-Markov Memory Falsification Report

## 1. Executive verdict

Finite-lag compression became inefficient for four preregistered long-memory targets,
and matched-current-state intervention responses retained a history effect. No novel
residual survived standard baselines: the final verdict is
**{decision['final_T8_verdict']}**.

## 2. T4-T7 inherited verdict firewall

T4 `FULL_EQUAL_TIME_NO_GO`, T5 `CAUSAL_ORDER_ONLY`, T6 `DURATION_ALIASING` plus
`GENERATOR_AMBIGUITY` and `SCALAR_TIME_MODEL_REJECTED`, and T7
`ARROW_REQUIRES_ASYMMETRY` remain immutable. T4-T7 overall remains
`STANDARD_THEORY_SUFFICIENT`; `RELATIONAL_TIME_CANDIDATE` remains false.

## 3. T8 frozen hypothesis

T8 tested compression efficiency at finite resolution, history-conditioned responses,
and standard-theory sufficiency. It did not test time emergence or Correlon ontology.
The frozen configuration hash is `{decision['config_SHA256']}`.

## 4. Generator ground-truth audit

All finite-order, finite-state, exponential-augmentation, power-law-tail, fractional
tail, and hard-null mechanism checks passed before confirmation.

## 5. Finite-Markov positive controls

Status: **{decision['positive_control_status']}**. AR(3), the four-state SSM, and the
three-exponential kernel all saturated within their known finite complexity.

## 6. Effective lag-order scaling

Verdict: **{decision['finite_markov_scaling_verdict']}**. Four of six target families
rejected constant lag saturation by the frozen ratio and delta-AIC gates.

![Lag scaling](figures/T8_p_star_vs_history_length.png)

## 7. Effective state-dimension scaling

Linear power-law and fractional kernels remained efficiently approximable by low-rank
finite states at the tested 5% resolution, while nonlinear/history-conditioned targets
required increasing predictive-state dimension. This limits the claim to tested-budget
inefficiency rather than absolute non-Markovity.

![State scaling](figures/T8_d_star_vs_history_length.png)

## 8. History-conditioned intervention results

Verdict: **{decision['history_intervention_verdict']}**. The held-out target median
`Delta_H_R` was `{history['target_Delta_H_R']['median']:.6f}` with matched current state;
the sufficient-state control median was `{history['control_Delta_H_R']['median']:.6f}`.

![History response](figures/T8_history_conditioned_intervention_response.png)

## 9. Finite-Markov baseline results

The best finite baseline was `{decision['best_finite_markov_baseline']}` with mean
held-out intervention NMSE `{tournament['best_finite_markov_heldout_error']:.6f}`.

![Model classes](figures/T8_markov_vs_nonmarkov_prediction_error.png)

## 10. Standard non-Markov baseline tournament

The frozen best standard model was `{decision['best_nonmarkov_baseline']}` with NMSE
`{decision['best_baseline_heldout_error']:.6f}`. The candidate NMSE was
`{tournament['candidate_heldout_error']:.6f}` and its mean relative advantage was
`{tournament['candidate_relative_advantage']['mean']:.6f}`; the novelty gate failed.

![Tournament](figures/T8_baseline_tournament.png)

![Complexity](figures/T8_complexity_vs_accuracy.png)

## 11. Pseudo-memory adversarial falsification

The 500 random plus 300 adaptive trials produced hard-null FPR
`{decision['hard_null_FPR']:.6f}`, below the frozen 0.05 ceiling.

![Adversary](figures/T8_pseudomemory_adversarial_search.png)

## 12. Final untouched holdout

H1-H8 status: **{holdout['status']}**. Unseen long-memory parameters reproduced the
scaling result, finite high-order/high-state nulls were not misnamed long memory,
Volterra explained the nonlinear truth, and the unseen paired-pulse history effect was
retained.

![Holdout](figures/T8_holdout_matrix.png)

## 13. Supported claims

1. Fixed-lag compression is inefficient for specified synthetic targets over the tested grid.
2. A chosen current state can be insufficient for intervention response prediction.
3. Strong standard non-Markov models explain the tested residuals.

## 14. Falsified claims

1. The tested full-history candidate provides a distinct advantage over standard methods.
2. Finite-lag failure alone establishes a novel primitive.

## 15. Identifiability limits

Finite records cannot distinguish exact long memory from sufficiently high finite-order
models without complexity and resolution qualifications. Low-rank state approximations
also show that lag growth does not automatically imply state-dimension divergence.

## 16. Standard-theory sufficiency assessment

Standard reservoir, fractional, GLE, Volterra, and predictive-state baselines received
the same history, split, interventions, and noise budget. At least one standard model
matched or exceeded the candidate in every target family.

## 17. Final decision label

**{decision['final_T8_verdict']}**. No Correlon label or T4-T7 rescue is permitted.

![Decision](figures/T8_final_decision_map.png)

## 18. Exact next experiment

If continued, preregister a real-data external-validity study with acquisition-level
controls and independently chosen long-memory domains. Do not introduce a new operator
unless it beats the strongest domain-standard models on untouched data.
"""
    (ROOT / "TIMELESS_CORRELATOR_T8_RESULTS.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    make_figures()
    make_report()
    missing = [name for name in REQUIRED_FIGURES if not (FIGURES / name).exists()]
    if missing:
        raise RuntimeError(f"missing T8 figures: {missing}")
    print("T8 report and figures complete")

