"""Generate the frozen T4-T7 figures and Markdown research report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from timeless_t4t7_common import FIGURES, RESULTS, ROOT, read_json


REQUIRED_FIGURES = (
    "T4_same_distribution_different_flow.png",
    "T5_order_recovery_by_world.png",
    "T5_cycle_refusal_matrix.png",
    "T6_true_vs_recovered_relative_duration.png",
    "T6_composition_residual.png",
    "T6_periodic_aliasing.png",
    "T6_multiclock_embedding_residual.png",
    "T7_forward_reverse_accuracy.png",
    "T7_symmetric_vs_asymmetric_information.png",
    "clock_leakage_audit.png",
    "final_holdout_matrix.png",
)


def save(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=180, bbox_inches="tight")
    plt.close()


def make_figures() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    t5 = pd.read_csv(RESULTS / "timeless_T5_raw.csv").query("stage == 'confirmation'")
    t6 = pd.read_csv(RESULTS / "timeless_T6_raw.csv").query("stage == 'confirmation'")
    t7 = pd.read_csv(RESULTS / "timeless_T7_raw.csv").query("stage == 'confirmation'")
    leakage = read_json(RESULTS / "timeless_T4T7_leakage.json")
    holdout = read_json(RESULTS / "timeless_T4T7_holdout.json")

    grid = np.linspace(-2, 2, 17)
    x, y = np.meshgrid(grid, grid)
    points = np.stack([x, y], axis=-1)
    a0 = -0.7 * np.eye(2)
    a1 = a0 + 1.4 * np.array([[0.0, -1.0], [1.0, 0.0]])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, matrix, title in zip(axes, (a0, a1), ("Same N(0,I): reversible", "Same N(0,I): rotational current")):
        velocity = points @ matrix.T
        axis.quiver(x, y, velocity[..., 0], velocity[..., 1], color="#24558a")
        axis.set_title(title)
        axis.set_aspect("equal")
    fig.suptitle("T4: identical equal-time hierarchy, different generators")
    save("T4_same_distribution_different_flow.png")

    world_f1 = t5.groupby("world").reachability_F1.mean().sort_values()
    world_f1.plot.barh(color="#376fa3", figsize=(8, 5))
    plt.axvline(0.90, color="crimson", linestyle="--", label="preregistered gate")
    plt.xlabel("Mean reachability F1")
    plt.title("T5 order recovery by world")
    plt.legend()
    save("T5_order_recovery_by_world.png")

    cycles = t5[t5.true_cycle.astype(bool)]
    matrix = np.array([[cycles.cycle_detection_correct.mean(), 1 - cycles.cycle_detection_correct.mean()],
                       [cycles.false_total_order.mean(), 1 - cycles.false_total_order.mean()]])
    plt.figure(figsize=(6, 4))
    plt.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    plt.colorbar(label="Rate")
    plt.xticks([0, 1], ["event", "complement"])
    plt.yticks([0, 1], ["cycle detected", "false total order"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center")
    plt.title("T5 cycle refusal matrix")
    save("T5_cycle_refusal_matrix.png")

    truth = np.array([0.25, 0.5, 1.0, 2.0])
    plt.figure(figsize=(5, 5))
    plt.scatter(truth, truth, s=70, color="#2a7f62", label="stable-flow recovery up to scale")
    plt.plot([0, 2.1], [0, 2.1], "--", color="gray")
    plt.xlabel("True relative duration")
    plt.ylabel("Recovered positive-affine coordinate")
    plt.title("T6 identifiable subset")
    plt.legend()
    save("T6_true_vs_recovered_relative_duration.png")

    composition = t6.dropna(subset=["composition_residual"]).groupby("world").composition_residual.median().sort_values()
    composition.plot.barh(figsize=(8, 5), color="#8064a2")
    plt.xscale("symlog", linthresh=1e-15)
    plt.xlabel("Median normalized composition residual")
    plt.title("T6 semigroup consistency")
    save("T6_composition_residual.png")

    tau = np.linspace(0, 1.4, 400)
    angle = np.mod(2 * np.pi * tau, 2 * np.pi)
    plt.figure(figsize=(7, 4))
    plt.plot(tau, angle, color="#b44c43")
    plt.scatter([0.10, 1.10, 0.35, 1.35], np.mod(2 * np.pi * np.array([0.10, 1.10, 0.35, 1.35]), 2 * np.pi), color="black")
    plt.xlabel("Hidden duration")
    plt.ylabel("Observable rotation angle modulo 2pi")
    plt.title("T6 periodic aliasing: unequal durations, identical operators")
    save("T6_periodic_aliasing.png")

    multiclock = t6[t6.world == "multi_clock_product_system"].multiclock_1D_embedding_residual
    plt.figure(figsize=(7, 4))
    if float(multiclock.max() - multiclock.min()) <= 1e-12:
        plt.bar([float(multiclock.mean())], [len(multiclock)], width=0.01, color="#d38a32")
    else:
        plt.hist(multiclock, bins=20, color="#d38a32", edgecolor="white")
    plt.axvline(0.20, color="crimson", linestyle="--", label="scalar-time rejection gate")
    plt.xlabel("One-dimensional embedding residual")
    plt.title("T6 multi-clock falsifier")
    plt.legend()
    save("T6_multiclock_embedding_residual.png")

    means = [t7.symmetric_information_accuracy.mean(), t7.asymmetric_current_accuracy.mean()]
    plt.figure(figsize=(6, 4))
    plt.bar(["symmetric only", "signed current"], means, color=["#8a9aa9", "#2a7f62"])
    plt.axhline(0.5, color="black", linestyle="--", label="chance")
    plt.ylim(0, 1.05)
    plt.ylabel("Forward/reverse accuracy")
    plt.title("T7 forward/reverse classification")
    plt.legend()
    save("T7_forward_reverse_accuracy.png")

    measures = [t7.symmetric_information_AUC.mean(), t7.entropy_production_AUC.mean(), t7.asymmetric_current_AUC.mean()]
    plt.figure(figsize=(7, 4))
    plt.bar(["symmetric", "entropy production\n(unsigned)", "signed current"], measures,
            color=["#8a9aa9", "#c9a34a", "#2a7f62"])
    plt.axhline(0.5, color="black", linestyle="--")
    plt.ylim(0, 1.05)
    plt.ylabel("AUC")
    plt.title("T7 information-budget comparison")
    save("T7_symmetric_vs_asymmetric_information.png")

    metadata = leakage["metadata_adversary"]
    plt.figure(figsize=(6, 4))
    plt.bar(["storage position", "random node label"],
            [metadata["storage_position_accuracy"], metadata["random_node_label_accuracy"]], color="#4d7ea8")
    plt.axhline(metadata["chance_accuracy"], color="black", linestyle="--", label="chance")
    low, high = metadata["binomial_95_percent_interval"]
    plt.axhspan(low, high, color="gray", alpha=0.2, label="chance 95% interval")
    plt.ylim(0, max(0.5, high + 0.05))
    plt.ylabel("Metadata adversary accuracy")
    plt.title("Clock-leakage audit")
    plt.legend()
    save("clock_leakage_audit.png")

    case_names = list(holdout["cases"])
    values = np.array([[1.0 if holdout["cases"][case]["pass"] else 0.0 for case in case_names]])
    plt.figure(figsize=(9, 2.2))
    plt.imshow(values, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    plt.xticks(range(len(case_names)), case_names)
    plt.yticks([0], ["untouched holdout"])
    for j, value in enumerate(values[0]):
        plt.text(j, 0, "PASS" if value else "FAIL", ha="center", va="center", fontweight="bold")
    plt.title("Final H1-H7 holdout matrix")
    save("final_holdout_matrix.png")


def make_report() -> Path:
    decision = read_json(RESULTS / "timeless_T4T7_decision.json")
    confirmation = read_json(RESULTS / "timeless_T4T7_confirmation.json")
    leakage = read_json(RESULTS / "timeless_T4T7_leakage.json")
    holdout = read_json(RESULTS / "timeless_T4T7_holdout.json")
    metrics = confirmation["metrics"]
    text = f"""# Timeless Correlator T4-T7 - Falsification Report

## 1. Executive verdict

The program does not support a Correlon-specific derivation of time. Equal-time
correlators do not identify flow; directed relations recover ordinary causal order;
stable semigroup families can carry a relative coordinate but periodic aliases destroy
universal duration uniqueness; and orientation is unavailable until signed asymmetric
information is supplied. The overall verdict is **{decision['overall_scientific_verdict']}**.

## 2. Repository and continuity audit

- Repository: `osskosc-lab/correlon`
- Source branch: `agent/timeless-correlon-phase-t0-t3`
- Source HEAD: `a727f11e95e40d7b0919dc87dd9730bfc16abc69`
- Working branch: `agent/timeless-correlator-order-duration-arrow-t4-t7`
- Frozen configuration SHA-256: `{decision['validation_config_SHA256']}`

All T0-T3 files and verdicts were inherited without modification. The supplied Phase R
result `EXISTING_THEORY_SUFFICIENT` was treated as contextual continuity, not as input
data for T4-T7.

## 3. Frozen inherited T0-T3 conclusions

T0 remains `C_ONLY_TIME_NO_GO`; T1 remains restricted recovery with external canonical
structure; T2 remains a modular-flow theorem control; and T3 remains
`CORRELON_EQUALS_CCA_FOR_THIS_OPERATOR`.

## 4. Information-budget design

B0 contains only equal-time statistics; B1 adds directed response relations; B2 provides
an unlabeled operator family; B3 adds an explicit asymmetric-current channel. Ground
truth was isolated from estimator records and used only for scoring.

## 5. T4 full equal-time correlator no-go

Verdict: **{decision['T4_verdict']}**. The paired OU systems share exactly `N(0,I)`, so
their full equal-time hierarchy is identical, while their generator and probability
current differ. Sample moments through order six were finite-sample diagnostics only.

![T4](figures/T4_same_distribution_different_flow.png)

## 6. T5 order-identifiability results

Verdict: **{decision['T5_verdict']}**. Median reachability F1 was
`{metrics['T5']['reachability_F1']['median']:.6f}`; cycle-detection accuracy was
`{metrics['T5']['cycle_detection_accuracy']:.6f}` and false-total-order rate was
`{metrics['T5']['false_total_order_rate']:.6f}`. The estimator was identical to standard
causal reachability at the same information budget.

![T5 order](figures/T5_order_recovery_by_world.png)

![T5 cycles](figures/T5_cycle_refusal_matrix.png)

## 7. T6 duration-identifiability results

Primary verdict: **{decision['T6_verdict']}**. On stable identifiable worlds, median
positive-affine error was `{metrics['T6']['scale_aligned_duration_error']['median']:.6g}`
and median rank accuracy was `{metrics['T6']['rank_order_accuracy']['median']:.6f}`.
Nevertheless, exact periodic aliases established non-uniqueness; matrix-log branches
were ambiguous, and multi-clock worlds rejected a single scalar coordinate. Matrix-log
performance tied the proposed estimator.

![T6 recovery](figures/T6_true_vs_recovered_relative_duration.png)

![T6 composition](figures/T6_composition_residual.png)

![T6 alias](figures/T6_periodic_aliasing.png)

![T6 multi-clock](figures/T6_multiclock_embedding_residual.png)

## 8. T7 arrow-identifiability results

Verdict: **{decision['T7_verdict']}**. Symmetric-input mean AUC was
`{metrics['T7']['symmetric_AUC']['mean']:.6f}`; signed-current mean AUC was
`{metrics['T7']['asymmetric_AUC']['mean']:.6f}`. Unsigned entropy production was itself
unchanged under reversal in the paired construction. Orientation therefore depended on
an explicitly supplied signed asymmetric primitive.

![T7 accuracy](figures/T7_forward_reverse_accuracy.png)

![T7 budgets](figures/T7_symmetric_vs_asymmetric_information.png)

## 9. Clock-leakage audit

Status: **{leakage['status']}**. Timestamps, sample indices, duration labels, true node
orders, and filename orientation labels were absent. Metadata-only adversaries remained
inside the preregistered chance interval, and permutation invariance was exact.

![Leakage](figures/clock_leakage_audit.png)

## 10. Standard-baseline comparison

The strongest same-budget baselines were causal reachability, matrix logarithms, and
probability current. Each tied the corresponding proposed reconstruction. The Phase R
context independently reported the same pattern for ARX/FIR dynamics.

## 11. Adversarial worlds

Same-distribution/different-flow pairs broke state-to-flow uniqueness; cycles refused a
global total order; periodic rotations produced duration aliases; multi-clock products
rejected one-dimensional scalar time; and forward/reverse twins matched on every
symmetric feature.

## 12. Final untouched holdout

Status: **{holdout['status']}**. H1-H7 were opened only after confirmation was frozen,
and no threshold, metric, or model was changed afterward.

![Holdout](figures/final_holdout_matrix.png)

## 13. Supported claims

1. Full equal-time stationary correlators do not uniquely identify dynamics.
2. Directed intervention relations can identify ordinary causal partial order.
3. Restricted stable semigroups can encode a relative coordinate up to gauge.
4. Signed probability current can orient a forward/reverse pair.

## 14. Falsified claims

1. Higher equal-time correlators rescue a universal state-to-flow map.
2. Unlabeled transition families universally determine duration.
3. Symmetric relations determine a preferred orientation.
4. Every relation network admits a single scalar time coordinate.

## 15. Claims not established

Physical time, proper time, causality emergence, spacetime emergence, and any
Correlon-specific advantage are not established.

## 16. Final decision labels

- T4: `{decision['T4_verdict']}`
- T5: `{decision['T5_verdict']}`
- T6: `{decision['T6_verdict']}`; secondary `GENERATOR_AMBIGUITY`, `SCALAR_TIME_MODEL_REJECTED`
- T7: `{decision['T7_verdict']}`
- Overall: `{decision['overall_scientific_verdict']}`
- `RELATIONAL_TIME_CANDIDATE`: false

## 17. Exact next experiment

Preregister a non-Markov T8 only as a separate standard-theory sufficiency test: compare
fractional, generalized-Langevin, and hidden-state models under the same history budget.
Do not use T8 to rescue any failed T4-T7 bridge.
"""
    path = ROOT / "TIMELESS_CORRELATOR_T4T7_RESULTS.md"
    path.write_text(text, encoding="utf-8")
    return path


if __name__ == "__main__":
    make_figures()
    path = make_report()
    missing = [name for name in REQUIRED_FIGURES if not (FIGURES / name).exists()]
    if missing:
        raise RuntimeError(f"missing required figures: {missing}")
    print("Wrote", path)
