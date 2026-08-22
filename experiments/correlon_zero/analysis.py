"""Execution, aggregation, decision, plotting, and report helpers for Correlon Zero."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .adversary import run_adversarial_search
from .baselines import baseline_scores
from .generators import World, generate_world
from .metrics import correlon_score, retention
from .transforms import apply_destroy, apply_preserve
from .utils import ROOT, ensure_output_dirs, implementation_manifest, seed_for, write_json


CORRELON = "correlon_v1"
BLUE = "#2563eb"
ORANGE = "#ea580c"
GOLD = "#ca8a04"
OLIVE = "#4d7c0f"
PINK = "#db2777"
INK = "#172033"
GRID = "#d8dee9"


def metric_names(config: dict[str, Any]) -> list[str]:
    return [CORRELON, *config["baselines"]]


def _result_path(config: dict[str, Any], name: str) -> Path:
    return ROOT / config["outputs"]["results_dir"] / name


def _figure_path(config: dict[str, Any], name: str) -> Path:
    return ROOT / config["outputs"]["figures_dir"] / name


def score_world(world: World, config: dict[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    null_seed = seed_for(int(world.meta["seed"]), "metric_null", config)
    primary, diagnostics = correlon_score(world.x, world.y, config["metric"], null_seed)
    scores = {CORRELON: primary, **baseline_scores(world.x, world.y, config)}
    return scores, diagnostics


def evaluate_target_seed(
    seed: int, stage: str, config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    transform_rows: list[dict[str, Any]] = []
    world_rows: list[dict[str, Any]] = []
    for target in config["targets"]:
        world = generate_world(target, seed, config)
        if not world.meta["direct_edge"] or not world.meta["persistent_identity"]:
            raise AssertionError(f"target generator lacks its frozen mechanism: {target}")
        base_scores, diagnostics = score_world(world, config)
        by_metric: dict[str, dict[str, list[tuple[str, float]]]] = {
            metric: {"PRESERVE": [], "DESTROY": []} for metric in metric_names(config)
        }
        for transform_set, transforms in (
            ("PRESERVE", config["preserve_transformations"]),
            ("DESTROY", config["destroy_transformations"]),
        ):
            for transform_name in transforms:
                transformed = (
                    apply_preserve(world, transform_name, config)
                    if transform_set == "PRESERVE"
                    else apply_destroy(world, transform_name, config)
                )
                transformed_scores, _ = score_world(transformed, config)
                for metric in metric_names(config):
                    value = retention(
                        float(base_scores[metric]),
                        float(transformed_scores[metric]),
                        float(config["metric"]["retention_epsilon"]),
                    )
                    by_metric[metric][transform_set].append((transform_name, value))
                    transform_rows.append(
                        {
                            "stage": stage,
                            "seed": int(seed),
                            "target_class": target,
                            "metric": metric,
                            "transform_set": transform_set,
                            "transform": transform_name,
                            "base_score": float(base_scores[metric]),
                            "transformed_score": float(transformed_scores[metric]),
                            "retention": float(value),
                        }
                    )
        for metric in metric_names(config):
            preserve = by_metric[metric]["PRESERVE"]
            destroy = by_metric[metric]["DESTROY"]
            weakest = min(preserve, key=lambda item: item[1])
            strongest = max(destroy, key=lambda item: item[1])
            row = {
                "stage": stage,
                "seed": int(seed),
                "target_class": target,
                "metric": metric,
                "base_score": float(base_scores[metric]),
                "P": float(weakest[1]),
                "D": float(strongest[1]),
                "Delta": float(weakest[1] - strongest[1]),
                "weakest_preserve_transform": weakest[0],
                "strongest_destroy_transform": strongest[0],
            }
            if metric == CORRELON:
                row.update(
                    {
                        "T_iso": float(diagnostics["T_iso"]),
                        "T_floor": float(diagnostics["T_floor"]),
                        "T_pers": float(diagnostics["T_pers"]),
                    }
                )
            world_rows.append(row)
    return transform_rows, world_rows


def evaluate_negative_seed(seed: int, stage: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for negative_class in config["negative_classes"]:
        world = generate_world(negative_class, seed, config)
        if bool(world.meta["direct_edge"]):
            raise AssertionError(f"negative class contains a direct edge: {negative_class}")
        scores, diagnostics = score_world(world, config)
        for metric, value in scores.items():
            row = {
                "stage": stage,
                "seed": int(seed),
                "negative_class": negative_class,
                "metric": metric,
                "score": float(value),
                "direct_edge": False,
                "mechanism": str(world.meta["mechanism"]),
            }
            if metric == CORRELON:
                row.update(
                    {
                        "T_iso": float(diagnostics["T_iso"]),
                        "T_floor": float(diagnostics["T_floor"]),
                        "T_pers": float(diagnostics["T_pers"]),
                    }
                )
            rows.append(row)
    return rows


def _write_frames(
    transform_rows: list[dict[str, Any]],
    world_rows: list[dict[str, Any]],
    negative_rows: list[dict[str, Any]],
    config: dict[str, Any],
    prefix: str,
) -> None:
    pd.DataFrame(transform_rows).to_csv(_result_path(config, f"{prefix}_transform_results.csv"), index=False)
    pd.DataFrame(world_rows).to_csv(_result_path(config, f"{prefix}_world_results.csv"), index=False)
    pd.DataFrame(negative_rows).to_csv(_result_path(config, f"{prefix}_negative_scores.csv"), index=False)


def _thresholds_from_pilot(negative: pd.DataFrame, config: dict[str, Any]) -> dict[str, float]:
    calibration = negative[negative.negative_class == "independent_noise"]
    expected = int(config["seeds"]["pilot_count"])
    thresholds: dict[str, float] = {}
    for metric in metric_names(config):
        values = calibration.loc[calibration.metric == metric, "score"].to_numpy(dtype=float)
        if len(values) != expected:
            raise AssertionError(f"pilot threshold for {metric} expected {expected} scores, found {len(values)}")
        thresholds[metric] = float(max(np.quantile(values, 0.95, method="linear"), 1e-6))
    return thresholds


def run_pilot(config: dict[str, Any]) -> dict[str, Any]:
    ensure_output_dirs(config)
    transform_rows: list[dict[str, Any]] = []
    world_rows: list[dict[str, Any]] = []
    negative_rows: list[dict[str, Any]] = []
    start = int(config["seeds"]["pilot_start"])
    count = int(config["seeds"]["pilot_count"])
    for position, seed in enumerate(range(start, start + count), start=1):
        transforms, worlds = evaluate_target_seed(seed, "pilot", config)
        transform_rows.extend(transforms)
        world_rows.extend(worlds)
        negative_rows.extend(evaluate_negative_seed(seed, "pilot", config))
        print(f"pilot {position}/{count} complete", flush=True)
    _write_frames(transform_rows, world_rows, negative_rows, config, "correlon_zero_pilot")
    world_frame = pd.DataFrame(world_rows)
    negative_frame = pd.DataFrame(negative_rows)
    thresholds = _thresholds_from_pilot(negative_frame, config)
    pilot_summary = {
        "protocol": config["protocol"],
        "stage": "pilot",
        "seed_start": start,
        "seed_count": count,
        "positive_thresholds": thresholds,
        "calibration_class": "independent_noise",
        "quantile": 0.95,
        "quantile_method": "linear",
        "target_world_delta_means": {
            metric: float(group.Delta.mean()) for metric, group in world_frame.groupby("metric", sort=False)
        },
        "implementation_manifest": implementation_manifest(config),
        "confirmatory_data_opened": False,
    }
    write_json(_result_path(config, "correlon_zero_pilot_summary.json"), pilot_summary)
    return pilot_summary


def aggregate_seed_results(world: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (seed, metric), group in world.groupby(["seed", "metric"], sort=True):
        weakest_index = group.P.idxmin()
        strongest_index = group.D.idxmax()
        rows.append(
            {
                "seed": int(seed),
                "metric": metric,
                "P": float(group.P.min()),
                "D": float(group.D.max()),
                "Delta": float(group.P.min() - group.D.max()),
                "weakest_target_class": str(group.loc[weakest_index, "target_class"]),
                "weakest_preserve_transform": str(group.loc[weakest_index, "weakest_preserve_transform"]),
                "strongest_destroy_target_class": str(group.loc[strongest_index, "target_class"]),
                "strongest_destroy_transform": str(group.loc[strongest_index, "strongest_destroy_transform"]),
                "target_class_count": int(group.target_class.nunique()),
            }
        )
    result = pd.DataFrame(rows)
    baselines = set(config["baselines"])
    for seed, group in result.groupby("seed"):
        baseline_rows = group[group.metric.isin(baselines)]
        if len(baseline_rows) != len(baselines):
            raise AssertionError("per-seed baseline set is incomplete")
        strongest_index = baseline_rows.Delta.idxmax()
        strongest_metric = str(result.loc[strongest_index, "metric"])
        strongest_delta = float(result.loc[strongest_index, "Delta"])
        mask = (result.seed == seed) & (result.metric == CORRELON)
        result.loc[mask, "strongest_baseline_metric"] = strongest_metric
        result.loc[mask, "strongest_baseline_delta"] = strongest_delta
        result.loc[mask, "baseline_advantage"] = result.loc[mask, "Delta"] - strongest_delta
    return result


def _bootstrap_mean_ci(values: Iterable[float], seed: int, resamples: int) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    if len(array) < 2:
        raise ValueError("bootstrap CI requires at least two observations")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(array), size=(resamples, len(array)))
    means = np.mean(array[draws], axis=1)
    return (
        float(np.quantile(means, 0.025, method="linear")),
        float(np.quantile(means, 0.975, method="linear")),
    )


def _describe(values: np.ndarray, config: dict[str, Any], seed_offset: int) -> dict[str, Any]:
    low, high = _bootstrap_mean_ci(
        values,
        int(config["seeds"]["bootstrap"]) + seed_offset,
        int(config["statistics"]["bootstrap_resamples"]),
    )
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "standard_deviation": float(np.std(values, ddof=1)),
        "mean_CI95": [low, high],
        "q05": float(np.quantile(values, float(config["statistics"]["lower_tail_quantile"]), method="linear")),
        "failure_count_Delta_le_zero": int(np.sum(values <= 0.0)),
    }


def _fpr_records(negative: pd.DataFrame, config: dict[str, Any]) -> list[dict[str, Any]]:
    thresholds = config["thresholds"]["positive_thresholds"]
    if set(thresholds) != set(metric_names(config)):
        raise RuntimeError("all pilot-frozen positive thresholds are required")
    records: list[dict[str, Any]] = []
    for (metric, negative_class), group in negative.groupby(["metric", "negative_class"], sort=False):
        threshold = float(thresholds[metric])
        crossings = group.score.to_numpy(dtype=float) > threshold
        records.append(
            {
                "metric": metric,
                "negative_class": negative_class,
                "threshold": threshold,
                "n": int(len(group)),
                "crossing_count": int(np.sum(crossings)),
                "false_positive_rate": float(np.mean(crossings)),
                "mean_score": float(group.score.mean()),
                "median_score": float(group.score.median()),
                "maximum_score": float(group.score.max()),
            }
        )
    return records


def summarize_confirmatory(
    transform: pd.DataFrame,
    world: pd.DataFrame,
    negative: pd.DataFrame,
    seed_results: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    metric_summary: dict[str, Any] = {}
    for index, metric in enumerate(metric_names(config)):
        group = seed_results[seed_results.metric == metric]
        metric_summary[metric] = {
            "P": _describe(group.P.to_numpy(dtype=float), config, 100 + 3 * index),
            "D": _describe(group.D.to_numpy(dtype=float), config, 101 + 3 * index),
            "Delta": _describe(group.Delta.to_numpy(dtype=float), config, 102 + 3 * index),
            "weakest_preserve_transform": str(
                world.loc[world[world.metric == metric].P.idxmin(), "weakest_preserve_transform"]
            ),
            "strongest_destroy_transform": str(
                world.loc[world[world.metric == metric].D.idxmax(), "strongest_destroy_transform"]
            ),
        }
    correlon_rows = seed_results[seed_results.metric == CORRELON]
    advantage = correlon_rows.baseline_advantage.to_numpy(dtype=float)
    advantage_summary = _describe(advantage, config, 800)
    baseline_means = {
        metric: float(group.Delta.mean())
        for metric, group in seed_results[seed_results.metric.isin(config["baselines"])].groupby("metric")
    }
    strongest_baseline = max(baseline_means, key=baseline_means.get)
    fpr = _fpr_records(negative, config)
    fpr_frame = pd.DataFrame(fpr)
    representation = transform[
        (transform.metric == CORRELON)
        & (transform.transform.isin(["P1_node_permutation", "P4_orthogonal_basis_rotation"]))
    ].copy()
    representation["failed"] = representation.retention < float(config["thresholds"]["representation_retention"])
    representation_rates = {
        name: float(group.failed.mean()) for name, group in representation.groupby("transform")
    }
    max_representation_failure = max(representation_rates.values())
    correlon_fpr = fpr_frame[fpr_frame.metric == CORRELON]
    common_fpr = float(
        correlon_fpr.loc[correlon_fpr.negative_class == "common_driver", "false_positive_rate"].iloc[0]
    )
    matched_classes = {"matched_low_rank", "matched_spectrum", "matched_autocorrelation"}
    matched_max_fpr = float(
        correlon_fpr[correlon_fpr.negative_class.isin(matched_classes)].false_positive_rate.max()
    )
    summary = {
        "protocol": config["protocol"],
        "version": config["version"],
        "stage": "confirmatory_complete_adversary_pending",
        "seed_start": int(config["seeds"]["confirmatory_start"]),
        "seed_count": int(config["seeds"]["confirmatory_count"]),
        "target_classes": config["targets"],
        "negative_classes": config["negative_classes"],
        "positive_thresholds": config["thresholds"]["positive_thresholds"],
        "metric_summary": metric_summary,
        "baseline_comparison": {
            "strongest_baseline_by_mean_delta": strongest_baseline,
            "strongest_baseline_mean_delta": float(baseline_means[strongest_baseline]),
            "baseline_mean_deltas": baseline_means,
            "paired_advantage": advantage_summary,
        },
        "false_positive_audit": fpr,
        "representation_audit": {
            "failure_rates": representation_rates,
            "maximum_failure_rate": float(max_representation_failure),
            "retention_threshold": float(config["thresholds"]["representation_retention"]),
        },
        "pre_adversary_failure_flags": {
            "representation_dependence": bool(
                max_representation_failure
                > float(config["thresholds"]["maximum_representation_failure_rate"])
            ),
            "common_driver": bool(common_fpr > float(config["thresholds"]["maximum_false_positive_rate"])),
            "matched_null": bool(matched_max_fpr > float(config["thresholds"]["maximum_false_positive_rate"])),
            "baseline_equals_or_exceeds": bool(
                baseline_means[strongest_baseline]
                >= metric_summary[CORRELON]["Delta"]["mean"]
            ),
        },
        "implementation_manifest": implementation_manifest(config),
        "decision": {"status": "PENDING_ADVERSARIAL", "primary_label": None},
    }
    return summary


def run_confirmatory(config: dict[str, Any]) -> dict[str, Any]:
    ensure_output_dirs(config)
    if not config["thresholds"]["positive_thresholds"]:
        raise RuntimeError("pilot thresholds must be frozen in config before confirmation")
    transform_rows: list[dict[str, Any]] = []
    world_rows: list[dict[str, Any]] = []
    negative_rows: list[dict[str, Any]] = []
    start = int(config["seeds"]["confirmatory_start"])
    count = int(config["seeds"]["confirmatory_count"])
    for position, seed in enumerate(range(start, start + count), start=1):
        transforms, worlds = evaluate_target_seed(seed, "confirmatory", config)
        transform_rows.extend(transforms)
        world_rows.extend(worlds)
        negative_rows.extend(evaluate_negative_seed(seed, "confirmatory", config))
        if position % 5 == 0 or position == count:
            _write_frames(
                transform_rows,
                world_rows,
                negative_rows,
                config,
                "correlon_zero",
            )
            print(f"confirmatory {position}/{count} checkpoint saved", flush=True)
    transform = pd.DataFrame(transform_rows)
    world = pd.DataFrame(world_rows)
    negative = pd.DataFrame(negative_rows)
    seed_results = aggregate_seed_results(world, config)
    seed_results.to_csv(_result_path(config, "correlon_zero_seed_results.csv"), index=False)
    summary = summarize_confirmatory(transform, world, negative, seed_results, config)
    write_json(_result_path(config, "correlon_zero_summary.json"), summary)
    return summary


def _decision_from_summary(summary: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    flags = dict(summary["pre_adversary_failure_flags"])
    adversarial = summary["adversarial_audit"]
    flags["adversarial"] = bool(adversarial["formal_repeated_crossing"])
    secondary: list[str] = []
    if flags["representation_dependence"]:
        secondary.append("FALSIFIED_REPRESENTATION_DEPENDENCE")
    if flags["common_driver"]:
        secondary.append("FALSIFIED_COMMON_DRIVER")
    if flags["matched_null"]:
        secondary.append("FALSIFIED_MATCHED_NULL")
    if flags["baseline_equals_or_exceeds"]:
        secondary.append("FALSIFIED_BY_BASELINE")
    if flags["adversarial"]:
        secondary.append("FALSIFIED_ADVERSARIAL")

    if flags["representation_dependence"]:
        primary = "FALSIFIED_REPRESENTATION_DEPENDENCE"
    elif flags["common_driver"]:
        primary = "FALSIFIED_COMMON_DRIVER"
    elif flags["matched_null"]:
        primary = "FALSIFIED_MATCHED_NULL"
    elif flags["baseline_equals_or_exceeds"]:
        primary = "FALSIFIED_BY_BASELINE"
    elif flags["adversarial"]:
        primary = "FALSIFIED_ADVERSARIAL"
    else:
        correlon_delta = summary["metric_summary"][CORRELON]["Delta"]
        advantage = summary["baseline_comparison"]["paired_advantage"]
        fpr_frame = pd.DataFrame(summary["false_positive_audit"])
        correlon_fpr = fpr_frame[fpr_frame.metric == CORRELON]
        core_pass = bool(
            correlon_delta["mean"] >= float(config["thresholds"]["minimum_delta"])
            and correlon_delta["q05"] > 0.0
            and correlon_fpr.false_positive_rate.max()
            <= float(config["thresholds"]["maximum_false_positive_rate"])
        )
        unique_pass = bool(
            advantage["mean"] >= float(config["thresholds"]["minimum_baseline_advantage"])
            and advantage["mean_CI95"][0] > 0.0
        )
        if core_pass and unique_pass:
            primary = "SURVIVES_ZERO"
        elif core_pass and advantage["mean"] > 0.0:
            primary = "SURVIVES_WITHOUT_UNIQUENESS"
        else:
            primary = "INCONCLUSIVE_POWER"
    return {
        "status": "FINAL",
        "primary_label": primary,
        "all_failure_labels": secondary,
        "failure_flags": flags,
    }


def run_adversarial_and_finalize(config: dict[str, Any]) -> dict[str, Any]:
    summary_path = _result_path(config, "correlon_zero_summary.json")
    if not summary_path.exists():
        raise FileNotFoundError("confirmatory summary is required before adversarial search")
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    adversarial = run_adversarial_search(config)
    summary["stage"] = "confirmatory_and_adversarial_complete"
    summary["adversarial_audit"] = adversarial
    summary["decision"] = _decision_from_summary(summary, config)
    write_json(summary_path, summary)
    return summary


def _heatmap(
    frame: pd.DataFrame,
    transforms: list[str],
    metrics: list[str],
    title: str,
    subtitle: str,
    path: Path,
) -> None:
    pivot = frame.pivot(index="metric", columns="transform", values="retention").reindex(
        index=metrics, columns=transforms
    )
    fig, ax = plt.subplots(figsize=(12, 6.2))
    image = ax.imshow(pivot.to_numpy(), vmin=0.0, vmax=1.0, cmap="Blues", aspect="auto")
    ax.set_xticks(np.arange(len(transforms)), [name.split("_", 1)[0] for name in transforms])
    ax.set_yticks(np.arange(len(metrics)), metrics)
    ax.set_title(title, loc="left", color=INK, fontsize=14, fontweight="bold")
    ax.text(0.0, 1.01, subtitle, transform=ax.transAxes, color="#596579", fontsize=9, va="bottom")
    for row in range(len(metrics)):
        for column in range(len(transforms)):
            value = pivot.iloc[row, column]
            ax.text(column, row, f"{value:.2f}", ha="center", va="center", color="white" if value > 0.55 else INK, fontsize=8)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("mean directional retention")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _null_plane(config: dict[str, Any], count: int = 10) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    start = int(config["seeds"]["confirmatory_start"])
    for seed in range(start, start + count):
        for negative_class in config["negative_classes"]:
            world = generate_world(negative_class, seed, config)
            base, _ = score_world(world, config)
            preserve = []
            destroy = []
            for name in config["preserve_transformations"]:
                transformed, _ = score_world(apply_preserve(world, name, config), config)
                preserve.append(retention(base[CORRELON], transformed[CORRELON]))
            for name in config["destroy_transformations"]:
                transformed, _ = score_world(apply_destroy(world, name, config), config)
                destroy.append(retention(base[CORRELON], transformed[CORRELON]))
            rows.append(
                {
                    "seed": seed,
                    "negative_class": negative_class,
                    "P": float(min(preserve)),
                    "D": float(max(destroy)),
                    "Delta": float(min(preserve) - max(destroy)),
                    "descriptive_seed_subset": True,
                }
            )
        print(f"null-plane descriptive seed {seed - start + 1}/{count}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(_result_path(config, "correlon_zero_null_plane.csv"), index=False)
    return frame


def make_figures(config: dict[str, Any]) -> list[str]:
    ensure_output_dirs(config)
    transform = pd.read_csv(_result_path(config, "correlon_zero_transform_results.csv"))
    world = pd.read_csv(_result_path(config, "correlon_zero_world_results.csv"))
    negative = pd.read_csv(_result_path(config, "correlon_zero_negative_scores.csv"))
    seeds = pd.read_csv(_result_path(config, "correlon_zero_seed_results.csv"))
    adversary = pd.read_csv(_result_path(config, "correlon_zero_adversarial_cases.csv"))
    metrics = metric_names(config)
    mean_transform = transform.groupby(["metric", "transform"], as_index=False).retention.mean()
    preservation_path = _figure_path(config, "correlon_zero_preservation_by_transform.png")
    _heatmap(
        mean_transform,
        config["preserve_transformations"],
        metrics,
        "Preservation retention by transformation",
        "Confirmatory 200 seeds × 3 target classes; values are directional score retention",
        preservation_path,
    )
    destruction_path = _figure_path(config, "correlon_zero_destroy_residual_by_transform.png")
    _heatmap(
        mean_transform,
        config["destroy_transformations"],
        metrics,
        "Destroy-condition residual retention",
        "Higher values indicate that a metric survived a mechanism-destroying operation",
        destruction_path,
    )

    delta_path = _figure_path(config, "correlon_zero_delta_distribution.png")
    fig, ax = plt.subplots(figsize=(12, 5.8))
    arrays = [seeds.loc[seeds.metric == metric, "Delta"].to_numpy() for metric in metrics]
    boxes = ax.boxplot(arrays, labels=metrics, patch_artist=True, showfliers=False)
    for index, box in enumerate(boxes["boxes"]):
        box.set_facecolor(BLUE if index == 0 else "#dbe4f0")
        box.set_edgecolor(INK)
    ax.axhline(0.0, color=INK, linewidth=1.2)
    ax.axhline(float(config["thresholds"]["minimum_delta"]), color=ORANGE, linestyle="--", linewidth=1.2, label="minimum Delta=0.10")
    ax.set_ylabel("Delta = P - D")
    ax.set_title("Confirmatory preserve-destroy separation", loc="left", color=INK, fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(delta_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    summary_path = _result_path(config, "correlon_zero_summary.json")
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    strongest = summary["baseline_comparison"]["strongest_baseline_by_mean_delta"]
    pivot = seeds[seeds.metric.isin([CORRELON, strongest])].pivot(index="seed", columns="metric", values="Delta")
    paired_path = _figure_path(config, "correlon_zero_vs_strongest_baseline.png")
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    ax.scatter(pivot[strongest], pivot[CORRELON], s=20, alpha=0.65, color=BLUE, edgecolors="none")
    limits = [float(min(pivot.min().min(), -0.05)), float(max(pivot.max().max(), 0.15))]
    ax.plot(limits, limits, color=INK, linestyle="--", linewidth=1.1, label="equal separation")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel(f"{strongest} Delta")
    ax.set_ylabel("Correlon v1 Delta")
    ax.set_title("Paired-seed separation versus strongest baseline", loc="left", color=INK, fontsize=13, fontweight="bold")
    ax.grid(color=GRID, linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(paired_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    thresholds = config["thresholds"]["positive_thresholds"]
    corr_negative = negative[negative.metric == CORRELON].copy()
    fpr = (
        corr_negative.assign(crossing=corr_negative.score > float(thresholds[CORRELON]))
        .groupby("negative_class", as_index=False)
        .crossing.mean()
        .sort_values("crossing", ascending=True)
    )
    fpr_path = _figure_path(config, "correlon_zero_false_positive_rate.png")
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    colors = [ORANGE if value > float(config["thresholds"]["maximum_false_positive_rate"]) else BLUE for value in fpr.crossing]
    ax.barh(fpr.negative_class, fpr.crossing, color=colors)
    ax.axvline(float(config["thresholds"]["maximum_false_positive_rate"]), color=INK, linestyle="--", label="maximum FPR=0.05")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("false-positive rate")
    ax.set_title("Correlon v1 false positives by null class", loc="left", color=INK, fontsize=14, fontweight="bold")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fpr_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    trajectory_path = _figure_path(config, "correlon_zero_adversarial_trajectory.png")
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.plot(adversary.trial + 1, adversary.running_max, color=BLUE, linewidth=2.0, label="running maximum")
    ax.axhline(float(thresholds[CORRELON]), color=ORANGE, linestyle="--", linewidth=1.5, label="frozen positive threshold")
    ax.set_xlabel("random-search trial")
    ax.set_ylabel("Correlon v1 score")
    ax.set_title("Anti-Correlon search trajectory", loc="left", color=INK, fontsize=14, fontweight="bold")
    ax.grid(color=GRID, linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(trajectory_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    best_path = _figure_path(config, "correlon_zero_best_adversarial_example.png")
    archive = np.load(_result_path(config, "correlon_zero_best_adversarial.npz"), allow_pickle=True)
    x = archive["x"]
    y = archive["y"]
    x_pc = (x - x.mean(0)) @ np.linalg.svd(x - x.mean(0), full_matrices=False)[2][0]
    y_pc = (y - y.mean(0)) @ np.linalg.svd(y - y.mean(0), full_matrices=False)[2][0]
    lags = np.arange(-30, 31)
    correlations = []
    for lag in lags:
        if lag < 0:
            correlations.append(np.corrcoef(x_pc[-lag:], y_pc[:lag])[0, 1])
        elif lag > 0:
            correlations.append(np.corrcoef(x_pc[:-lag], y_pc[lag:])[0, 1])
        else:
            correlations.append(np.corrcoef(x_pc, y_pc)[0, 1])
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 6.8), gridspec_kw={"height_ratios": [1.5, 1.0]})
    extent = min(250, len(x_pc))
    axes[0].plot(np.arange(extent), x_pc[:extent], color=BLUE, label="X first PC")
    axes[0].plot(np.arange(extent), y_pc[:extent], color=ORANGE, alpha=0.8, label="Y first PC")
    axes[0].set_title("Best admissible adversarial null: leading observed components", loc="left", color=INK, fontweight="bold")
    axes[0].legend(frameon=False)
    axes[0].grid(color=GRID, linewidth=0.5)
    axes[1].plot(lags, correlations, color=OLIVE, marker="o", markersize=3)
    axes[1].axhline(0.0, color=INK, linewidth=0.8)
    axes[1].set_xlabel("lag (samples)")
    axes[1].set_ylabel("PC cross-correlation")
    axes[1].grid(color=GRID, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(best_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    null_plane = _null_plane(config, count=10)
    target_plane = world[world.metric == CORRELON]
    plane_path = _figure_path(config, "correlon_zero_preserve_destroy_plane.png")
    fig, ax = plt.subplots(figsize=(8.2, 7.0))
    for target, group in target_plane.groupby("target_class"):
        ax.scatter(group.P, group.D, s=18, alpha=0.28, label=f"target: {target}")
    centroids = null_plane.groupby("negative_class", as_index=False)[["P", "D"]].mean()
    ax.scatter(centroids.P, centroids.D, marker="x", s=70, linewidths=2.0, color=ORANGE, label="null class centroids (10-seed descriptive subset)")
    for row in centroids.itertuples(index=False):
        ax.annotate(str(row.negative_class), (row.P, row.D), xytext=(4, 3), textcoords="offset points", fontsize=7)
    x_grid = np.linspace(0.0, 1.0, 100)
    ax.plot(x_grid, x_grid - float(config["thresholds"]["minimum_delta"]), color=INK, linestyle="--", linewidth=1.0, label="Delta=0.10")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("P: worst preservation retention")
    ax.set_ylabel("D: worst destroy residual retention")
    ax.set_title("Preserve-Destroy plane for targets and nulls", loc="left", color=INK, fontsize=14, fontweight="bold")
    ax.grid(color=GRID, linewidth=0.5)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(plane_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    paths = [
        preservation_path,
        destruction_path,
        delta_path,
        paired_path,
        fpr_path,
        trajectory_path,
        best_path,
        plane_path,
    ]
    chart_map = pd.DataFrame(
        [
            {"figure": path.name, "family": "heatmap" if "transform" in path.name else "static", "source": "confirmatory CSV", "qa": "pending visual inspection"}
            for path in paths
        ]
    )
    chart_map.to_csv(_result_path(config, "correlon_zero_chart_map.csv"), index=False)
    return [str(path.relative_to(ROOT)) for path in paths]


def _fmt(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def _next_experiment(primary_label: str) -> str:
    if primary_label == "FALSIFIED_COMMON_DRIVER":
        return (
            "Run a new, separately preregistered mechanism-blindness closure test with direct-coupling and latent-common-driver "
            "systems matched on covariance, spectrum, autocorrelation, rank, and v1 score. Hold out entire generator families and "
            "ask whether any candidate observable can separate the mechanisms without interventions. Do not revise v1 on the present seeds."
        )
    if primary_label == "FALSIFIED_REPRESENTATION_DEPENDENCE":
        return (
            "Isolate the single failing invertible representation transform, derive the exact equivariance condition, and test a new "
            "operator on fresh generators and fresh seeds. The present v1 remains rejected."
        )
    if primary_label == "FALSIFIED_MATCHED_NULL":
        return (
            "Use the winning matched-null statistic as an explicit conditioning variable in a new held-out generator-family test. "
            "Any v2 score must add separation after exact matching and must use fresh seeds."
        )
    if primary_label == "FALSIFIED_BY_BASELINE":
        return (
            "Treat the strongest baseline as the null explanatory model and test only a preregistered residual statistic on fresh "
            "world families. Do not relabel baseline performance as Correlon evidence."
        )
    if primary_label == "FALSIFIED_ADVERSARIAL":
        return (
            "Freeze the discovered null parameter region, generate a held-out adversarial family from disjoint RNG seeds, and test "
            "whether any separately preregistered v2 rejects those nulls without losing target preservation."
        )
    return (
        "Replicate the complete frozen battery on disjoint generator families and seed blocks before changing the operational definition."
    )


def write_reports(config: dict[str, Any], figure_paths: list[str]) -> list[str]:
    summary_path = _result_path(config, "correlon_zero_summary.json")
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary["decision"]["status"] != "FINAL":
        raise RuntimeError("final decision is required before report generation")
    transform = pd.read_csv(_result_path(config, "correlon_zero_transform_results.csv"))
    world = pd.read_csv(_result_path(config, "correlon_zero_world_results.csv"))
    negative = pd.read_csv(_result_path(config, "correlon_zero_negative_scores.csv"))
    adversary = pd.read_csv(_result_path(config, "correlon_zero_adversarial_cases.csv"))
    primary = summary["decision"]["primary_label"]
    corr_stats = summary["metric_summary"][CORRELON]
    baseline = summary["baseline_comparison"]
    advantage = baseline["paired_advantage"]
    fpr_frame = pd.DataFrame(summary["false_positive_audit"])
    corr_fpr = fpr_frame[fpr_frame.metric == CORRELON].sort_values("false_positive_rate", ascending=False)
    worst_null = corr_fpr.iloc[0]
    weakest_preserve = (
        transform[(transform.metric == CORRELON) & (transform.transform_set == "PRESERVE")]
        .groupby("transform")
        .retention.mean()
        .sort_values()
    )
    strongest_destroy = (
        transform[(transform.metric == CORRELON) & (transform.transform_set == "DESTROY")]
        .groupby("transform")
        .retention.mean()
        .sort_values(ascending=False)
    )
    metric_rows = []
    for metric, record in summary["metric_summary"].items():
        delta = record["Delta"]
        metric_rows.append(
            [
                metric,
                _fmt(record["P"]["mean"]),
                _fmt(record["D"]["mean"]),
                _fmt(delta["mean"]),
                f"[{_fmt(delta['mean_CI95'][0])}, {_fmt(delta['mean_CI95'][1])}]",
                _fmt(delta["q05"]),
                delta["failure_count_Delta_le_zero"],
            ]
        )
    preserve_rows = [
        [name, _fmt(value), "PRESERVE (frozen)"] for name, value in weakest_preserve.items()
    ]
    destroy_rows = [
        [name, _fmt(value), "DESTROY (frozen)"] for name, value in strongest_destroy.items()
    ]
    fpr_rows = [
        [
            row.negative_class,
            _fmt(row.false_positive_rate, 3),
            int(row.crossing_count),
            int(row.n),
            _fmt(row.maximum_score),
        ]
        for row in corr_fpr.itertuples(index=False)
    ]
    baseline_rows = [
        [metric, _fmt(value), _fmt(corr_stats["Delta"]["mean"] - value)]
        for metric, value in sorted(baseline["baseline_mean_deltas"].items(), key=lambda item: item[1], reverse=True)
    ]
    adversarial_summary = summary["adversarial_audit"]
    next_experiment = _next_experiment(primary)
    result_markdown = f"""# Correlon Zero — Adversarial Invariance Falsification Results

## 1. Executive verdict

**{primary}**

The frozen v1 observable was tested without post-confirmatory repair. Its mean preserve-destroy separation was `{_fmt(corr_stats['Delta']['mean'])}` with 95% bootstrap CI `[{_fmt(corr_stats['Delta']['mean_CI95'][0])}, {_fmt(corr_stats['Delta']['mean_CI95'][1])}]`; the 5th percentile was `{_fmt(corr_stats['Delta']['q05'])}`. The strongest named false-positive class was `{worst_null.negative_class}` with FPR `{_fmt(worst_null.false_positive_rate, 3)}` against the frozen positive threshold.

This verdict concerns only the frozen operational definition in the tested synthetic domain. It makes no ontological or direct-causal claim.

## 2. Frozen hypothesis

The candidate had to remain stable under P1-P7, disappear under D1-D8, keep its lower-tail `Delta=P-D` positive, hold every negative-class FPR at or below `0.05`, and exceed the per-seed strongest baseline by `0.10`.

## 3. Operational Correlon definition

The frozen operator was the Phase-4 whitened cross-covariance SVD. The only primary scalar was:

```text
CorrelonZero_v1 = sqrt(clip(T_iso,0,1) * clip(T_floor,0,1))
```

`T_iso` is the dominant singular-gap excess over six circular-shift nulls. `T_floor` is the 10th-percentile adjacent-mode continuity excess over the same null family. The source specification is commit `8544101`; the final implementation hash is `{summary['implementation_manifest']['combined_sha256']}`.

## 4. Preserve transformation results

The weakest average preservation condition was `{weakest_preserve.index[0]}` with mean retention `{_fmt(weakest_preserve.iloc[0])}`. P1 and P4 representation-failure rates were `{summary['representation_audit']['failure_rates']}`.

{_markdown_table(['transformation', 'mean retention', 'frozen set'], preserve_rows)}

![Preservation by transformation](figures/correlon_zero_preservation_by_transform.png)

## 5. Destroy transformation results

The largest residual retention was under `{strongest_destroy.index[0]}` at `{_fmt(strongest_destroy.iloc[0])}`. High D means the score remained after the preregistered mechanism-destroying operation.

{_markdown_table(['transformation', 'mean residual retention', 'frozen set'], destroy_rows)}

![Destroy residual](figures/correlon_zero_destroy_residual_by_transform.png)

## 6. Conventional baseline comparison

The strongest baseline by mean Delta was `{baseline['strongest_baseline_by_mean_delta']}` at `{_fmt(baseline['strongest_baseline_mean_delta'])}`. The conservative paired advantage of Correlon over the per-seed maximum baseline was `{_fmt(advantage['mean'])}` with CI `[{_fmt(advantage['mean_CI95'][0])}, {_fmt(advantage['mean_CI95'][1])}]`.

{_markdown_table(['metric', 'mean Delta', 'Correlon minus baseline'], baseline_rows)}

![Paired baseline comparison](figures/correlon_zero_vs_strongest_baseline.png)

The whitened cross-covariance construction remains CCA-equivalent for the audited operator; CCA identity is therefore an interpretation constraint, not an omitted novelty claim.

## 7. Common-driver falsification

The common-driver false-positive rate was `{_fmt(float(corr_fpr.loc[corr_fpr.negative_class == 'common_driver', 'false_positive_rate'].iloc[0]), 3)}`. The common-driver class contains no direct X/Y edge by construction. Crossing the positive boundary therefore cannot be interpreted as evidence of direct causation.

## 8. Matched-null falsification

{_markdown_table(['negative class', 'FPR', 'crossings', 'n', 'maximum score'], fpr_rows)}

![False-positive audit](figures/correlon_zero_false_positive_rate.png)

## 9. Anti-Correlon adversarial search

The deterministic 300-trial search found maximum null score `{_fmt(adversarial_summary['maximum_score'])}`. `{adversarial_summary['crossing_count']}` candidates crossed the frozen threshold `{_fmt(adversarial_summary['threshold'])}`, for crossing rate `{_fmt(adversarial_summary['crossing_rate'], 3)}`. Every candidate had `direct_XY_edge=false` by construction.

![Adversarial trajectory](figures/correlon_zero_adversarial_trajectory.png)

![Best adversarial example](figures/correlon_zero_best_adversarial_example.png)

## 10. Lower-tail and seed robustness

{_markdown_table(['metric', 'mean P', 'mean D', 'mean Delta', 'Delta CI95', 'Delta q05', 'Delta<=0'], metric_rows)}

![Delta distribution](figures/correlon_zero_delta_distribution.png)

The confirmatory unit was the seed after worst-case aggregation across all three target classes. No favorable seed selection or post-hoc transform removal was used.

## 11. Supported claims

- The report identifies how the frozen score behaved under every named transformation and null class.
- Any P1/P4 retention result supports only the corresponding tested representation invariance.
- Any positive target score is descriptive; it does not imply direct causation.

## 12. Falsified claims

Primary label: **{primary}**.

All recorded failure labels: `{summary['decision']['all_failure_labels']}`.

The experiment does not support interpreting strong correlation, persistence, low rank, common-driver dependence, or predictive dependence as uniquely Correlon-specific merely because v1 is positive.

## 13. Remaining ambiguity

The experiment is synthetic and bounded to the frozen generator families, sample sizes, transformations, and scalar readout. A failure identifies an operational non-uniqueness or insensitivity; it does not prove that no future relational invariant can exist. A pass on any secondary condition cannot rescue a failed primary condition.

## 14. Final decision

**{primary}** under the preregistered precedence. The Preserve-Destroy plane below shows target worlds and a clearly marked 10-seed descriptive null subset.

![Preserve-Destroy plane](figures/correlon_zero_preserve_destroy_plane.png)

## 15. Exact next experiment

{next_experiment}
"""
    (ROOT / "CORRELON_ZERO_RESULTS.md").write_text(result_markdown, encoding="utf-8", newline="\n")

    top_cases = adversary.sort_values("correlon_score", ascending=False).head(10)
    top_rows = [
        [
            int(row.trial),
            _fmt(row.correlon_score),
            bool(row.crosses_threshold),
            int(row.latent_rank),
            _fmt(row.rho, 3),
            _fmt(row.strength, 3),
            _fmt(row.noise, 3),
            int(row.sample_length),
        ]
        for row in top_cases.itertuples(index=False)
    ]
    adversarial_markdown = f"""# Correlon Zero — Anti-Correlon Adversarial Audit

## Frozen objective

Maximize the frozen `CorrelonZero_v1` score subject to `direct_XY_edge=false`. The search used seed `424242`, 300 random-search trials, and no metric or threshold update.

## Result

- Frozen threshold: `{_fmt(adversarial_summary['threshold'])}`
- Maximum null score: `{_fmt(adversarial_summary['maximum_score'])}`
- Crossing count: `{adversarial_summary['crossing_count']} / {adversarial_summary['trials']}`
- Repeated-crossing gate: `{'FAIL' if adversarial_summary['formal_repeated_crossing'] else 'PASS'}` for v1
- Best parameters: `{adversarial_summary['best_parameters']}`

{_markdown_table(['trial', 'score', 'crosses', 'rank', 'rho', 'strength', 'noise', 'T'], top_rows)}

## Mechanism firewall

Candidates were generated only from exogenous latent factors, optional oscillatory persistence, independent observation noise, deterministic drift, and observation delays. Neither X nor Y was used to update the other. The best-case archive is `results/correlon_zero_best_adversarial.npz`.

## Interpretation

An adversarial crossing is a false positive for the frozen operational criterion, not evidence that the null secretly contains the target edge. The result is permanent for v1 and cannot be repaired on these seeds.
"""
    (ROOT / "CORRELON_ZERO_ADVERSARIAL_AUDIT.md").write_text(
        adversarial_markdown, encoding="utf-8", newline="\n"
    )

    decision_rows = [
        ["Delta mean >= 0.10", _fmt(corr_stats["Delta"]["mean"]), corr_stats["Delta"]["mean"] >= 0.10],
        ["Delta q05 > 0", _fmt(corr_stats["Delta"]["q05"]), corr_stats["Delta"]["q05"] > 0.0],
        ["Every negative FPR <= 0.05", _fmt(corr_fpr.false_positive_rate.max(), 3), corr_fpr.false_positive_rate.max() <= 0.05],
        ["Baseline advantage >= 0.10", _fmt(advantage["mean"]), advantage["mean"] >= 0.10],
        ["P1/P4 failure rate <= 0.05", _fmt(summary["representation_audit"]["maximum_failure_rate"], 3), summary["representation_audit"]["maximum_failure_rate"] <= 0.05],
        ["Adversarial crossings < 3", adversarial_summary["crossing_count"], not adversarial_summary["formal_repeated_crossing"]],
    ]
    final_markdown = f"""# Correlon Zero — Final Decision

## Decision

**{primary}**

{_markdown_table(['criterion', 'observed', 'pass'], decision_rows)}

## Failure matrix

- Primary label: `{primary}`
- All failure labels: `{summary['decision']['all_failure_labels']}`
- Strongest baseline: `{baseline['strongest_baseline_by_mean_delta']}`
- Strongest named null by FPR: `{worst_null.negative_class}` (`{_fmt(worst_null.false_positive_rate, 3)}`)
- Maximum adversarial null score: `{_fmt(adversarial_summary['maximum_score'])}`

## Claim boundary

The decision falsifies or supports only `CorrelonZero_v1` as frozen in commit `8544101`. It does not establish an ontological Correlon, direct causation, or a physical discovery.

## Exact next experiment

{next_experiment}
"""
    (ROOT / "CORRELON_ZERO_FINAL_DECISION.md").write_text(final_markdown, encoding="utf-8", newline="\n")

    cemetery_map = {
        "FALSIFIED_COMMON_DRIVER": "correlon_zero_v1_common_driver_failure",
        "FALSIFIED_REPRESENTATION_DEPENDENCE": "correlon_zero_v1_basis_dependence",
        "FALSIFIED_MATCHED_NULL": "correlon_zero_v1_matched_null_failure",
        "FALSIFIED_BY_BASELINE": "correlon_zero_v1_baseline_failure",
        "FALSIFIED_ADVERSARIAL": "correlon_zero_v1_adversarial_false_positive",
    }
    cemetery_root = ROOT / config["outputs"]["cemetery_dir"]
    cemetery_root.mkdir(parents=True, exist_ok=True)
    if primary in cemetery_map:
        entry = cemetery_root / cemetery_map[primary]
        entry.mkdir(parents=True, exist_ok=True)
        metadata = f"""# Correlon Zero v1 Cemetery Entry

- metric definition: `sqrt(clip(T_iso,0,1) * clip(T_floor,0,1))`
- preregistration commit: `8544101`
- final implementation hash: `{summary['implementation_manifest']['combined_sha256']}`
- falsifying experiment: `CORRELON_ZERO_ADVERSARIAL_INVARIANCE_FALSIFICATION_v1.0`
- seed policy: pilot `0..19`; confirmatory `10000..10199`; adversary `424242`
- failure criterion: `{primary}` under the frozen precedence
- result summary: mean Delta `{_fmt(corr_stats['Delta']['mean'])}`; worst named-null FPR `{_fmt(worst_null.false_positive_rate, 3)}`; adversarial crossings `{adversarial_summary['crossing_count']}`
- reason for rejection: the frozen operational definition failed at least one preregistered necessary condition; no v1 repair was attempted

See `CORRELON_ZERO_RESULTS.md`, `CORRELON_ZERO_ADVERSARIAL_AUDIT.md`, and `CORRELON_ZERO_FINAL_DECISION.md`.
"""
        (entry / "README.md").write_text(metadata, encoding="utf-8", newline="\n")
    else:
        (cemetery_root / "README.md").write_text(
            "# Correlon Cemetery\n\nNo v1 cemetery entry was required by the final primary label.\n",
            encoding="utf-8",
            newline="\n",
        )
    return [
        "CORRELON_ZERO_RESULTS.md",
        "CORRELON_ZERO_ADVERSARIAL_AUDIT.md",
        "CORRELON_ZERO_FINAL_DECISION.md",
        *figure_paths,
    ]
