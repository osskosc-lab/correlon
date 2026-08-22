"""Independent recomputation and artifact QA for Correlon Zero results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import numpy as np
import pandas as pd

from .analysis import CORRELON, metric_names
from .utils import ROOT, implementation_manifest, write_json


def _path(config: dict[str, Any], name: str) -> Path:
    return ROOT / config["outputs"]["results_dir"] / name


def _independent_bootstrap(values: np.ndarray, resamples: int = 5000) -> list[float]:
    rng = np.random.default_rng(881122)
    means = np.empty(resamples, dtype=float)
    for index in range(resamples):
        means[index] = float(np.mean(rng.choice(values, size=len(values), replace=True)))
    return [
        float(np.quantile(means, 0.025, method="linear")),
        float(np.quantile(means, 0.975, method="linear")),
    ]


def run_independent_validation(config: dict[str, Any], figure_paths: list[str]) -> dict[str, Any]:
    transform = pd.read_csv(_path(config, "correlon_zero_transform_results.csv"))
    world = pd.read_csv(_path(config, "correlon_zero_world_results.csv"))
    negative = pd.read_csv(_path(config, "correlon_zero_negative_scores.csv"))
    seed_results = pd.read_csv(_path(config, "correlon_zero_seed_results.csv"))
    adversary = pd.read_csv(_path(config, "correlon_zero_adversarial_cases.csv"))
    with _path(config, "correlon_zero_summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    seed_count = int(config["seeds"]["confirmatory_count"])
    target_count = len(config["targets"])
    negative_count = len(config["negative_classes"])
    metric_count = len(metric_names(config))
    transform_count = len(config["preserve_transformations"]) + len(config["destroy_transformations"])
    expected = {
        "transform_rows": seed_count * target_count * metric_count * transform_count,
        "world_rows": seed_count * target_count * metric_count,
        "negative_rows": seed_count * negative_count * metric_count,
        "seed_rows": seed_count * metric_count,
        "adversary_rows": int(config["adversary"]["trials"]),
    }
    observed = {
        "transform_rows": len(transform),
        "world_rows": len(world),
        "negative_rows": len(negative),
        "seed_rows": len(seed_results),
        "adversary_rows": len(adversary),
    }
    row_counts_match = expected == observed

    independent_rows: list[dict[str, Any]] = []
    for (seed, metric), group in world.groupby(["seed", "metric"], sort=True):
        independent_rows.append(
            {
                "seed": int(seed),
                "metric": metric,
                "P_check": float(group.P.min()),
                "D_check": float(group.D.max()),
                "Delta_check": float(group.P.min() - group.D.max()),
            }
        )
    independent = pd.DataFrame(independent_rows)
    paired = seed_results.merge(independent, on=["seed", "metric"], how="outer", validate="one_to_one")
    seed_aggregate_max_error = float(
        np.nanmax(
            np.abs(
                paired[["P", "D", "Delta"]].to_numpy(dtype=float)
                - paired[["P_check", "D_check", "Delta_check"]].to_numpy(dtype=float)
            )
        )
    )
    seed_aggregates_match = bool(seed_aggregate_max_error <= 1e-12)

    thresholds = config["thresholds"]["positive_thresholds"]
    fpr_checks: list[dict[str, Any]] = []
    summary_fpr = pd.DataFrame(summary["false_positive_audit"])
    for (metric, negative_class), group in negative.groupby(["metric", "negative_class"], sort=False):
        fpr = float(np.mean(group.score.to_numpy(dtype=float) > float(thresholds[metric])))
        recorded = float(
            summary_fpr.loc[
                (summary_fpr.metric == metric) & (summary_fpr.negative_class == negative_class),
                "false_positive_rate",
            ].iloc[0]
        )
        fpr_checks.append(
            {
                "metric": metric,
                "negative_class": negative_class,
                "recomputed": fpr,
                "recorded": recorded,
                "match": bool(abs(fpr - recorded) <= 1e-15),
            }
        )
    fpr_match = bool(all(row["match"] for row in fpr_checks))

    corr = seed_results[seed_results.metric == CORRELON].sort_values("seed")
    delta = corr.Delta.to_numpy(dtype=float)
    delta_point = {
        "mean": float(np.mean(delta)),
        "median": float(np.median(delta)),
        "standard_deviation": float(np.std(delta, ddof=1)),
        "q05": float(np.quantile(delta, 0.05, method="linear")),
        "independent_mean_CI95": _independent_bootstrap(delta),
    }
    recorded_delta = summary["metric_summary"][CORRELON]["Delta"]
    delta_point_match = bool(
        abs(delta_point["mean"] - float(recorded_delta["mean"])) <= 1e-15
        and abs(delta_point["q05"] - float(recorded_delta["q05"])) <= 1e-15
    )

    baseline_rows = seed_results[seed_results.metric.isin(config["baselines"])]
    baseline_means = baseline_rows.groupby("metric").Delta.mean()
    strongest_baseline = str(baseline_means.idxmax())
    baseline_selection_match = bool(
        strongest_baseline == summary["baseline_comparison"]["strongest_baseline_by_mean_delta"]
    )
    pivot = seed_results.pivot(index="seed", columns="metric", values="Delta")
    per_seed_max = pivot[config["baselines"]].max(axis=1)
    independent_advantage = pivot[CORRELON] - per_seed_max
    advantage_mean = float(independent_advantage.mean())
    advantage_match = bool(
        abs(advantage_mean - float(summary["baseline_comparison"]["paired_advantage"]["mean"])) <= 1e-15
    )

    adversary_max = float(adversary.correlon_score.max())
    adversary_crossings = int(adversary.crosses_threshold.astype(bool).sum())
    adversary_match = bool(
        abs(adversary_max - float(summary["adversarial_audit"]["maximum_score"])) <= 1e-15
        and adversary_crossings == int(summary["adversarial_audit"]["crossing_count"])
        and not adversary.direct_edge.astype(bool).any()
    )

    figure_checks: list[dict[str, Any]] = []
    for relative in figure_paths:
        path = ROOT / relative
        pixels = mpimg.imread(path)
        figure_checks.append(
            {
                "path": relative,
                "exists": path.exists(),
                "bytes": int(path.stat().st_size),
                "height": int(pixels.shape[0]),
                "width": int(pixels.shape[1]),
                "finite": bool(np.all(np.isfinite(pixels))),
                "qa_pass": bool(path.stat().st_size > 10_000 and min(pixels.shape[0], pixels.shape[1]) >= 500),
            }
        )
    figure_qa_pass = bool(all(item["qa_pass"] and item["finite"] for item in figure_checks))

    manifest = implementation_manifest(config)
    manifest_match = bool(manifest["combined_sha256"] == summary["implementation_manifest"]["combined_sha256"])
    checks = {
        "row_counts_match": row_counts_match,
        "seed_aggregates_match": seed_aggregates_match,
        "false_positive_rates_match": fpr_match,
        "primary_delta_point_estimates_match": delta_point_match,
        "strongest_baseline_selection_match": baseline_selection_match,
        "paired_advantage_match": advantage_match,
        "adversary_recomputation_match": adversary_match,
        "implementation_manifest_match": manifest_match,
        "figure_qa_pass": figure_qa_pass,
        "all_values_finite": bool(
            np.all(np.isfinite(transform[["base_score", "transformed_score", "retention"]].to_numpy(dtype=float)))
            and np.all(np.isfinite(world[["base_score", "P", "D", "Delta"]].to_numpy(dtype=float)))
            and np.all(np.isfinite(negative.score.to_numpy(dtype=float)))
        ),
    }
    payload = {
        "validator": "independent_correlon_zero_validator_v1.0",
        "checks": checks,
        "all_checks_pass": bool(all(checks.values())),
        "expected_row_counts": expected,
        "observed_row_counts": observed,
        "seed_aggregate_max_error": seed_aggregate_max_error,
        "primary_delta": delta_point,
        "strongest_baseline": strongest_baseline,
        "paired_advantage_mean": advantage_mean,
        "adversary_maximum": adversary_max,
        "adversary_crossings": adversary_crossings,
        "fpr_checks": fpr_checks,
        "figure_checks": figure_checks,
        "implementation_manifest": manifest,
    }
    write_json(_path(config, "correlon_zero_independent_validation.json"), payload)
    if not payload["all_checks_pass"]:
        raise AssertionError(f"independent validation failed: {checks}")
    return payload

