"""T3: Correlon relation-to-state bridge with matched low-dimensional baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from timeless_common import (
    FIGURES,
    RESULTS,
    canonical_J,
    eig_floor,
    ensure_output_dirs,
    git_context,
    heldout_field_metrics,
    inv_spd,
    invsqrt_spd,
    json_dump,
    json_load,
    normalized_frobenius,
    random_orthogonal,
    scale_aligned_error,
    sha256_json,
    sqrt_spd,
)


DIMS = [4, 8, 16]
TRUE_RANKS = [1, 2, 4]
STRENGTHS = [0.1, 0.25, 0.5, 1.0]
RANK_CANDIDATES = [1, 2, 4]
TRAIN_SAMPLES = 2000
HELDOUT_SAMPLES = 1000
SCREENING_HELDOUT_SAMPLES = 64
EIGEN_FLOOR = 1e-8
IDENTITY_TOLERANCE = 1e-10
LOW_DIM_METHODS = ["block_diagonal", "pca_rank_matched", "random_rank_matched", "raw_svd"]
ALL_METHODS = ["correlon", "cca_equivalent", *LOW_DIM_METHODS, "full_state_oracle"]
STAGE_SEEDS = {
    "development": list(range(0, 50)),
    "validation": list(range(1000, 1050)),
    "confirmation": list(range(10000, 10200)),
}


def make_case(d: int, true_rank: int, strength: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7100000 + seed + d * 10000 + true_rank * 1000 + int(strength * 100))
    qx = random_orthogonal(d, rng)
    qy = random_orthogonal(d, rng)
    kxx = (qx * np.linspace(1.2, 2.4, d)) @ qx.T
    kyy = (qy * np.linspace(1.3, 2.5, d)) @ qy.T
    u = random_orthogonal(d, rng)[:, :true_rank]
    v = random_orthogonal(d, rng)[:, :true_rank]
    singular = 0.45 * strength * np.linspace(1.0, 0.75, true_rank)
    kxy = (u * singular) @ v.T
    k = np.block([[kxx, kxy], [kxy.T, kyy]])
    if np.min(np.linalg.eigvalsh(k)) <= 0:
        raise AssertionError("T3 precision matrix is not SPD")
    covariance = inv_spd(k, EIGEN_FLOOR)
    return k, covariance


def gaussian_samples(covariance: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    chol = np.linalg.cholesky(eig_floor(covariance, EIGEN_FLOOR))
    return rng.normal(size=(n, covariance.shape[0])) @ chol.T


def assemble_covariance(cxx: np.ndarray, cxy: np.ndarray, cyy: np.ndarray) -> np.ndarray:
    out = np.block([[cxx, cxy], [cxy.T, cyy]])
    return eig_floor(out, EIGEN_FLOOR)


def pca_rank_matched(covariance: np.ndarray, rank: int) -> np.ndarray:
    w, v = np.linalg.eigh(eig_floor(covariance, EIGEN_FLOOR))
    order = np.argsort(w)[::-1]
    w = w[order]
    v = v[:, order]
    k = min(2 * rank, covariance.shape[0])
    residual = float(np.mean(w[k:])) if k < len(w) else float(EIGEN_FLOOR)
    return eig_floor((v[:, :k] * w[:k]) @ v[:, :k].T + residual * np.eye(covariance.shape[0]), EIGEN_FLOOR)


def cross_reconstructions(
    covariance_hat: np.ndarray,
    d: int,
    rank: int,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], float]:
    cxx = eig_floor(covariance_hat[:d, :d], EIGEN_FLOOR)
    cyy = eig_floor(covariance_hat[d:, d:], EIGEN_FLOOR)
    cxy = covariance_hat[:d, d:]
    sx = sqrt_spd(cxx, EIGEN_FLOOR)
    sy = sqrt_spd(cyy, EIGEN_FLOOR)
    q = invsqrt_spd(cxx, EIGEN_FLOOR) @ cxy @ invsqrt_spd(cyy, EIGEN_FLOOR)
    u, s, vt = np.linalg.svd(q, full_matrices=False)
    k = min(rank, d)
    q_rank = (u[:, :k] * s[:k]) @ vt[:k, :]
    cxy_correlon = sx @ q_rank @ sy

    raw_u, raw_s, raw_vt = np.linalg.svd(cxy, full_matrices=False)
    cxy_raw = (raw_u[:, :k] * raw_s[:k]) @ raw_vt[:k, :]

    random_u = random_orthogonal(d, rng)[:, :k]
    random_v = random_orthogonal(d, rng)[:, :k]
    q_random = (random_u * s[:k]) @ random_v.T
    cxy_random = sx @ q_random @ sy

    methods = {
        "correlon": assemble_covariance(cxx, cxy_correlon, cyy),
        "cca_equivalent": assemble_covariance(cxx, cxy_correlon.copy(), cyy),
        "block_diagonal": assemble_covariance(cxx, np.zeros_like(cxy), cyy),
        "pca_rank_matched": pca_rank_matched(covariance_hat, rank),
        "random_rank_matched": assemble_covariance(cxx, cxy_random, cyy),
        "raw_svd": assemble_covariance(cxx, cxy_raw, cyy),
    }
    identity_diff = normalized_frobenius(methods["correlon"], methods["cca_equivalent"])
    return methods, identity_diff


def frequency_error(a_hat: np.ndarray, a_true: np.ndarray, scale: float) -> float:
    f_hat = np.sort(np.abs(np.imag(np.linalg.eigvals(scale * a_hat))))
    f_true = np.sort(np.abs(np.imag(np.linalg.eigvals(a_true))))
    return float(np.mean(np.abs(f_hat - f_true) / np.maximum(f_true, 1e-12)))


def evaluate_method(
    covariance_method: np.ndarray,
    covariance_true: np.ndarray,
    j: np.ndarray,
    heldout: np.ndarray,
    field_n: int,
) -> dict:
    k_method = inv_spd(covariance_method, EIGEN_FLOOR)
    k_true = inv_spd(covariance_true, EIGEN_FLOOR)
    a_method = j @ k_method
    a_true = j @ k_true
    scale, error = scale_aligned_error(a_method, a_true)
    field = heldout_field_metrics(scale * a_method, a_true, heldout[:field_n])
    return {
        "scale": scale,
        "generator_error": error,
        "vector_field_cosine": field["vector_field_cosine"],
        "orbit_shape_error": field["orbit_shape_error"],
        "frequency_error": frequency_error(a_method, a_true, scale),
        "stability_max_real_eigenvalue": float(np.max(np.abs(np.real(np.linalg.eigvals(a_method))))),
    }


def selection_from_validation(validation: pd.DataFrame) -> dict:
    corr = validation[validation.method == "correlon"].groupby("rank").generator_error.mean()
    corr_rank = int(corr.idxmin())
    baseline = validation[validation.method.isin(LOW_DIM_METHODS)].groupby(["method", "rank"]).generator_error.mean()
    base_method, base_rank = baseline.idxmin()
    identity_max = float(validation.cca_identity_diff.max())
    return {
        "correlon_rank": corr_rank,
        "best_baseline_method": str(base_method),
        "best_baseline_rank": int(base_rank),
        "validation_correlon_error": float(corr.loc[corr_rank]),
        "validation_best_baseline_error": float(baseline.loc[(base_method, base_rank)]),
        "cca_identity_max_diff": identity_max,
        "identity_tolerance": IDENTITY_TOLERANCE,
        "selection_rule": "minimum mean primary generator error on validation; full oracle and CCA identity row excluded from baseline selection",
    }


def freeze_config(selection: dict) -> dict:
    config = {
        "program": "Timeless Correlon",
        "version": "1.0",
        "t0": {"gamma": [0.25, 0.5, 1.0], "omega": [0.0, 0.25, 0.5, 1.0, 2.0], "trajectory_length": 5000, "burn_in": 1000, "dt": 0.01, "seeds": [0, 99]},
        "t1": {"dimensions": [2, 4, 8, 16], "condition_numbers": [1.5, 3.0, 10.0, 30.0], "betas": [0.5, 1.0, 2.0, 4.0], "sample_sizes": [500, 2000, 10000], "seeds": [0, 99], "eigenvalue_floor": EIGEN_FLOOR},
        "t2": {"dimensions": [2, 4, 8], "betas": [0.25, 0.5, 1.0, 2.0, 4.0], "seeds": [0, 99], "flow_scale_grid": [0.0, 0.25, 0.5, 0.75, 1.0]},
        "t3": {
            "dimensions": DIMS,
            "true_ranks": TRUE_RANKS,
            "coupling_strengths": STRENGTHS,
            "rank_candidates": RANK_CANDIDATES,
            "train_samples": TRAIN_SAMPLES,
            "heldout_samples": HELDOUT_SAMPLES,
            "screening_heldout_samples": SCREENING_HELDOUT_SAMPLES,
            "eigenvalue_floor": EIGEN_FLOOR,
            "methods": ALL_METHODS,
            "development_seeds": [0, 49],
            "validation_seeds": [1000, 1049],
            "confirmation_seeds": [10000, 10199],
            "selection": selection,
        },
        "claim_firewall": {
            "correlon_positive_is_not_direct_causation": True,
            "correlon_equals_cca_for_this_operator_forbids_novelty_claim": True,
            "supplied_J_is_not_emergent": True,
        },
    }
    return config


def run_stage(stage: str) -> pd.DataFrame:
    ensure_output_dirs()
    if stage not in STAGE_SEEDS:
        raise ValueError(f"unknown stage: {stage}")
    seeds = STAGE_SEEDS[stage]
    selection = None
    if stage == "confirmation":
        selection_path = RESULTS / "timeless_T3_selection.json"
        config_path = RESULTS / "timeless_correlon_config.json"
        if not selection_path.exists() or not config_path.exists():
            raise RuntimeError("confirmation is blocked: validation freeze/configuration is missing")
        selection = json_load(selection_path)
        config = json_load(config_path)
        expected = selection.get("config_sha256")
        actual = sha256_json(config)
        if expected != actual:
            raise RuntimeError(f"confirmation is blocked: configuration hash mismatch {expected} != {actual}")

    rows: list[dict] = []
    for seed in seeds:
        for d in DIMS:
            j = canonical_J(2 * d)
            for true_rank in TRUE_RANKS:
                for strength in STRENGTHS:
                    _, covariance_true = make_case(d, true_rank, strength, seed)
                    train_rng = np.random.default_rng(7200000 + seed + d * 10000 + true_rank * 1000 + int(strength * 100))
                    hold_rng = np.random.default_rng(7300000 + seed + d * 10000 + true_rank * 1000 + int(strength * 100))
                    train = gaussian_samples(covariance_true, TRAIN_SAMPLES, train_rng)
                    covariance_hat = (train.T @ train) / float(TRAIN_SAMPLES)
                    heldout = gaussian_samples(covariance_true, HELDOUT_SAMPLES, hold_rng)
                    field_n = HELDOUT_SAMPLES if stage == "confirmation" else SCREENING_HELDOUT_SAMPLES
                    rank_plan = RANK_CANDIDATES
                    method_plan = ["correlon", "cca_equivalent", *LOW_DIM_METHODS, "full_state_oracle"]
                    if stage == "confirmation" and selection is not None:
                        rank_plan = sorted({int(selection["correlon_rank"]), int(selection["best_baseline_rank"])})
                        method_plan = ["correlon", "cca_equivalent", selection["best_baseline_method"], "full_state_oracle"]

                    for rank in rank_plan:
                        method_covariances, identity_diff = cross_reconstructions(
                            covariance_hat,
                            d,
                            rank,
                            np.random.default_rng(7400000 + seed + d * 10000 + true_rank * 1000 + int(strength * 100) + rank),
                        )
                        method_covariances["full_state_oracle"] = covariance_true
                        for method in method_plan:
                            if method not in method_covariances:
                                continue
                            metrics = evaluate_method(method_covariances[method], covariance_true, j, heldout, field_n)
                            rows.append(
                                {
                                    "stage": stage,
                                    "seed": seed,
                                    "dimension": d,
                                    "true_rank": true_rank,
                                    "strength": strength,
                                    "rank": rank,
                                    "method": method,
                                    "train_samples": TRAIN_SAMPLES,
                                    "heldout_samples": HELDOUT_SAMPLES,
                                    "field_evaluation_samples": field_n,
                                    "cca_identity_diff": identity_diff,
                                    **metrics,
                                }
                            )

    raw = pd.DataFrame(rows)
    raw_path = RESULTS / f"timeless_T3_{stage}_raw.csv"
    raw.to_csv(raw_path, index=False)
    existing = [RESULTS / f"timeless_T3_{name}_raw.csv" for name in ["development", "validation", "confirmation"]]
    available = [pd.read_csv(path) for path in existing if path.exists()]
    if available:
        pd.concat(available, ignore_index=True).to_csv(RESULTS / "timeless_T3_raw.csv", index=False)

    if stage == "validation":
        selection = selection_from_validation(raw)
        config = freeze_config(selection)
        config_sha = sha256_json(config)
        json_dump(RESULTS / "timeless_correlon_config.json", config)
        json_dump(RESULTS / "timeless_T3_selection.json", {**selection, "config_sha256": config_sha})
        print("T3 validation selection:", selection)
        print("T3 frozen configuration SHA-256:", config_sha)

    if stage == "confirmation":
        make_confirmation_figure(raw)
    print(f"T3 {stage}: {len(raw)} rows written to {raw_path}")
    return raw


def make_confirmation_figure(confirmation: pd.DataFrame) -> None:
    means = confirmation.groupby("method").generator_error.mean().sort_values()
    labels = ["Correlon" if x == "correlon" else "CCA-equivalent" if x == "cca_equivalent" else "Full oracle" if x == "full_state_oracle" else x for x in means.index]
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    bars = ax.bar(labels, means.values, color=["#d62728" if x == "correlon" else "#9467bd" if x == "cca_equivalent" else "#2ca02c" if x == "full_state_oracle" else "#1f77b4" for x in means.index])
    ax.set_ylabel("held-out normalized generator error")
    ax.set_title("T3: Correlon versus matched compressed baselines")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=18)
    for bar, value in zip(bars, means.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.savefig(FIGURES / "T3_method_comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=list(STAGE_SEEDS), required=True)
    args = parser.parse_args()
    run_stage(args.stage)
