"""T1: restricted Gaussian-Gibbs covariance-to-flow recovery."""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from timeless_common import (
    FIGURES,
    RESULTS,
    canonical_J,
    ci95,
    eig_floor,
    ensure_output_dirs,
    flow_matrix_error,
    git_context,
    heldout_field_metrics,
    inv_spd,
    json_dump,
    normalized_frobenius,
    random_complex_structure,
    random_orthogonal,
    scale_aligned_error,
)


DIMS = [2, 4, 8, 16]
CONDITIONS = [1.5, 3.0, 10.0, 30.0]
BETAS = [0.5, 1.0, 2.0, 4.0]
SAMPLE_SIZES = [500, 2000, 10000]
EIGEN_FLOOR = 1e-8


def random_spd(dim: int, condition: float, rng: np.random.Generator) -> np.ndarray:
    q = random_orthogonal(dim, rng)
    spectrum = np.geomspace(1.0, condition, dim)
    return (q * spectrum) @ q.T


def gaussian_samples(covariance: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    chol = np.linalg.cholesky(eig_floor(covariance, EIGEN_FLOOR))
    return rng.normal(size=(n, covariance.shape[0])) @ chol.T


def frequency_error(a_hat: np.ndarray, a_true: np.ndarray, scale: float) -> float:
    f_hat = np.sort(np.abs(np.imag(np.linalg.eigvals(scale * a_hat))))
    f_true = np.sort(np.abs(np.imag(np.linalg.eigvals(a_true))))
    return float(np.mean(np.abs(f_hat - f_true) / np.maximum(f_true, 1e-12)))


def pca_rank_matched(covariance: np.ndarray, rank: int) -> np.ndarray:
    w, v = np.linalg.eigh(eig_floor(covariance, EIGEN_FLOOR))
    order = np.argsort(w)[::-1]
    w = w[order]
    v = v[:, order]
    k = min(rank, len(w))
    residual = float(np.mean(w[k:])) if k < len(w) else float(EIGEN_FLOOR)
    return eig_floor((v[:, :k] * w[:k]) @ v[:, :k].T + residual * np.eye(len(w)), EIGEN_FLOOR)


def evaluate_covariance(
    covariance_hat: np.ndarray,
    covariance_true: np.ndarray,
    j: np.ndarray,
    heldout: np.ndarray,
) -> dict:
    k_hat = inv_spd(covariance_hat, EIGEN_FLOOR)
    k_true = inv_spd(covariance_true, EIGEN_FLOOR)
    a_hat = j @ k_hat
    a_true = j @ k_true
    scale, gen_error = scale_aligned_error(a_hat, a_true)
    field = heldout_field_metrics(scale * a_hat, a_true, heldout)
    return {
        "scale": scale,
        "generator_error": gen_error,
        "vector_field_cosine": field["vector_field_cosine"],
        "orbit_shape_error": field["orbit_shape_error"],
        "frequency_error": frequency_error(a_hat, a_true, scale),
        "trajectory_error": flow_matrix_error(a_hat, a_true, scale, t=0.25),
    }


def make_figure(raw: pd.DataFrame) -> None:
    subset = raw[(raw.condition <= 10.0) & (raw.sample_size.isin(SAMPLE_SIZES))]
    grouped = subset.groupby("sample_size").generator_error
    means = grouped.mean().reindex(SAMPLE_SIZES)
    lows = grouped.quantile(0.25).reindex(SAMPLE_SIZES)
    highs = grouped.quantile(0.75).reindex(SAMPLE_SIZES)
    x = np.asarray(SAMPLE_SIZES, dtype=float)
    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    ax.plot(x, means.values, marker="o", color="#1f77b4", label="median/mean cell aggregate")
    ax.fill_between(x, lows.values, highs.values, color="#1f77b4", alpha=0.18, label="interquartile range")
    ax.set_xscale("log")
    ax.set_xlabel("training samples")
    ax.set_ylabel("scale-aligned normalized generator error")
    ax.set_title("T1: covariance-to-generator recovery")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(FIGURES / "T1_generator_recovery.png", dpi=180)
    plt.close(fig)


def run(nseeds: int = 100, seed_start: int = 0) -> dict:
    ensure_output_dirs()
    rows: list[dict] = []
    control_rows: list[dict] = []
    j_cache = {dim: canonical_J(dim) for dim in DIMS}

    for dim in DIMS:
        j = j_cache[dim]
        for condition in CONDITIONS:
            for beta in BETAS:
                for sample_size in SAMPLE_SIZES:
                    for seed in range(seed_start, seed_start + nseeds):
                        rng = np.random.default_rng(3100000 + seed + dim * 100000 + int(condition * 1000) * 100 + int(beta * 10))
                        k_true = random_spd(dim, condition, rng)
                        covariance_true = inv_spd(beta * k_true, EIGEN_FLOOR)
                        samples = gaussian_samples(covariance_true, sample_size, rng)
                        covariance_hat = (samples.T @ samples) / float(sample_size)
                        heldout = gaussian_samples(covariance_true, 128, rng)
                        metrics = evaluate_covariance(covariance_hat, covariance_true, j, heldout)
                        row = {
                            "dimension": dim,
                            "condition": condition,
                            "beta": beta,
                            "sample_size": sample_size,
                            "seed": seed,
                            **metrics,
                        }

                        if sample_size == 2000 and beta == 1.0 and condition <= 10.0:
                            control_rng = np.random.default_rng(4100000 + seed + dim * 1000 + int(condition * 100))
                            permutation = control_rng.permutation(dim)
                            controls = {
                                "diagonal_only": np.diag(np.diag(covariance_hat)),
                                "coordinate_shuffled": covariance_hat[np.ix_(permutation, permutation)],
                            }
                            w, v = np.linalg.eigh(eig_floor(covariance_hat, EIGEN_FLOOR))
                            q = random_orthogonal(dim, control_rng)
                            controls["random_SPD_matched_spectrum"] = (q * w) @ q.T
                            controls["PCA_rank_matched"] = pca_rank_matched(covariance_hat, max(1, dim // 2))
                            for control_name, control_cov in controls.items():
                                cmetrics = evaluate_covariance(control_cov, covariance_true, j, heldout)
                                control_rows.append(
                                    {
                                        "dimension": dim,
                                        "condition": condition,
                                        "beta": beta,
                                        "sample_size": sample_size,
                                        "seed": seed,
                                        "control": control_name,
                                        **cmetrics,
                                    }
                                )

                            j_alt = random_complex_structure(dim, control_rng)
                            a_true = j @ inv_spd(covariance_true, EIGEN_FLOOR)
                            a_alt = j_alt @ inv_spd(covariance_true, EIGEN_FLOOR)
                            alt_scale, alt_error = scale_aligned_error(a_alt, a_true)
                            control_rows.append(
                                {
                                    "dimension": dim,
                                    "condition": condition,
                                    "beta": beta,
                                    "sample_size": sample_size,
                                    "seed": seed,
                                    "control": "unknown_symplectic_structure",
                                    "scale": alt_scale,
                                    "generator_error": alt_error,
                                    "vector_field_cosine": np.nan,
                                    "orbit_shape_error": np.nan,
                                    "frequency_error": np.nan,
                                    "trajectory_error": flow_matrix_error(a_alt, a_true, alt_scale, t=0.25),
                                }
                            )
                        rows.append(row)

    raw = pd.DataFrame(rows)
    controls = pd.DataFrame(control_rows)
    raw.to_csv(RESULTS / "timeless_T1_raw.csv", index=False)
    controls.to_csv(RESULTS / "timeless_T1_controls.csv", index=False)

    target = raw[(raw.sample_size >= 2000) & (raw.condition <= 10.0)]
    median_error = float(target.generator_error.median())
    median_cosine = float(target.vector_field_cosine.median())
    median_by_n = raw[raw.condition <= 10.0].groupby("sample_size").generator_error.median().reindex(SAMPLE_SIZES)
    monotonic = bool(median_by_n.iloc[0] >= median_by_n.iloc[1] >= median_by_n.iloc[2])
    positive_pass = bool(median_error <= 0.10 and median_cosine >= 0.95 and monotonic)

    ambiguity = controls[controls.control == "unknown_symplectic_structure"]
    amb_lo, amb_hi = ci95(ambiguity.generator_error)
    algebraic_ambiguity = bool(amb_lo >= 0.20)

    summary = (
        raw.groupby(["dimension", "condition", "sample_size"])
        .agg(
            generator_error_median=("generator_error", "median"),
            generator_error_mean=("generator_error", "mean"),
            vector_field_cosine_median=("vector_field_cosine", "median"),
            orbit_shape_error_median=("orbit_shape_error", "median"),
            frequency_error_median=("frequency_error", "median"),
            trajectory_error_median=("trajectory_error", "median"),
        )
        .reset_index()
    )
    summary.to_csv(RESULTS / "timeless_T1_summary.csv", index=False)
    make_figure(raw)

    decision = {
        "phase": "T1",
        "verdict": "RESTRICTED_GIBBS_GAUSSIAN_RECOVERY" if positive_pass else "IMPLEMENTATION_OR_MODEL_FAILURE",
        "positive_control_gate": {
            "median_generator_error_N_ge_2000_cond_le_10": median_error,
            "median_vector_field_cosine_N_ge_2000_cond_le_10": median_cosine,
            "median_generator_error_by_sample_size_cond_le_10": {str(k): float(v) for k, v in median_by_n.items()},
            "generator_error_threshold": 0.10,
            "vector_field_cosine_threshold": 0.95,
            "monotonicity": monotonic,
            "pass": positive_pass,
        },
        "unknown_symplectic_structure_ablation": {
            "generator_error_CI95": [amb_lo, amb_hi],
            "threshold_lower_bound": 0.20,
            "pass": algebraic_ambiguity,
            "interpretation": "C requires additional algebraic/symplectic structure" if algebraic_ambiguity else "ambiguity gate did not pass",
        },
        "allowed_claim": "C can encode a flow generator only in the restricted Gibbs-Gaussian regime when the canonical algebraic structure is supplied independently.",
        "configuration": {
            "dimensions": DIMS,
            "condition_numbers": CONDITIONS,
            "betas": BETAS,
            "sample_sizes": SAMPLE_SIZES,
            "seeds": [seed_start, seed_start + nseeds - 1],
            "eigenvalue_floor": EIGEN_FLOOR,
        },
        "git": git_context(),
    }
    json_dump(RESULTS / "timeless_T1_decision.json", decision)
    print(summary.to_string(index=False))
    print("\nT1 decision:", decision["verdict"])
    print("T1 gate:", decision["positive_control_gate"])
    print("T1 unknown-J gate:", decision["unknown_symplectic_structure_ablation"])
    return decision


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=0)
    args = parser.parse_args()
    run(args.seeds, args.seed_start)
