"""T2: finite-dimensional modular-flow theorem reproduction."""

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
    best_positive_scale,
    ensure_output_dirs,
    git_context,
    json_dump,
    normalized_frobenius,
)


BETAS = [0.25, 0.5, 1.0, 2.0, 4.0]
DIMS = [2, 4, 8]
S_GRID = np.linspace(0.0, 1.0, 5)


def hermitian_basis(dim: int) -> list[np.ndarray]:
    basis: list[np.ndarray] = []
    for i in range(dim):
        m = np.zeros((dim, dim), dtype=complex)
        m[i, i] = 1.0
        basis.append(m)
    for i in range(dim):
        for j in range(i + 1, dim):
            sym = np.zeros((dim, dim), dtype=complex)
            sym[i, j] = sym[j, i] = 1.0 / np.sqrt(2.0)
            anti = np.zeros((dim, dim), dtype=complex)
            anti[i, j] = -1j / np.sqrt(2.0)
            anti[j, i] = 1j / np.sqrt(2.0)
            basis.extend([sym, anti])
    return basis


def diagonal_basis_from_h(h: np.ndarray) -> list[np.ndarray]:
    w, v = np.linalg.eigh(h)
    del w
    return [np.outer(v[:, i], v[:, i].conj()) for i in range(h.shape[0])]


def random_hermitian(dim: int, rng: np.random.Generator) -> np.ndarray:
    z = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    h = (z + z.conj().T) / 2.0
    h -= np.trace(h).real / dim * np.eye(dim)
    return h / max(np.linalg.norm(h), 1e-12) * np.sqrt(dim)


def pauli_hamiltonian() -> np.ndarray:
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    return 0.7 * sx + 0.35 * sz


def two_qubit_hamiltonian() -> np.ndarray:
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    eye = np.eye(2, dtype=complex)
    return (
        0.5 * np.kron(sz, eye)
        + 0.4 * np.kron(eye, sx)
        + 0.2 * np.kron(sz, sz)
        + 0.1 * np.kron(sx, sx)
    )


def gibbs_state(h: np.ndarray, beta: float) -> np.ndarray:
    w, v = np.linalg.eigh(h)
    weights = np.exp(-beta * (w - np.min(w)))
    rho = (v * weights) @ v.conj().T
    return rho / np.trace(rho).real


def modular_hamiltonian(rho: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh((rho + rho.conj().T) / 2.0)
    return (v * (-np.log(np.maximum(w, 1e-300)))) @ v.conj().T


def unitary_from_hermitian(h: np.ndarray, t: float) -> np.ndarray:
    w, v = np.linalg.eigh((h + h.conj().T) / 2.0)
    return (v * np.exp(1j * t * w)) @ v.conj().T


def evolve(u: np.ndarray, observable: np.ndarray) -> np.ndarray:
    return u @ observable @ u.conj().T


def flow_error(
    k_mod: np.ndarray,
    h: np.ndarray,
    scale: float,
    basis: list[np.ndarray],
    s_grid: np.ndarray = S_GRID,
) -> tuple[float, float]:
    errors = []
    expectation_errors = []
    psi = np.ones(h.shape[0], dtype=complex)
    psi /= np.linalg.norm(psi)
    for s in s_grid:
        u_mod = unitary_from_hermitian(k_mod, s)
        u_phys = unitary_from_hermitian(h, scale * s)
        for observable in basis:
            a = evolve(u_mod, observable)
            b = evolve(u_phys, observable)
            errors.append(normalized_frobenius(a, b))
            expectation_errors.append(abs(np.vdot(psi, (a - b) @ psi).real))
    return float(np.max(errors)), float(np.mean(expectation_errors))


def generator_metrics(k_mod: np.ndarray, h: np.ndarray, basis: list[np.ndarray]) -> dict:
    dim = h.shape[0]
    identity = np.eye(dim, dtype=complex)
    k0 = k_mod - np.trace(k_mod).real / dim * identity
    h0 = h - np.trace(h).real / dim * identity
    scale = best_positive_scale(h0, k0)
    generator_error = normalized_frobenius(scale * h0, k0)
    f_error, expectation_error = flow_error(k_mod, h, scale, basis)
    return {
        "scale_estimate": scale,
        "generator_error": generator_error,
        "flow_frobenius_error": f_error,
        "expectation_trajectory_error": expectation_error,
    }


def run(nseeds: int = 100, seed_start: int = 0) -> dict:
    ensure_output_dirs()
    rows: list[dict] = []
    controls: list[dict] = []

    for dim in DIMS:
        basis = hermitian_basis(dim)
        for seed in range(seed_start, seed_start + nseeds):
            h = random_hermitian(dim, np.random.default_rng(5100000 + seed + dim * 1000))
            for beta in BETAS:
                rho = gibbs_state(h, beta)
                k_mod = modular_hamiltonian(rho)
                metrics = generator_metrics(k_mod, h, basis)
                rows.append({"model": "random_hermitian", "dimension": dim, "seed": seed, "beta": beta, **metrics})

    h_pauli = pauli_hamiltonian()
    for beta in BETAS:
        rho = gibbs_state(h_pauli, beta)
        k_mod = modular_hamiltonian(rho)
        rows.append({"model": "two_level_pauli", "dimension": 2, "seed": -1, "beta": beta, **generator_metrics(k_mod, h_pauli, hermitian_basis(2))})

    h_two = two_qubit_hamiltonian()
    for beta in BETAS:
        rho = gibbs_state(h_two, beta)
        k_mod = modular_hamiltonian(rho)
        rows.append({"model": "weakly_coupled_two_qubit", "dimension": 4, "seed": -1, "beta": beta, **generator_metrics(k_mod, h_two, hermitian_basis(4))})

    rng = np.random.default_rng(6100000)
    for dim in DIMS:
        basis = hermitian_basis(dim)
        diag_basis = diagonal_basis_from_h(random_hermitian(dim, rng))
        for seed in range(min(nseeds, 50)):
            hidden_h = random_hermitian(dim, rng)
            eigenvalues = np.exp(rng.uniform(-2.0, 2.0, size=dim))
            unitary = random_hermitian(dim, rng)
            _, unitary_vecs = np.linalg.eigh(unitary)
            rho_non_gibbs = (unitary_vecs * (eigenvalues / eigenvalues.sum())) @ unitary_vecs.conj().T
            k_non = modular_hamiltonian(rho_non_gibbs)
            non_metrics = generator_metrics(k_non, hidden_h, basis)
            controls.append({"control": "generic_non_gibbs_unrelated_H", "dimension": dim, **non_metrics})

            h = random_hermitian(dim, rng)
            beta = 1.0
            rho = gibbs_state(h, beta)
            rot = random_hermitian(dim, rng)
            _, rot_u = np.linalg.eigh(rot)
            rho_rot = rot_u @ rho @ rot_u.conj().T
            k_rot = modular_hamiltonian(rho_rot)
            rotated_metrics = generator_metrics(k_rot, h, basis)
            controls.append({"control": "gibbs_eigenvectors_rotated", "dimension": dim, **rotated_metrics})

            # A deliberately over-restricted algebra: all projectors in H's eigenbasis commute.
            k_correct = modular_hamiltonian(gibbs_state(h, 1.0))
            wrong_flow, wrong_expectation = flow_error(k_correct, h, 1.0, diagonal_basis_from_h(h))
            controls.append(
                {
                    "control": "incorrect_diagonal_observable_algebra",
                    "dimension": dim,
                    "scale_estimate": 1.0,
                    "generator_error": np.nan,
                    "flow_frobenius_error": wrong_flow,
                    "expectation_trajectory_error": wrong_expectation,
                }
            )

    raw = pd.DataFrame(rows)
    control_df = pd.DataFrame(controls)
    raw.to_csv(RESULTS / "timeless_T2_raw.csv", index=False)
    control_df.to_csv(RESULTS / "timeless_T2_controls.csv", index=False)

    positive_gate = bool(raw.generator_error.max() <= 1e-8 and raw.flow_frobenius_error.max() <= 1e-6)
    summary = (
        raw.groupby(["model", "dimension", "beta"])
        .agg(
            generator_error_max=("generator_error", "max"),
            flow_error_max=("flow_frobenius_error", "max"),
            expectation_error_max=("expectation_trajectory_error", "max"),
            scale_mean=("scale_estimate", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(RESULTS / "timeless_T2_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    for model, group in raw[raw.seed >= 0].groupby("model"):
        grouped = group.groupby("beta").flow_frobenius_error.max().reindex(BETAS)
        ax.plot(BETAS, grouped.values, marker="o", label=model)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("beta")
    ax.set_ylabel("max flow Frobenius error after scale match")
    ax.set_title("T2: modular versus physical Gibbs flow")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(FIGURES / "T2_modular_vs_physical_flow.png", dpi=180)
    plt.close(fig)

    decision = {
        "phase": "T2",
        "verdict": "MODULAR_FLOW_CONTROL_PASS" if positive_gate else "IMPLEMENTATION_FAILURE",
        "positive_control_gate": {
            "max_generator_error": float(raw.generator_error.max()),
            "max_flow_error": float(raw.flow_frobenius_error.max()),
            "generator_threshold": 1e-8,
            "flow_threshold": 1e-6,
            "pass": positive_gate,
        },
        "negative_control_summary": {
            control: {
                "generator_error_median": float(group.generator_error.median()) if group.generator_error.notna().any() else None,
                "flow_error_median": float(group.flow_frobenius_error.median()),
            }
            for control, group in control_df.groupby("control")
        },
        "claim_firewall": "This is a known modular-flow theorem reproduction and not Correlon-specific evidence.",
        "configuration": {"dimensions": DIMS, "betas": BETAS, "s_grid": S_GRID.tolist(), "seeds": [seed_start, seed_start + nseeds - 1]},
        "git": git_context(),
    }
    json_dump(RESULTS / "timeless_T2_decision.json", decision)
    print(summary.to_string(index=False))
    print("\nT2 decision:", decision["verdict"])
    print("T2 gate:", decision["positive_control_gate"])
    return decision


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=0)
    args = parser.parse_args()
    run(args.seeds, args.seed_start)
