"""Numerical helpers shared by the Timeless Correlon experiments."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from scipy.linalg import expm


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def ensure_output_dirs() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def json_load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_json(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def git_context() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit, branch = "unknown", "unknown"
    return {"commit": commit, "branch": branch}


def sym(a: np.ndarray) -> np.ndarray:
    return (a + a.T.conj()) / 2


def eig_floor(a: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    a = sym(np.asarray(a))
    w, v = np.linalg.eigh(a)
    scale = max(1.0, float(np.max(np.abs(w))))
    w = np.maximum(w, floor * scale)
    return sym((v * w) @ v.T.conj())


def inv_spd(a: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    a = eig_floor(a, floor)
    w, v = np.linalg.eigh(a)
    return sym((v * (1.0 / w)) @ v.T.conj())


def sqrt_spd(a: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    a = eig_floor(a, floor)
    w, v = np.linalg.eigh(a)
    return sym((v * np.sqrt(w)) @ v.T.conj())


def invsqrt_spd(a: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    a = eig_floor(a, floor)
    w, v = np.linalg.eigh(a)
    return sym((v * (1.0 / np.sqrt(w))) @ v.T.conj())


def normalized_frobenius(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(b)), 1e-15)
    return float(np.linalg.norm(a - b) / denom)


def ci95(values: Iterable[float]) -> tuple[float, float]:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return (float("nan"), float("nan"))
    se = float(np.std(x, ddof=1) / np.sqrt(x.size))
    mean = float(np.mean(x))
    return mean - 1.96 * se, mean + 1.96 * se


def bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    seed: int = 12345,
    n_boot: int = 2000,
) -> tuple[float, float]:
    x = np.asarray(values)
    if x.ndim == 0:
        x = x.reshape(1)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boot = np.asarray([statistic(x[row]) for row in idx], dtype=float)
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def best_positive_scale(estimate: np.ndarray, target: np.ndarray) -> float:
    den = float(np.real(np.vdot(estimate, estimate)))
    if den <= 1e-15:
        return 0.0
    num = float(np.real(np.vdot(estimate, target)))
    return max(0.0, num / den)


def scale_aligned_error(estimate: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    scale = best_positive_scale(estimate, target)
    return scale, normalized_frobenius(scale * estimate, target)


def canonical_J(n: int) -> np.ndarray:
    if n % 2:
        raise ValueError("canonical symplectic dimension must be even")
    half = n // 2
    out = np.zeros((n, n), dtype=float)
    out[:half, half:] = np.eye(half)
    out[half:, :half] = -np.eye(half)
    return out


def random_orthogonal(n: int, rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(n, n)))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q @ np.diag(signs)


def random_complex_structure(n: int, rng: np.random.Generator) -> np.ndarray:
    j = canonical_J(n)
    q = random_orthogonal(n, rng)
    return q @ j @ q.T


def hamiltonian_flow_generator(k: np.ndarray, j: np.ndarray) -> np.ndarray:
    return j @ k


def heldout_field_metrics(
    estimate_a: np.ndarray,
    target_a: np.ndarray,
    samples: np.ndarray,
) -> dict:
    ve = samples @ estimate_a.T
    vt = samples @ target_a.T
    ve_norm = np.linalg.norm(ve, axis=1)
    vt_norm = np.linalg.norm(vt, axis=1)
    cos = np.sum(ve * vt, axis=1) / np.maximum(ve_norm * vt_norm, 1e-15)
    ve_unit = ve / np.maximum(ve_norm[:, None], 1e-15)
    vt_unit = vt / np.maximum(vt_norm[:, None], 1e-15)
    return {
        "vector_field_cosine": float(np.mean(cos)),
        "orbit_shape_error": float(np.sqrt(np.mean(np.sum((ve_unit - vt_unit) ** 2, axis=1))) / 2.0),
    }


def flow_matrix_error(estimate_a: np.ndarray, target_a: np.ndarray, scale: float, t: float = 0.25) -> float:
    e = expm(scale * estimate_a * t)
    target = expm(target_a * t)
    return normalized_frobenius(e, target)
