"""Shared, deterministic infrastructure for Timeless Correlator T4-T7."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from timeless_common import FIGURES, RESULTS, ROOT, bootstrap_ci, ci95, ensure_output_dirs


STAGE_SEEDS = {
    "development": range(0, 30),
    "validation": range(1000, 1050),
    "confirmation": range(20000, 20300),
}
BOOTSTRAP_SEED = 881122
CONFIG_PATH = RESULTS / "timeless_T4T7_config.json"
VALIDATION_PATH = RESULTS / "timeless_T4T7_validation.json"
CONFIRMATION_PATH = RESULTS / "timeless_T4T7_confirmation.json"
CONFIRMATION_MARKER = RESULTS / ".timeless_T4T7_confirmation_opened"


def canonical_json_hash(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def seeds_for(stage: str) -> list[int]:
    if stage not in STAGE_SEEDS:
        raise ValueError(f"unknown stage: {stage}")
    return list(STAGE_SEEDS[stage])


def replace_stage_rows(path: Path, rows: list[dict], stage: str) -> pd.DataFrame:
    """Replace development/validation rows; never overwrite confirmation rows."""
    ensure_output_dirs()
    incoming = pd.DataFrame(rows)
    if incoming.empty:
        raise ValueError(f"no rows generated for {path.name}")
    incoming["stage"] = stage
    if path.exists():
        current = pd.read_csv(path)
        if stage == "confirmation" and bool((current.stage == "confirmation").any()):
            raise RuntimeError(f"confirmation rows already exist in {path}")
        current = current[current.stage != stage]
        incoming = pd.concat([current, incoming], ignore_index=True)
    incoming.to_csv(path, index=False)
    return incoming


def assert_confirmation_freeze() -> tuple[dict, dict]:
    if not CONFIG_PATH.exists() or not VALIDATION_PATH.exists():
        raise RuntimeError("confirmation requires frozen config and validation hash")
    config = read_json(CONFIG_PATH)
    validation = read_json(VALIDATION_PATH)
    actual = canonical_json_hash(config)
    if actual != validation.get("validation_config_SHA256"):
        raise RuntimeError("frozen config SHA-256 mismatch")
    leakage = RESULTS / "timeless_T4T7_leakage.json"
    if not leakage.exists() or read_json(leakage).get("status") != "PASS":
        raise RuntimeError("confirmation requires a passing clock-leakage audit")
    return config, validation


def summary(values: Iterable[float], failure_mask: Iterable[bool] | None = None) -> dict:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0, "mean": None, "median": None, "standard_deviation": None,
                "CI95": [None, None], "5th_percentile": None, "failure_count": 0}
    ci = bootstrap_ci(x, seed=BOOTSTRAP_SEED, n_boot=2000)
    failures = int(np.sum(list(failure_mask))) if failure_mask is not None else 0
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "standard_deviation": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "CI95": [float(ci[0]), float(ci[1])],
        "5th_percentile": float(np.quantile(x, 0.05)),
        "failure_count": failures,
    }


def transitive_closure(adjacency: np.ndarray) -> np.ndarray:
    reach = np.asarray(adjacency, dtype=bool).copy()
    n = reach.shape[0]
    for k in range(n):
        reach |= reach[:, [k]] & reach[[k], :]
    return reach


def has_directed_cycle(adjacency: np.ndarray) -> bool:
    reach = transitive_closure(adjacency)
    return bool(np.any(np.diag(reach)))


def topological_order(adjacency: np.ndarray) -> list[int] | None:
    adjacency = np.asarray(adjacency, dtype=bool)
    indegree = adjacency.sum(axis=0).astype(int)
    ready = sorted(np.flatnonzero(indegree == 0).tolist())
    order: list[int] = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for child in np.flatnonzero(adjacency[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(int(child))
                ready.sort()
    return order if len(order) == adjacency.shape[0] else None


def binary_metrics(predicted: np.ndarray, truth: np.ndarray) -> tuple[float, float, float]:
    p = np.asarray(predicted, dtype=bool).copy()
    t = np.asarray(truth, dtype=bool).copy()
    np.fill_diagonal(p, False)
    np.fill_diagonal(t, False)
    tp = int(np.sum(p & t))
    fp = int(np.sum(p & ~t))
    fn = int(np.sum(~p & t))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-15)
    return float(precision), float(recall), float(f1)


def kendall_tau(order: list[int], truth_order: list[int]) -> float:
    if len(order) < 2:
        return 1.0
    pos = {node: i for i, node in enumerate(truth_order)}
    concordant = discordant = 0
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            if pos[order[i]] < pos[order[j]]:
                concordant += 1
            else:
                discordant += 1
    return (concordant - discordant) / max(concordant + discordant, 1)


def auc_score(labels: Iterable[int], scores: Iterable[float]) -> float:
    y = np.asarray(list(labels), dtype=int)
    s = np.asarray(list(scores), dtype=float)
    pos = s[y == 1]
    neg = s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    wins = sum(float(a > b) + 0.5 * float(a == b) for a in pos for b in neg)
    return float(wins / (len(pos) * len(neg)))


def positive_affine_error(estimate: np.ndarray, truth: np.ndarray) -> float:
    x = np.asarray(estimate, dtype=float)
    y = np.asarray(truth, dtype=float)
    design = np.column_stack([x, np.ones_like(x)])
    slope, offset = np.linalg.lstsq(design, y, rcond=None)[0]
    if slope <= 0:
        return 1.0
    fit = slope * x + offset
    denom = max(float(np.linalg.norm(y - np.mean(y))), 1e-15)
    return float(np.linalg.norm(fit - y) / denom)


def pairwise_rank_accuracy(estimate: np.ndarray, truth: np.ndarray) -> float:
    e = np.asarray(estimate, dtype=float)
    t = np.asarray(truth, dtype=float)
    correct = total = 0.0
    for i in range(len(e)):
        for j in range(i + 1, len(e)):
            st = np.sign(t[j] - t[i])
            se = np.sign(e[j] - e[i])
            if st == 0:
                continue
            correct += 1.0 if se == st else 0.5 if se == 0 else 0.0
            total += 1.0
    return float(correct / max(total, 1.0))


def current_git() -> dict:
    from timeless_common import git_context

    return git_context()


__all__ = [
    "ROOT", "RESULTS", "FIGURES", "STAGE_SEEDS", "BOOTSTRAP_SEED",
    "CONFIG_PATH", "VALIDATION_PATH", "CONFIRMATION_PATH", "CONFIRMATION_MARKER",
    "canonical_json_hash", "write_json", "read_json", "seeds_for",
    "replace_stage_rows", "assert_confirmation_freeze", "summary", "ci95",
    "transitive_closure", "has_directed_cycle", "topological_order",
    "binary_metrics", "kendall_tau", "auc_score", "positive_affine_error",
    "pairwise_rank_accuracy", "current_git",
]
