"""Shared deterministic infrastructure for Timeless Correlator T8."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from timeless_common import ROOT, bootstrap_ci, git_context


RESULTS = Path(os.environ.get("TIMELESS_T8_RESULTS_DIR", ROOT / "results")).resolve()
FIGURES = Path(os.environ.get("TIMELESS_T8_FIGURES_DIR", ROOT / "figures")).resolve()


HISTORY_LENGTHS = (16, 32, 64, 128, 256, 512, 1024)
LAG_GRID = (1, 2, 4, 8, 16, 32, 64, 128, 256)
STATE_GRID = (1, 2, 4, 8, 16, 32, 64, 128)
STAGE_SEEDS = {
    "development": range(30000, 30030),
    "validation": range(31000, 31050),
    "confirmation": range(40000, 40300),
}
BOOTSTRAP_SEED = 908172
ERROR_TARGET = 0.05
CONFIG_PATH = RESULTS / "timeless_T8_config.json"
FREEZE_PATH = RESULTS / "timeless_T8_validation_freeze.json"
CONFIRMATION_PATH = RESULTS / "timeless_T8_confirmation_summary.json"
CONFIRMATION_MARKER = RESULTS / ".timeless_T8_confirmation_opened"


def seeds_for(stage: str) -> list[int]:
    if stage not in STAGE_SEEDS:
        raise ValueError(stage)
    return list(STAGE_SEEDS[stage])


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_hash(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def replace_stage_rows(path: Path, rows: list[dict], stage: str) -> pd.DataFrame:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    incoming = pd.DataFrame(rows)
    if incoming.empty:
        raise ValueError(f"no rows for {path}")
    incoming["stage"] = stage
    if path.exists():
        existing = pd.read_csv(path)
        if stage == "confirmation" and bool((existing.stage == "confirmation").any()):
            raise RuntimeError(f"confirmation rows already exist: {path.name}")
        existing = existing[existing.stage != stage]
        incoming = pd.concat([existing, incoming], ignore_index=True)
    incoming.to_csv(path, index=False)
    return incoming


def stage_frame(path: Path, stage: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    selected = frame[frame.stage == stage].copy()
    if selected.empty:
        raise RuntimeError(f"missing {stage} rows in {path.name}")
    return selected


def summary(values: Iterable[float], failures: Iterable[bool] | None = None) -> dict:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0, "mean": None, "median": None, "standard_deviation": None,
                "CI95": [None, None], "5th_percentile": None, "failure_count": 0}
    ci = bootstrap_ci(x, seed=BOOTSTRAP_SEED, n_boot=2000)
    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "standard_deviation": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "CI95": [float(ci[0]), float(ci[1])],
        "5th_percentile": float(np.quantile(x, 0.05)),
        "failure_count": int(np.sum(list(failures))) if failures is not None else 0,
    }


def paired_relative_gain(candidate: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    return 1.0 - np.asarray(candidate, float) / np.maximum(np.asarray(baseline, float), 1e-15)


def assert_freeze() -> tuple[dict, dict]:
    if not CONFIG_PATH.exists() or not FREEZE_PATH.exists():
        raise RuntimeError("confirmation requires T8 config and validation freeze")
    config = read_json(CONFIG_PATH)
    freeze = read_json(FREEZE_PATH)
    if canonical_hash(config) != freeze.get("config_SHA256"):
        raise RuntimeError("T8 config SHA-256 mismatch")
    if freeze.get("positive_control_status") != "PASS":
        raise RuntimeError("T8 positive controls did not pass")
    return config, freeze


def current_git() -> dict:
    return git_context()


__all__ = [
    "ROOT", "RESULTS", "FIGURES", "HISTORY_LENGTHS", "LAG_GRID", "STATE_GRID",
    "STAGE_SEEDS", "BOOTSTRAP_SEED", "ERROR_TARGET", "CONFIG_PATH", "FREEZE_PATH",
    "CONFIRMATION_PATH", "CONFIRMATION_MARKER", "seeds_for", "write_json",
    "read_json", "canonical_hash", "replace_stage_rows", "stage_frame", "summary",
    "paired_relative_gain", "assert_freeze", "current_git",
]
