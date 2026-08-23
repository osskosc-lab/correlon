"""History-matched intervention construction and T8 response experiment."""

from __future__ import annotations

import argparse

import numpy as np

from Timeless_T8_generators import kernel
from timeless_t8_common import RESULTS, replace_stage_rows, seeds_for


TRAINING_INTERVENTIONS = (
    ("pulse", 0.5, 0), ("pulse", 1.0, 0), ("step", 1.0, 0),
)
HELDOUT_INTERVENTIONS = (
    ("pulse", 0.25, 0), ("pulse", 1.5, 0), ("paired_pulse", 1.0, 13),
)


def matched_history_pair(seed: int, length: int = 256) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    base = rng.normal(scale=0.35, size=length)
    shape = np.linspace(-1.0, 1.0, length)
    h1 = base + 0.8 * shape
    h2 = base - 0.8 * shape
    shared_current = float(rng.normal(scale=0.1))
    h1[-1] = shared_current
    h2[-1] = shared_current
    return h1, h2


def response(history: np.ndarray, intervention: tuple[str, float, int], target: bool) -> float:
    kind, amplitude, separation = intervention
    current = float(history[-1])
    if not target:
        latent_sufficient_state = current
        return float(amplitude * 0.6 * latent_sufficient_state)
    weights = kernel("L6_history_dependent_intervention", len(history))[::-1]
    weights = weights / max(float(np.sum(np.abs(weights))), 1e-15)
    memory = float(np.dot(weights, history))
    paired = 0.15 * np.exp(-separation / 40.0) if kind == "paired_pulse" else 0.0
    return float(amplitude * (0.15 * current + 1.8 * np.tanh(3.0 * memory) + paired))


def run_seed(seed: int) -> list[dict]:
    h1, h2 = matched_history_pair(seed)
    rows = []
    for target, world in ((False, "finite_markov_sufficient_state_control"),
                          (True, "long_memory_history_target")):
        for split, interventions in (("training", TRAINING_INTERVENTIONS),
                                     ("heldout", HELDOUT_INTERVENTIONS)):
            for kind, amplitude, separation in interventions:
                spec = (kind, amplitude, separation)
                r1, r2 = response(h1, spec, target), response(h2, spec, target)
                rows.append({
                    "seed": seed,
                    "world": world,
                    "split": split,
                    "intervention_kind": kind,
                    "amplitude": amplitude,
                    "paired_pulse_separation": separation,
                    "current_state_H1": h1[-1],
                    "current_state_H2": h2[-1],
                    "current_state_match_error": abs(float(h1[-1] - h2[-1])),
                    "history_distance": float(np.linalg.norm(h1[:-1] - h2[:-1])),
                    "response_H1": r1,
                    "response_H2": r2,
                    "Delta_H_R": abs(r1 - r2),
                    "history_only_variable_changed": True,
                    "heldout_intervention_not_in_training": split == "heldout",
                })
    return rows


def run(stage: str) -> None:
    rows = [row for seed in seeds_for(stage) for row in run_seed(seed)]
    replace_stage_rows(RESULTS / "timeless_T8_interventions.csv", rows, stage)
    print(f"T8 interventions {stage}: {len(rows)} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("development", "validation", "confirmation"), required=True)
    run(parser.parse_args().stage)

