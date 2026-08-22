"""Anti-Correlon random search over null systems with no direct X/Y edge."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .generators import World
from .metrics import correlon_score
from .utils import ROOT, seed_for


@dataclass(frozen=True)
class AdversarialParameters:
    latent_rank: int
    rho: float
    frequency: float
    strength: float
    noise: float
    drift: float
    x_delay: int
    y_delay: int
    sample_length: int


def _normalize(values: np.ndarray) -> np.ndarray:
    centered = values - np.mean(values, axis=0, keepdims=True)
    return centered / np.maximum(np.std(centered, axis=0, ddof=1, keepdims=True), 1e-12)


def _orthogonal_columns(dimension: int, rank: int, rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(dimension, rank)))
    return q[:, :rank]


def _shift_without_wrap(values: np.ndarray, delay: int) -> np.ndarray:
    if delay == 0:
        return values.copy()
    shifted = np.zeros_like(values)
    shifted[delay:] = values[:-delay]
    return shifted


def _latent_oscillators(
    length: int,
    rank: int,
    rho: float,
    frequency: float,
    rng: np.random.Generator,
) -> np.ndarray:
    latent = np.zeros((length, rank), dtype=float)
    innovations = rng.normal(size=(length, rank))
    if frequency <= 1e-8:
        scale = np.sqrt(max(1.0 - rho * rho, 1e-6))
        for index in range(1, length):
            latent[index] = rho * latent[index - 1] + scale * innovations[index]
    else:
        coefficient = 2.0 * rho * np.cos(2.0 * np.pi * frequency)
        second = -(rho**2)
        for index in range(2, length):
            latent[index] = coefficient * latent[index - 1] + second * latent[index - 2] + innovations[index]
    return _normalize(latent)


def sample_parameters(rng: np.random.Generator, config: dict[str, Any]) -> AdversarialParameters:
    settings = config["adversary"]
    return AdversarialParameters(
        latent_rank=int(rng.choice(settings["latent_ranks"])),
        rho=float(rng.uniform(settings["rho_low"], settings["rho_high"])),
        frequency=float(rng.uniform(settings["frequency_low"], settings["frequency_high"])),
        strength=float(rng.uniform(settings["strength_low"], settings["strength_high"])),
        noise=float(rng.uniform(settings["noise_low"], settings["noise_high"])),
        drift=float(rng.uniform(settings["drift_low"], settings["drift_high"])),
        x_delay=int(rng.choice(settings["delays"])),
        y_delay=int(rng.choice(settings["delays"])),
        sample_length=int(rng.choice(settings["sample_lengths"])),
    )


def generate_adversarial_null(
    parameters: AdversarialParameters,
    trial: int,
    rng: np.random.Generator,
    config: dict[str, Any],
) -> World:
    x_dim = int(config["data"]["x_dim"])
    y_dim = int(config["data"]["y_dim"])
    burn_in = int(config["data"]["burn_in"])
    total = parameters.sample_length + burn_in + max(parameters.x_delay, parameters.y_delay)
    latent = _latent_oscillators(total, parameters.latent_rank, parameters.rho, parameters.frequency, rng)
    x_latent = _shift_without_wrap(latent, parameters.x_delay)
    y_latent = _shift_without_wrap(latent, parameters.y_delay)
    load_x = _orthogonal_columns(x_dim, parameters.latent_rank, rng)
    load_y = _orthogonal_columns(y_dim, parameters.latent_rank, rng)
    coordinate = np.linspace(-1.0, 1.0, total)
    drift_signal = coordinate + 0.5 * np.sin(2.0 * np.pi * coordinate)
    drift_x = _orthogonal_columns(x_dim, 1, rng)[:, 0]
    drift_y = _orthogonal_columns(y_dim, 1, rng)[:, 0]
    x = parameters.strength * x_latent @ load_x.T
    y = parameters.strength * y_latent @ load_y.T
    x += parameters.drift * drift_signal[:, None] * drift_x
    y += parameters.drift * drift_signal[:, None] * drift_y
    x += parameters.noise * rng.normal(size=(total, x_dim))
    y += parameters.noise * rng.normal(size=(total, y_dim))
    start = burn_in + max(parameters.x_delay, parameters.y_delay)
    x = _normalize(x[start : start + parameters.sample_length])
    y = _normalize(y[start : start + parameters.sample_length])
    return World(
        x=x,
        y=y,
        meta={
            "world_class": "anti_correlon_null",
            "seed": int(config["seeds"]["adversary"]),
            "trial": int(trial),
            "target": False,
            "direct_edge": False,
            "persistent_identity": False,
            "oracle_edge_strength": 0.0,
            "mechanism": "latent_common_factors_plus_independent_noise_and_drift",
        },
    )


def run_adversarial_search(config: dict[str, Any]) -> dict[str, Any]:
    threshold = config["thresholds"]["positive_thresholds"].get("correlon_v1")
    if threshold is None:
        raise RuntimeError("pilot-frozen correlon_v1 threshold is required before adversarial search")
    master = np.random.default_rng(int(config["seeds"]["adversary"]))
    trials = int(config["adversary"]["trials"])
    rows: list[dict[str, Any]] = []
    best_score = -np.inf
    best_world: World | None = None
    best_parameters: AdversarialParameters | None = None
    running_max = -np.inf
    for trial in range(trials):
        parameters = sample_parameters(master, config)
        trial_rng = np.random.default_rng(int(config["seeds"]["adversary"]) + 10_007 * (trial + 1))
        world = generate_adversarial_null(parameters, trial, trial_rng, config)
        score, diagnostics = correlon_score(
            world.x,
            world.y,
            config["metric"],
            seed_for(int(config["seeds"]["adversary"]), "metric_null", config, offset=trial),
        )
        running_max = max(running_max, score)
        row = {
            "trial": trial,
            **asdict(parameters),
            "correlon_score": float(score),
            "threshold": float(threshold),
            "crosses_threshold": bool(score > float(threshold)),
            "running_max": float(running_max),
            "direct_edge": False,
            "T_iso": float(diagnostics["T_iso"]),
            "T_floor": float(diagnostics["T_floor"]),
        }
        rows.append(row)
        if score > best_score:
            best_score = float(score)
            best_world = world
            best_parameters = parameters
        if (trial + 1) % 25 == 0:
            print(f"adversary {trial + 1}/{trials}: running_max={running_max:.6f}", flush=True)
    if best_world is None or best_parameters is None:
        raise AssertionError("adversarial search produced no candidate")
    frame = pd.DataFrame(rows)
    result_path = ROOT / config["outputs"]["results_dir"] / "correlon_zero_adversarial_cases.csv"
    frame.to_csv(result_path, index=False)
    np.savez_compressed(
        ROOT / config["outputs"]["results_dir"] / "correlon_zero_best_adversarial.npz",
        x=best_world.x,
        y=best_world.y,
        parameters=np.array([asdict(best_parameters)], dtype=object),
    )
    crossing_count = int(frame.crosses_threshold.sum())
    return {
        "trials": trials,
        "threshold": float(threshold),
        "maximum_score": float(best_score),
        "crossing_count": crossing_count,
        "crossing_rate": float(crossing_count / trials),
        "formal_repeated_crossing": bool(crossing_count >= int(config["thresholds"]["adversarial_crossing_count"])),
        "best_trial": int(frame.loc[frame.correlon_score.idxmax(), "trial"]),
        "best_parameters": asdict(best_parameters),
        "direct_edge_absent_by_construction": True,
    }

