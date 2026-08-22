"""Synthetic target and null worlds for Correlon Zero."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .utils import seed_for


@dataclass(frozen=True)
class World:
    x: np.ndarray
    y: np.ndarray
    meta: dict[str, Any]

    def with_data(self, x: np.ndarray, y: np.ndarray, **meta_updates: Any) -> "World":
        metadata = dict(self.meta)
        metadata.update(meta_updates)
        return replace(self, x=np.asarray(x, dtype=float), y=np.asarray(y, dtype=float), meta=metadata)


def _orthogonal_columns(dimension: int, count: int, rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(dimension, count)))
    return q[:, :count]


def _normalize_channels(values: np.ndarray) -> np.ndarray:
    centered = values - np.mean(values, axis=0, keepdims=True)
    return centered / np.maximum(np.std(centered, axis=0, ddof=1, keepdims=True), 1e-12)


def _finalize(x: np.ndarray, y: np.ndarray, metadata: dict[str, Any], burn_in: int) -> World:
    x_out = _normalize_channels(np.asarray(x[burn_in:], dtype=float))
    y_out = _normalize_channels(np.asarray(y[burn_in:], dtype=float))
    if len(x_out) != len(y_out) or not np.all(np.isfinite(x_out)) or not np.all(np.isfinite(y_out)):
        raise FloatingPointError("synthetic world contains invalid data")
    return World(x=x_out, y=y_out, meta=metadata)


def _ar_scalar(length: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    values = np.zeros(length, dtype=float)
    innovations = rng.normal(size=length)
    scale = np.sqrt(max(1.0 - rho * rho, 1e-6))
    for index in range(1, length):
        values[index] = rho * values[index - 1] + scale * innovations[index]
    return _normalize_channels(values[:, None])[:, 0]


def _persistent_shared_mode(
    time_points: int,
    x_dim: int,
    y_dim: int,
    burn_in: int,
    rng: np.random.Generator,
    edge_scale: float,
) -> World:
    total = time_points + burn_in
    x = np.zeros((total, x_dim), dtype=float)
    y = np.zeros((total, y_dim), dtype=float)
    u = _orthogonal_columns(x_dim, 1, rng)[:, 0]
    v = _orthogonal_columns(y_dim, 1, rng)[:, 0]
    ex = rng.normal(size=x.shape)
    ey = rng.normal(size=y.shape)
    cross = 0.28 * edge_scale
    for index in range(1, total):
        x[index] = 0.65 * x[index - 1] + cross * u * float(v @ y[index - 1]) + 0.60 * ex[index]
        y[index] = 0.65 * y[index - 1] + cross * v * float(u @ x[index - 1]) + 0.60 * ey[index]
    spectral_radius = 0.65 + abs(cross)
    if spectral_radius >= 1.0:
        raise AssertionError("persistent_shared_mode stability condition failed")
    return _finalize(
        x,
        y,
        {
            "mechanism": "fixed_bidirectional_rank1_cross_lag",
            "direct_edge": bool(edge_scale != 0.0),
            "persistent_identity": bool(edge_scale != 0.0),
            "oracle_edge_strength": float(abs(cross)),
            "target": True,
        },
        burn_in,
    )


def _direct_coupling(
    time_points: int,
    x_dim: int,
    y_dim: int,
    burn_in: int,
    rng: np.random.Generator,
    edge_scale: float,
) -> World:
    total = time_points + burn_in
    x = np.zeros((total, x_dim), dtype=float)
    y = np.zeros((total, y_dim), dtype=float)
    u = _orthogonal_columns(x_dim, 1, rng)[:, 0]
    v = _orthogonal_columns(y_dim, 1, rng)[:, 0]
    ex = rng.normal(size=x.shape)
    ey = rng.normal(size=y.shape)
    edge = 0.65 * edge_scale
    delay = 2
    for index in range(1, total):
        x[index] = 0.65 * x[index - 1] + 0.60 * ex[index]
        source_index = max(index - delay, 0)
        response = np.tanh(float(u @ x[source_index]))
        y[index] = 0.55 * y[index - 1] + edge * v * response + 0.60 * ey[index]
    return _finalize(
        x,
        y,
        {
            "mechanism": "fixed_nonlinear_X_to_Y_delay2",
            "direct_edge": bool(edge_scale != 0.0),
            "persistent_identity": bool(edge_scale != 0.0),
            "oracle_edge_strength": float(abs(edge)),
            "target": True,
        },
        burn_in,
    )


def _history_dependent(
    time_points: int,
    x_dim: int,
    y_dim: int,
    burn_in: int,
    rng: np.random.Generator,
    edge_scale: float,
) -> World:
    total = time_points + burn_in
    x = np.zeros((total, x_dim), dtype=float)
    y = np.zeros((total, y_dim), dtype=float)
    memory = np.zeros(total, dtype=float)
    u = _orthogonal_columns(x_dim, 1, rng)[:, 0]
    v = _orthogonal_columns(y_dim, 1, rng)[:, 0]
    ex = rng.normal(size=x.shape)
    ey = rng.normal(size=y.shape)
    edge = 0.90 * edge_scale
    for index in range(1, total):
        x[index] = 0.65 * x[index - 1] + 0.60 * ex[index]
        memory[index] = 0.92 * memory[index - 1] + 0.08 * float(u @ x[index - 1])
        y[index] = 0.55 * y[index - 1] + edge * v * np.tanh(memory[index]) + 0.60 * ey[index]
    return _finalize(
        x,
        y,
        {
            "mechanism": "fixed_long_history_X_to_Y",
            "direct_edge": bool(edge_scale != 0.0),
            "persistent_identity": bool(edge_scale != 0.0),
            "oracle_edge_strength": float(abs(edge)),
            "target": True,
        },
        burn_in,
    )


def phase_randomize_matrix(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    output = np.empty_like(values, dtype=float)
    for column in range(values.shape[1]):
        series = values[:, column]
        spectrum = np.fft.rfft(series)
        phases = rng.uniform(0.0, 2.0 * np.pi, size=len(spectrum))
        phases[0] = np.angle(spectrum[0])
        if len(series) % 2 == 0:
            phases[-1] = np.angle(spectrum[-1])
        randomized = np.abs(spectrum) * np.exp(1j * phases)
        output[:, column] = np.fft.irfft(randomized, n=len(series))
    return _normalize_channels(output)


def _fit_var1(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = values - np.mean(values, axis=0, keepdims=True)
    previous = centered[:-1]
    current = centered[1:]
    ridge = 1e-5 * np.eye(previous.shape[1])
    coefficient = np.linalg.solve(previous.T @ previous + ridge, previous.T @ current).T
    radius = float(np.max(np.abs(np.linalg.eigvals(coefficient))))
    if radius >= 0.98:
        coefficient *= 0.98 / max(radius, 1e-12)
    residual = current - previous @ coefficient.T
    covariance = residual.T @ residual / max(len(residual) - 1, 1) + 1e-6 * np.eye(values.shape[1])
    return coefficient, covariance


def simulate_var1(
    coefficient: np.ndarray,
    covariance: np.ndarray,
    time_points: int,
    burn_in: int,
    rng: np.random.Generator,
) -> np.ndarray:
    total = time_points + burn_in
    values = np.zeros((total, coefficient.shape[0]), dtype=float)
    innovations = rng.multivariate_normal(np.zeros(coefficient.shape[0]), covariance, size=total)
    for index in range(1, total):
        values[index] = coefficient @ values[index - 1] + innovations[index]
    return _normalize_channels(values[burn_in:])


def simulate_separate_var(
    x: np.ndarray,
    y: np.ndarray,
    burn_in: int,
    rng_x: np.random.Generator,
    rng_y: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    ax, qx = _fit_var1(x)
    ay, qy = _fit_var1(y)
    return (
        simulate_var1(ax, qx, len(x), burn_in, rng_x),
        simulate_var1(ay, qy, len(y), burn_in, rng_y),
    )


def _negative_world(
    world_class: str,
    time_points: int,
    x_dim: int,
    y_dim: int,
    burn_in: int,
    rng: np.random.Generator,
    seed: int,
    config: dict[str, Any],
) -> World:
    total = time_points + burn_in
    metadata: dict[str, Any] = {
        "mechanism": world_class,
        "direct_edge": False,
        "persistent_identity": False,
        "oracle_edge_strength": 0.0,
        "target": False,
    }
    if world_class == "independent_noise":
        x = np.zeros((total, x_dim))
        y = np.zeros((total, y_dim))
        ex = rng.normal(size=x.shape)
        ey = rng.normal(size=y.shape)
        for index in range(1, total):
            x[index] = 0.70 * x[index - 1] + ex[index]
            y[index] = 0.70 * y[index - 1] + ey[index]
        return _finalize(x, y, metadata, burn_in)

    if world_class == "common_driver":
        z = _ar_scalar(total, 0.93, rng)
        u = _orthogonal_columns(x_dim, 1, rng)[:, 0]
        v = _orthogonal_columns(y_dim, 1, rng)[:, 0]
        x = np.zeros((total, x_dim))
        y = np.zeros((total, y_dim))
        ex = rng.normal(size=x.shape)
        ey = rng.normal(size=y.shape)
        for index in range(1, total):
            x[index] = 0.55 * x[index - 1] + 0.75 * u * z[index] + 0.60 * ex[index]
            y[index] = 0.55 * y[index - 1] + 0.75 * v * z[index] + 0.60 * ey[index]
        metadata["persistent_identity"] = True
        metadata["mechanism"] = "latent_AR1_common_driver"
        return _finalize(x, y, metadata, burn_in)

    if world_class == "matched_low_rank":
        factor = rng.normal(size=total)
        u = _orthogonal_columns(x_dim, 1, rng)[:, 0]
        v = _orthogonal_columns(y_dim, 1, rng)[:, 0]
        x = 1.20 * factor[:, None] * u + 0.60 * rng.normal(size=(total, x_dim))
        y = 1.20 * factor[:, None] * v + 0.60 * rng.normal(size=(total, y_dim))
        metadata["persistent_identity"] = True
        metadata["mechanism"] = "iid_rank1_common_factor"
        return _finalize(x, y, metadata, burn_in)

    if world_class == "matched_spectrum":
        base = _direct_coupling(time_points, x_dim, y_dim, burn_in, rng, edge_scale=1.0)
        randomized_x = phase_randomize_matrix(base.x, rng)
        randomized_y = phase_randomize_matrix(base.y, rng)
        metadata["mechanism"] = "independent_phase_randomized_target_spectra"
        return World(randomized_x, randomized_y, metadata)

    if world_class == "matched_autocorrelation":
        base = _direct_coupling(time_points, x_dim, y_dim, burn_in, rng, edge_scale=1.0)
        x_out, y_out = simulate_separate_var(base.x, base.y, burn_in, rng, rng)
        metadata["mechanism"] = "separate_VAR1_autocorrelation_match"
        return World(x_out, y_out, metadata)

    if world_class in {"switching_mode", "transient_mode"}:
        x = np.zeros((total, x_dim))
        y = np.zeros((total, y_dim))
        ux = _orthogonal_columns(x_dim, 2, rng)
        vy = _orthogonal_columns(y_dim, 2, rng)
        ex = rng.normal(size=x.shape)
        ey = rng.normal(size=y.shape)
        for index in range(1, total):
            x[index] = 0.65 * x[index - 1] + 0.60 * ex[index]
            observed_index = max(index - 2, 0)
            active = True
            direction = 0
            record_index = index - burn_in
            if world_class == "switching_mode":
                direction = int(record_index >= time_points // 2)
            else:
                active = time_points // 3 <= record_index < 2 * time_points // 3
            edge = 0.65 if active else 0.0
            y[index] = (
                0.55 * y[index - 1]
                + edge * vy[:, direction] * np.tanh(float(ux[:, direction] @ x[observed_index]))
                + 0.60 * ey[index]
            )
        metadata["mechanism"] = "switching_direct_edge" if world_class == "switching_mode" else "transient_direct_edge"
        return _finalize(x, y, metadata, burn_in)

    if world_class == "nonstationary_drift":
        coordinate = np.linspace(-1.0, 1.0, total)
        drift = coordinate + 0.75 * np.sin(2.0 * np.pi * 1.5 * (coordinate + 1.0) / 2.0)
        u = _orthogonal_columns(x_dim, 1, rng)[:, 0]
        v = _orthogonal_columns(y_dim, 1, rng)[:, 0]
        x = 1.40 * drift[:, None] * u + rng.normal(scale=0.70, size=(total, x_dim))
        y = 1.40 * drift[:, None] * v + rng.normal(scale=0.70, size=(total, y_dim))
        metadata["mechanism"] = "shared_deterministic_nonstationary_drift"
        metadata["persistent_identity"] = True
        return _finalize(x, y, metadata, burn_in)

    if world_class == "finite_sample_spurious":
        from .metrics import correlon_score

        best: tuple[float, np.ndarray, np.ndarray] | None = None
        for candidate in range(8):
            candidate_rng = np.random.default_rng(seed_for(seed, world_class, config, offset=10_000 + candidate))
            x = np.zeros((total, x_dim))
            y = np.zeros((total, y_dim))
            ex = candidate_rng.normal(size=x.shape)
            ey = candidate_rng.normal(size=y.shape)
            for index in range(1, total):
                x[index] = 0.97 * x[index - 1] + ex[index]
                y[index] = 0.97 * y[index - 1] + ey[index]
            x_out = _normalize_channels(x[burn_in:])
            y_out = _normalize_channels(y[burn_in:])
            score, _ = correlon_score(
                x_out,
                y_out,
                config["metric"],
                seed_for(seed, "metric_null", config, offset=candidate),
            )
            if best is None or score > best[0]:
                best = (score, x_out, y_out)
        if best is None:
            raise AssertionError("finite-sample null selection produced no candidates")
        metadata["mechanism"] = "max_of_eight_independent_AR097_candidates"
        metadata["selection_score"] = float(best[0])
        return World(best[1], best[2], metadata)

    if world_class == "mixture_without_relational_mechanism":
        x = np.zeros((time_points, x_dim))
        y = np.zeros((time_points, y_dim))
        boundaries = np.linspace(0, time_points, 4, dtype=int)
        for block in range(3):
            start, stop = boundaries[block], boundaries[block + 1]
            factor = _ar_scalar(stop - start, 0.85, rng)
            u = _orthogonal_columns(x_dim, 1, rng)[:, 0]
            v = _orthogonal_columns(y_dim, 1, rng)[:, 0]
            x[start:stop] = 0.90 * factor[:, None] * u + 0.65 * rng.normal(size=(stop - start, x_dim))
            y[start:stop] = 0.90 * factor[:, None] * v + 0.65 * rng.normal(size=(stop - start, y_dim))
        metadata["mechanism"] = "blockwise_changing_common_factor_mixture"
        return World(_normalize_channels(x), _normalize_channels(y), metadata)

    raise ValueError(f"unknown negative world class: {world_class}")


def generate_world(
    world_class: str,
    seed: int,
    config: dict[str, Any],
    edge_scale: float = 1.0,
) -> World:
    data = config["data"]
    time_points = int(data["time_points"])
    x_dim = int(data["x_dim"])
    y_dim = int(data["y_dim"])
    burn_in = int(data["burn_in"])
    rng = np.random.default_rng(seed_for(seed, world_class, config))
    if world_class == "persistent_shared_mode":
        world = _persistent_shared_mode(time_points, x_dim, y_dim, burn_in, rng, edge_scale)
    elif world_class == "direct_coupling_SCM":
        world = _direct_coupling(time_points, x_dim, y_dim, burn_in, rng, edge_scale)
    elif world_class == "history_dependent_relational_mode":
        world = _history_dependent(time_points, x_dim, y_dim, burn_in, rng, edge_scale)
    else:
        world = _negative_world(world_class, time_points, x_dim, y_dim, burn_in, rng, seed, config)
    metadata = dict(world.meta)
    metadata.update({"world_class": world_class, "seed": int(seed), "edge_scale": float(edge_scale)})
    return World(world.x, world.y, metadata)

