"""Preregistered preserve and destroy transformations."""

from __future__ import annotations

from typing import Any

import numpy as np

from .generators import World, generate_world, phase_randomize_matrix, simulate_separate_var
from .utils import seed_for


def _orthogonal(dimension: int, rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q * signs


def _normalize(values: np.ndarray) -> np.ndarray:
    centered = values - np.mean(values, axis=0, keepdims=True)
    return centered / np.maximum(np.std(centered, axis=0, ddof=1, keepdims=True), 1e-12)


def _iaaft_1d(series: np.ndarray, rng: np.random.Generator, iterations: int) -> np.ndarray:
    target_sorted = np.sort(np.asarray(series, dtype=float))
    target_amplitude = np.abs(np.fft.rfft(series))
    surrogate = rng.permutation(series)
    for _ in range(iterations):
        spectrum = np.fft.rfft(surrogate)
        phases = np.exp(1j * np.angle(spectrum))
        adjusted = np.fft.irfft(target_amplitude * phases, n=len(series))
        order = np.argsort(adjusted, kind="mergesort")
        ranked = np.empty_like(order)
        ranked[order] = np.arange(len(order))
        surrogate = target_sorted[ranked]
    return surrogate


def iaaft_matrix(values: np.ndarray, rng: np.random.Generator, iterations: int) -> np.ndarray:
    output = np.empty_like(values, dtype=float)
    for column in range(values.shape[1]):
        output[:, column] = _iaaft_1d(values[:, column], rng, iterations)
    return output


def _coarse_grain(values: np.ndarray, factor: int) -> np.ndarray:
    usable = len(values) - (len(values) % factor)
    if usable < factor:
        raise ValueError("coarse graining has no complete block")
    return values[:usable].reshape(usable // factor, factor, values.shape[1]).mean(axis=1)


def _time_warp(values: np.ndarray, exponent: float) -> np.ndarray:
    original = np.linspace(0.0, 1.0, len(values))
    sample_at = np.linspace(0.0, 1.0, len(values)) ** exponent
    output = np.empty_like(values, dtype=float)
    for column in range(values.shape[1]):
        output[:, column] = np.interp(sample_at, original, values[:, column])
    return output


def _common_driver_replacement(world: World, rng: np.random.Generator, burn_in: int) -> World:
    x_centered = world.x - np.mean(world.x, axis=0, keepdims=True)
    y_centered = world.y - np.mean(world.y, axis=0, keepdims=True)
    cross = x_centered.T @ y_centered / (len(world.x) - 1)
    u, singular, vt = np.linalg.svd(cross, full_matrices=False)
    x_pc = x_centered @ u[:, 0]
    y_pc = y_centered @ vt.T[:, 0]

    def lag_one(series: np.ndarray) -> float:
        numerator = float(series[:-1] @ series[1:])
        denominator = float(np.sqrt((series[:-1] @ series[:-1]) * (series[1:] @ series[1:])))
        return float(np.clip(numerator / max(denominator, 1e-12), 0.0, 0.98))

    rho = float(np.clip(0.5 * (lag_one(x_pc) + lag_one(y_pc)), 0.0, 0.98))
    total = len(world.x) + burn_in
    latent = np.zeros(total)
    innovations = rng.normal(size=total)
    for index in range(1, total):
        latent[index] = rho * latent[index - 1] + np.sqrt(max(1.0 - rho * rho, 1e-6)) * innovations[index]
    latent = latent[burn_in:]
    latent = (latent - np.mean(latent)) / max(float(np.std(latent, ddof=1)), 1e-12)
    loading = float(np.clip(np.sqrt(max(float(singular[0]), 1e-6)), 0.1, 2.0))
    x = loading * latent[:, None] * u[:, 0] + rng.normal(size=world.x.shape)
    y = loading * latent[:, None] * vt.T[:, 0] + rng.normal(size=world.y.shape)
    return world.with_data(
        _normalize(x),
        _normalize(y),
        transform="D6_common_driver_replacement",
        mechanism="fitted_latent_common_driver",
        direct_edge=False,
        persistent_identity=True,
        oracle_edge_strength=0.0,
        destruction_verified_by_construction=True,
    )


def _block_shuffle(values: np.ndarray, block_length: int, order: np.ndarray) -> np.ndarray:
    blocks = [values[start : min(start + block_length, len(values))] for start in range(0, len(values), block_length)]
    if len(blocks) != len(order):
        raise ValueError("block order does not match block count")
    return np.concatenate([blocks[int(index)] for index in order], axis=0)


def apply_preserve(world: World, name: str, config: dict[str, Any]) -> World:
    if name not in config["preserve_transformations"]:
        raise ValueError(f"unregistered preserve transformation: {name}")
    rng = np.random.default_rng(seed_for(int(world.meta["seed"]), name, config))
    params = config["transform_parameters"]
    if name == "P1_node_permutation":
        px = rng.permutation(world.x.shape[1])
        py = rng.permutation(world.y.shape[1])
        return world.with_data(world.x[:, px], world.y[:, py], transform=name, mechanism_preserved=True)
    if name == "P2_global_scaling":
        low, high = np.log(float(params["scale_low"])), np.log(float(params["scale_high"]))
        sx = np.exp(rng.uniform(low, high, size=world.x.shape[1]))
        sy = np.exp(rng.uniform(low, high, size=world.y.shape[1]))
        return world.with_data(world.x * sx, world.y * sy, transform=name, mechanism_preserved=True)
    if name == "P3_affine_offset":
        scale = float(params["offset_sd"])
        ox = rng.uniform(-scale, scale, size=world.x.shape[1]) * np.std(world.x, axis=0, ddof=1)
        oy = rng.uniform(-scale, scale, size=world.y.shape[1]) * np.std(world.y, axis=0, ddof=1)
        return world.with_data(world.x + ox, world.y + oy, transform=name, mechanism_preserved=True)
    if name == "P4_orthogonal_basis_rotation":
        qx = _orthogonal(world.x.shape[1], rng)
        qy = _orthogonal(world.y.shape[1], rng)
        return world.with_data(world.x @ qx, world.y @ qy, transform=name, mechanism_preserved=True)
    if name == "P5_observation_noise":
        noise = float(params["observation_noise_sd"])
        x = world.x + rng.normal(size=world.x.shape) * np.std(world.x, axis=0, ddof=1) * noise
        y = world.y + rng.normal(size=world.y.shape) * np.std(world.y, axis=0, ddof=1) * noise
        return world.with_data(x, y, transform=name, mechanism_preserved=True)
    if name == "P6_time_reparameterization":
        exponent = float(params["time_warp_exponent"])
        return world.with_data(
            _time_warp(world.x, exponent),
            _time_warp(world.y, exponent),
            transform=name,
            mechanism_preserved=True,
            order_preserved=True,
        )
    if name == "P7_coarse_graining":
        factor = int(params["coarse_grain_factor"])
        return world.with_data(
            _coarse_grain(world.x, factor),
            _coarse_grain(world.y, factor),
            transform=name,
            mechanism_preserved=True,
            coarse_grain_factor=factor,
        )
    raise AssertionError(f"preserve transformation dispatch incomplete: {name}")


def apply_destroy(world: World, name: str, config: dict[str, Any]) -> World:
    if name not in config["destroy_transformations"]:
        raise ValueError(f"unregistered destroy transformation: {name}")
    rng = np.random.default_rng(seed_for(int(world.meta["seed"]), name, config))
    params = config["transform_parameters"]
    if name == "D1_temporal_shuffle":
        return world.with_data(
            world.x[rng.permutation(len(world.x))],
            world.y[rng.permutation(len(world.y))],
            transform=name,
            direct_edge=False,
            persistent_identity=False,
            oracle_edge_strength=0.0,
            destruction_verified_by_construction=True,
        )
    if name == "D2_phase_randomization":
        return world.with_data(
            phase_randomize_matrix(world.x, rng),
            phase_randomize_matrix(world.y, rng),
            transform=name,
            direct_edge=False,
            persistent_identity=False,
            oracle_edge_strength=0.0,
            destruction_verified_by_construction=True,
        )
    if name == "D3_IAAFT_surrogate":
        iterations = int(params["iaaft_iterations"])
        return world.with_data(
            iaaft_matrix(world.x, rng, iterations),
            iaaft_matrix(world.y, rng, iterations),
            transform=name,
            direct_edge=False,
            persistent_identity=False,
            oracle_edge_strength=0.0,
            destruction_verified_by_construction=True,
        )
    if name == "D4_edge_cutting":
        cut = generate_world(str(world.meta["world_class"]), int(world.meta["seed"]), config, edge_scale=0.0)
        return cut.with_data(
            cut.x,
            cut.y,
            transform=name,
            direct_edge=False,
            persistent_identity=False,
            oracle_edge_strength=0.0,
            destruction_verified_by_construction=True,
        )
    if name == "D5_source_swap":
        independent = generate_world(str(world.meta["world_class"]), int(world.meta["seed"]) + 1_000_003, config)
        y = _normalize(independent.y) * np.std(world.y, axis=0, ddof=1) + np.mean(world.y, axis=0)
        return world.with_data(
            world.x.copy(),
            y,
            transform=name,
            direct_edge=False,
            persistent_identity=False,
            oracle_edge_strength=0.0,
            destruction_verified_by_construction=True,
        )
    if name == "D6_common_driver_replacement":
        return _common_driver_replacement(world, rng, int(config["data"]["burn_in"]))
    if name == "D7_history_destruction":
        x, y = simulate_separate_var(
            world.x,
            world.y,
            int(config["data"]["burn_in"]),
            rng,
            np.random.default_rng(seed_for(int(world.meta["seed"]), name, config, offset=1)),
        )
        return world.with_data(
            x,
            y,
            transform=name,
            direct_edge=False,
            persistent_identity=False,
            oracle_edge_strength=0.0,
            destruction_verified_by_construction=True,
        )
    if name == "D8_block_shuffle":
        block_length = int(params["block_length"])
        block_count = int(np.ceil(len(world.x) / block_length))
        order_x = rng.permutation(block_count)
        order_y = rng.permutation(block_count)
        return world.with_data(
            _block_shuffle(world.x, block_length, order_x),
            _block_shuffle(world.y, block_length, order_y),
            transform=name,
            direct_edge=False,
            persistent_identity=False,
            oracle_edge_strength=0.0,
            destruction_verified_by_construction=True,
        )
    raise AssertionError(f"destroy transformation dispatch incomplete: {name}")


def apply_transformation(world: World, name: str, config: dict[str, Any]) -> World:
    if name in config["preserve_transformations"]:
        return apply_preserve(world, name, config)
    if name in config["destroy_transformations"]:
        return apply_destroy(world, name, config)
    raise ValueError(f"unknown transformation: {name}")

