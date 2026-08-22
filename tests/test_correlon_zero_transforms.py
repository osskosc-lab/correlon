from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from experiments.correlon_zero.baselines import baseline_scores, var_arx_predictive
from experiments.correlon_zero.generators import generate_world
from experiments.correlon_zero.metrics import correlon_components, correlon_score
from experiments.correlon_zero.transforms import apply_destroy, apply_preserve
from experiments.correlon_zero.utils import ROOT, git_blob, load_config, sha256_text_normalized


CONFIG = load_config("experiments/correlon_zero/config.json")


def test_frozen_source_hashes_match_preregistration() -> None:
    assert sha256_text_normalized("CORRELON_ZERO_PREREGISTRATION.md") == CONFIG["preregistration_sha256"]
    assert git_blob("experiments/Phase4_correlon_core_persistence.py") == CONFIG["phase4_blob"]


def test_phase4_readouts_are_reproduced_exactly() -> None:
    path = ROOT / "experiments" / "Phase4_correlon_core_persistence.py"
    specification = importlib.util.spec_from_file_location("legacy_phase4", path)
    assert specification is not None and specification.loader is not None
    legacy = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(legacy)
    seed = 17
    x, y = legacy.make_data("rank1_persistent", seed, T=600, d=6, q=6)
    observed = legacy.metrics(legacy.modes(x, y, win=140, step=40), max_lag=3)
    null = legacy.null_medians(x, y, seed, nnull=6)
    current = correlon_components(x, y, CONFIG["metric"], null_seed=seed + 7777)
    for key in ("strength", "gap", "rel_gap", "persistence_mean", "persistence_q10", "persistence_min"):
        assert np.isclose(current[f"observed_{key}"], observed[key], rtol=0.0, atol=1e-12)
        assert np.isclose(current[f"null_{key}"], null[key], rtol=0.0, atol=1e-12)
    assert np.isclose(current["T_iso"], observed["gap"] - null["gap"], atol=1e-12)
    assert np.isclose(current["T_floor"], observed["persistence_q10"] - null["persistence_q10"], atol=1e-12)


def test_all_target_generators_contain_the_frozen_oracle_mechanism() -> None:
    for target in CONFIG["targets"]:
        world = generate_world(target, 11, CONFIG)
        assert world.meta["direct_edge"] is True
        assert world.meta["persistent_identity"] is True
        assert world.meta["oracle_edge_strength"] > 0.0
        assert world.x.shape == (600, 6)
        assert world.y.shape == (600, 6)
        predictive = var_arx_predictive(world.x, world.y, lags=3, ridge=1e-3)
        assert predictive > 1e-3


def test_preserve_transforms_preserve_oracle_ground_truth() -> None:
    for target in CONFIG["targets"]:
        world = generate_world(target, 19, CONFIG)
        for name in CONFIG["preserve_transformations"]:
            transformed = apply_preserve(world, name, CONFIG)
            assert transformed.meta["direct_edge"] is True
            assert transformed.meta["persistent_identity"] is True
            assert transformed.meta["oracle_edge_strength"] == world.meta["oracle_edge_strength"]
            assert transformed.meta["mechanism_preserved"] is True
        coarse = apply_preserve(world, "P7_coarse_graining", CONFIG)
        assert len(coarse.x) == len(world.x) // 2
        warped = apply_preserve(world, "P6_time_reparameterization", CONFIG)
        assert warped.meta["order_preserved"] is True


def test_invertible_representation_transforms_preserve_frozen_score() -> None:
    world = generate_world("direct_coupling_SCM", 23, CONFIG)
    null_seed = 987654
    original, _ = correlon_score(world.x, world.y, CONFIG["metric"], null_seed)
    tolerances = {
        "P1_node_permutation": 1e-10,
        "P2_global_scaling": 2e-3,
        "P3_affine_offset": 1e-10,
        "P4_orthogonal_basis_rotation": 1e-10,
    }
    for name, tolerance in tolerances.items():
        transformed = apply_preserve(world, name, CONFIG)
        score, _ = correlon_score(transformed.x, transformed.y, CONFIG["metric"], null_seed)
        assert abs(score - original) <= tolerance


def test_destroy_transforms_remove_oracle_direct_edge() -> None:
    for target in CONFIG["targets"]:
        world = generate_world(target, 29, CONFIG)
        for name in CONFIG["destroy_transformations"]:
            transformed = apply_destroy(world, name, CONFIG)
            assert transformed.meta["direct_edge"] is False
            assert transformed.meta["persistent_identity"] is False or name == "D6_common_driver_replacement"
            assert transformed.meta["oracle_edge_strength"] == 0.0
            assert transformed.meta["destruction_verified_by_construction"] is True


def test_destroy_operations_preserve_only_their_declared_low_level_statistics() -> None:
    world = generate_world("direct_coupling_SCM", 31, CONFIG)
    shuffled = apply_destroy(world, "D1_temporal_shuffle", CONFIG)
    assert np.allclose(np.sort(shuffled.x, axis=0), np.sort(world.x, axis=0))
    assert np.allclose(np.sort(shuffled.y, axis=0), np.sort(world.y, axis=0))

    randomized = apply_destroy(world, "D2_phase_randomization", CONFIG)
    original_amplitude = np.abs(np.fft.rfft(world.x, axis=0))
    randomized_amplitude = np.abs(np.fft.rfft(randomized.x, axis=0))
    assert np.allclose(original_amplitude, randomized_amplitude, rtol=1e-8, atol=1e-8)

    surrogate = apply_destroy(world, "D3_IAAFT_surrogate", CONFIG)
    assert np.allclose(np.sort(surrogate.x, axis=0), np.sort(world.x, axis=0), atol=1e-12)
    spectrum_error = np.linalg.norm(
        np.abs(np.fft.rfft(surrogate.x, axis=0)) - original_amplitude
    ) / np.linalg.norm(original_amplitude)
    assert spectrum_error < 0.20

    swapped = apply_destroy(world, "D5_source_swap", CONFIG)
    assert np.array_equal(swapped.x, world.x)
    assert not np.array_equal(swapped.y, world.y)

    blocked = apply_destroy(world, "D8_block_shuffle", CONFIG)
    assert np.allclose(np.sort(blocked.x, axis=0), np.sort(world.x, axis=0))
    assert np.allclose(np.sort(blocked.y, axis=0), np.sort(world.y, axis=0))


def test_destroy_battery_reduces_cross_predictive_signal_as_a_group() -> None:
    world = generate_world("direct_coupling_SCM", 37, CONFIG)
    original = var_arx_predictive(world.x, world.y, lags=3, ridge=1e-3)
    destructive = [
        "D1_temporal_shuffle",
        "D2_phase_randomization",
        "D3_IAAFT_surrogate",
        "D4_edge_cutting",
        "D5_source_swap",
        "D7_history_destruction",
        "D8_block_shuffle",
    ]
    values = [
        var_arx_predictive(
            apply_destroy(world, name, CONFIG).x,
            apply_destroy(world, name, CONFIG).y,
            lags=3,
            ridge=1e-3,
        )
        for name in destructive
    ]
    assert float(np.mean(values)) < original


def test_every_negative_class_has_no_direct_edge() -> None:
    for negative_class in CONFIG["negative_classes"]:
        world = generate_world(negative_class, 41, CONFIG)
        assert world.meta["direct_edge"] is False
        assert world.meta["oracle_edge_strength"] == 0.0
        assert np.all(np.isfinite(world.x))
        assert np.all(np.isfinite(world.y))


def test_all_baselines_return_finite_nonnegative_scores() -> None:
    world = generate_world("history_dependent_relational_mode", 43, CONFIG)
    scores = baseline_scores(world.x, world.y, CONFIG)
    assert set(scores) == set(CONFIG["baselines"])
    assert all(np.isfinite(value) and value >= 0.0 for value in scores.values())
