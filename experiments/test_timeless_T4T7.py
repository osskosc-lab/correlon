"""Structural and numerical tests required by the T4-T7 preregistration."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


EXPERIMENTS = Path(__file__).resolve().parent
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import timeless_t4t7_common as common
import Timeless_T4T7_holdout as holdout
from Timeless_T4_full_equal_time_no_go import gaussian_population_moments_through_6, run_seed as run_t4
from Timeless_T5_order_identifiability import make_world, run_case as run_t5
from Timeless_T6_duration_identifiability import run_case as run_t6
from Timeless_T7_arrow_identifiability import (
    forward_chain,
    probability_current,
    reverse_chain,
    run_seed as run_t7,
    symmetric_features,
)


class TimelessT4T7Tests(unittest.TestCase):
    def test_no_timestamp_feature_leak(self) -> None:
        self.assertFalse(run_t4(3)[0]["input_has_timestamp"])
        self.assertFalse(run_t5(3, "linear_DAG", 8)["input_has_timestamp"])
        self.assertTrue(run_t6(3, "stable_linear_flow")["timestamps_hidden"])
        self.assertTrue(run_t7(3)["timestamps_hidden"])

    def test_no_storage_order_leak(self) -> None:
        rng = np.random.default_rng(8)
        matrix = rng.normal(size=(6, 6))
        order = rng.permutation(6)
        restored = matrix[order][:, order][np.argsort(order)][:, np.argsort(order)]
        self.assertTrue(np.allclose(matrix, restored))

    def test_equal_time_distribution_match(self) -> None:
        moments = gaussian_population_moments_through_6()
        self.assertEqual(len(moments), 27)
        rows = run_t4(11)
        self.assertTrue(all(row["analytic_equal_time_moment_max_difference_order_1_to_6"] == 0 for row in rows))

    def test_generator_difference_ground_truth(self) -> None:
        self.assertTrue(all(row["generator_difference_norm"] > 0 for row in run_t4(12)))

    def test_cycle_has_no_forced_DAG_order(self) -> None:
        row = run_t5(4, "directed_cycle", 8)
        self.assertTrue(row["cycle_detected"])
        self.assertFalse(row["false_total_order"])

    def test_duration_labels_hidden(self) -> None:
        row = run_t6(5, "stable_linear_flow")
        self.assertTrue(row["duration_labels_hidden"])
        self.assertNotIn("true_durations", row)

    def test_periodic_aliasing_ground_truth(self) -> None:
        row = run_t6(7, "periodic_rotation")
        self.assertTrue(row["periodic_alias_ground_truth"])

    def test_forward_reverse_matching(self) -> None:
        forward, stationary = forward_chain(17)
        reverse = reverse_chain(forward, stationary)
        self.assertTrue(np.allclose(symmetric_features(forward), symmetric_features(reverse)))

    def test_probability_current_sign_reversal(self) -> None:
        forward, stationary = forward_chain(18)
        reverse = reverse_chain(forward, stationary)
        self.assertTrue(np.allclose(probability_current(forward, stationary), -probability_current(reverse, stationary)))

    def test_confirmation_requires_freeze(self) -> None:
        old = (common.CONFIG_PATH, common.VALIDATION_PATH)
        try:
            with tempfile.TemporaryDirectory() as directory:
                common.CONFIG_PATH = Path(directory) / "missing-config.json"
                common.VALIDATION_PATH = Path(directory) / "missing-validation.json"
                with self.assertRaises(RuntimeError):
                    common.assert_confirmation_freeze()
        finally:
            common.CONFIG_PATH, common.VALIDATION_PATH = old

    def test_holdout_cannot_run_before_confirmation(self) -> None:
        old = (holdout.HOLDOUT_PATH, holdout.CONFIRMATION_PATH)
        try:
            with tempfile.TemporaryDirectory() as directory:
                holdout.HOLDOUT_PATH = Path(directory) / "holdout.json"
                holdout.CONFIRMATION_PATH = Path(directory) / "confirmation.json"
                with self.assertRaises(RuntimeError):
                    holdout.run()
        finally:
            holdout.HOLDOUT_PATH, holdout.CONFIRMATION_PATH = old

    def test_seed_reproducibility(self) -> None:
        a = run_t5(29, "nonlinear_DAG", 16)
        b = run_t5(29, "nonlinear_DAG", 16)
        self.assertEqual(json.dumps(a, sort_keys=True, allow_nan=True), json.dumps(b, sort_keys=True, allow_nan=True))


if __name__ == "__main__":
    unittest.main()

