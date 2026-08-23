"""Required structural tests for T8."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


EXPERIMENTS = Path(__file__).resolve().parent
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import timeless_t8_common as common
import Timeless_T8_holdout as holdout
from Timeless_T8_finite_markov import lag_complexity
from Timeless_T8_generators import (
    HARD_NULLS, exact_state_dimension, fractional_weights, has_target_history_mechanism,
    kernel, validate_ground_truth,
)
from Timeless_T8_interventions import (
    HELDOUT_INTERVENTIONS, TRAINING_INTERVENTIONS, matched_history_pair, response,
)
from Timeless_T8_scaling import scaling_row


class TimelessT8Tests(unittest.TestCase):
    def test_finite_AR_true_order(self) -> None:
        info = lag_complexity(kernel("M1_AR_finite", 128))
        self.assertEqual(info["p_required_exact"], 3)

    def test_finite_SSM_true_dimension(self) -> None:
        self.assertEqual(exact_state_dimension("M2_linear_SSM"), 4)

    def test_exponential_kernel_finite_augmentation(self) -> None:
        self.assertEqual(exact_state_dimension("M3_sum_exponential_kernel"), 3)

    def test_powerlaw_kernel_tail(self) -> None:
        audit = validate_ground_truth()
        self.assertTrue(audit["checks"]["powerlaw_tail"])

    def test_fractional_long_memory(self) -> None:
        weights = fractional_weights(4096, 0.35)
        slope = np.polyfit(np.log(np.arange(512, 4096)), np.log(weights[512:]), 1)[0]
        self.assertAlmostEqual(float(slope), -0.65, delta=0.03)

    def test_current_state_matching(self) -> None:
        h1, h2 = matched_history_pair(9)
        self.assertEqual(h1[-1], h2[-1])

    def test_history_swap_changes_only_history(self) -> None:
        h1, h2 = matched_history_pair(10)
        self.assertEqual(h1[-1], h2[-1])
        self.assertGreater(np.linalg.norm(h1[:-1] - h2[:-1]), 1.0)
        intervention = ("pulse", 1.0, 0)
        self.assertGreater(abs(response(h1, intervention, True) - response(h2, intervention, True)), 0.15)

    def test_holdout_interventions_not_in_training(self) -> None:
        self.assertTrue(set(HELDOUT_INTERVENTIONS).isdisjoint(set(TRAINING_INTERVENTIONS)))

    def test_hard_null_no_target_mechanism(self) -> None:
        self.assertFalse(any(has_target_history_mechanism(name) for name in HARD_NULLS))

    def test_confirmation_requires_freeze(self) -> None:
        old = (common.CONFIG_PATH, common.FREEZE_PATH)
        try:
            with tempfile.TemporaryDirectory() as directory:
                common.CONFIG_PATH = Path(directory) / "missing-config.json"
                common.FREEZE_PATH = Path(directory) / "missing-freeze.json"
                with self.assertRaises(RuntimeError):
                    common.assert_freeze()
        finally:
            common.CONFIG_PATH, common.FREEZE_PATH = old

    def test_holdout_locked_before_confirmation(self) -> None:
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
        self.assertEqual(scaling_row(30007, "L1_power_law_kernel", 128),
                         scaling_row(30007, "L1_power_law_kernel", 128))


if __name__ == "__main__":
    unittest.main()

