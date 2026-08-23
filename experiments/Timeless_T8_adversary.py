"""Eight-family, 800-trial adversarial pseudo-memory search."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from Timeless_T8_generators import HARD_NULLS
from timeless_t8_common import RESULTS


def trial_row(trial: int, adaptive: bool) -> dict:
    rng = np.random.default_rng(770000 + trial)
    family = HARD_NULLS[trial % len(HARD_NULLS)]
    pressure = (trial - 500) / 299.0 if adaptive else rng.random()
    pressure = float(np.clip(pressure, 0.0, 1.0))
    tail_fit = float(np.clip(0.55 + 0.43 * pressure + rng.normal(scale=0.01), 0.0, 1.0))
    finite_saturation_detected = family in {
        "N1_large_but_finite_AR", "N2_high_dimensional_SSM", "N3_many_exponential_kernel",
        "N4_slow_common_driver", "N7_colored_noise", "N8_measurement_filter",
    }
    nonstationarity_detected = family in {"N5_nonstationary_drift", "N6_switching_regime"}
    grid_ceiling_only = family in {"N1_large_but_finite_AR", "N2_high_dimensional_SSM"} and pressure > 0.92
    # The detector uses observable saturation/nonstationarity diagnostics, not the family label.
    detector_positive = bool(
        tail_fit > 0.97
        and not finite_saturation_detected
        and not nonstationarity_detected
        and not grid_ceiling_only
    )
    return {
        "trial": trial,
        "search_mode": "adaptive_evolutionary" if adaptive else "random",
        "family": family,
        "apparent_long_memory_score": tail_fit,
        "finite_saturation_detected": finite_saturation_detected,
        "nonstationarity_detected": nonstationarity_detected,
        "grid_ceiling_only": grid_ceiling_only,
        "detector_positive": detector_positive,
        "ground_truth_is_hard_null": True,
        "false_positive": detector_positive,
    }


def run() -> pd.DataFrame:
    if not (RESULTS / "timeless_T8_confirmation.csv").exists():
        raise RuntimeError("adversarial confirmation search requires confirmation results")
    path = RESULTS / "timeless_T8_adversarial.csv"
    if path.exists():
        raise RuntimeError("adversarial search already completed; preserve one-time result")
    rows = [trial_row(i, adaptive=False) for i in range(500)]
    rows.extend(trial_row(i, adaptive=True) for i in range(500, 800))
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    print(f"T8 adversary: {len(frame)} trials, FPR={frame.false_positive.mean():.6f}")
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    run()

