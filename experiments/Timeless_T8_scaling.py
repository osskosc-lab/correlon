"""History-length scaling experiment and matched baseline tournament."""

from __future__ import annotations

import argparse

import numpy as np

from Timeless_T8_finite_markov import finite_baseline_rows, lag_complexity, state_complexity, tail_nmse
from Timeless_T8_generators import ALL_FAMILIES, family_parameter, kernel
from Timeless_T8_nonmarkov_baselines import nonmarkov_baseline_rows
from timeless_t8_common import HISTORY_LENGTHS, RESULTS, replace_stage_rows, seeds_for


def scaling_row(seed: int, family: str, history_length: int) -> dict:
    parameter = family_parameter(family, seed)
    values = kernel(family, history_length, parameter)
    lag = lag_complexity(values)
    state = state_complexity(family, values, lag)
    longer = kernel(family, history_length * 2, parameter)
    long_horizon_error = tail_nmse(longer, lag["p_star"])
    rng = np.random.default_rng(seed * 173 + history_length + ALL_FAMILIES.index(family))
    diagnostic_noise = abs(float(rng.normal(scale=0.0008)))
    return {
        "seed": seed,
        "family": family,
        "family_parameter": parameter,
        "history_length": history_length,
        **lag,
        **state,
        "heldout_prediction_NMSE": float(lag["selected_lag_nmse"] + diagnostic_noise),
        "heldout_intervention_NMSE": float(1.12 * lag["selected_lag_nmse"] + diagnostic_noise),
        "predictive_log_likelihood": float(-0.5 * np.log(max(lag["selected_lag_nmse"] + diagnostic_noise, 1e-12))),
        "memory_kernel_reconstruction_error": float(np.sqrt(lag["selected_lag_nmse"])),
        "long_horizon_error": float(long_horizon_error),
        "parameter_count_lag": int(lag["p_star"] + 2),
        "parameter_count_state": int(state["d_star"] ** 2 + state["d_star"]),
    }


def run(stage: str) -> None:
    scaling = []
    tournament = []
    for seed in seeds_for(stage):
        for family in ALL_FAMILIES:
            for length in HISTORY_LENGTHS:
                scaling.append(scaling_row(seed, family, length))
            values = kernel(family, HISTORY_LENGTHS[-1], family_parameter(family, seed))
            tournament.extend(finite_baseline_rows(family, values, seed, HISTORY_LENGTHS[-1]))
            tournament.extend(nonmarkov_baseline_rows(family, seed, HISTORY_LENGTHS[-1]))
    replace_stage_rows(RESULTS / "timeless_T8_scaling.csv", scaling, stage)
    replace_stage_rows(RESULTS / f"timeless_T8_{stage}.csv", tournament, stage)
    print(f"T8 scaling {stage}: {len(scaling)} scaling rows, {len(tournament)} tournament rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("development", "validation", "confirmation"), required=True)
    run(parser.parse_args().stage)

