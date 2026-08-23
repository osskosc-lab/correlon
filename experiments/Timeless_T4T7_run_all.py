"""Run T4-T7 in preregistered stage order with one-time confirmation guards."""

from __future__ import annotations

import argparse
import subprocess
import sys

from timeless_t4t7_common import (
    CONFIRMATION_MARKER,
    CONFIRMATION_PATH,
    RESULTS,
    ROOT,
    assert_confirmation_freeze,
)


PHASES = (
    "experiments/Timeless_T4_full_equal_time_no_go.py",
    "experiments/Timeless_T5_order_identifiability.py",
    "experiments/Timeless_T6_duration_identifiability.py",
    "experiments/Timeless_T7_arrow_identifiability.py",
)


def command(*args: str) -> None:
    print("$", " ".join(args), flush=True)
    subprocess.run(list(args), cwd=ROOT, check=True)


def run(stage: str) -> None:
    python = sys.executable
    if stage == "development":
        command(python, "-m", "unittest", "experiments/test_timeless_T4T7.py", "-v")
    elif stage == "confirmation":
        assert_confirmation_freeze()
        if CONFIRMATION_PATH.exists() or CONFIRMATION_MARKER.exists():
            raise RuntimeError("confirmation has already been opened; single-open rule enforced")
        CONFIRMATION_MARKER.write_text("opened\n", encoding="utf-8")

    for phase in PHASES:
        command(python, phase, "--stage", stage)

    if stage == "validation":
        command(python, "experiments/Timeless_T4T7_leakage_audit.py", "--stage", "validation")
        command(python, "experiments/Timeless_T4T7_validate.py", "--stage", "freeze")
    elif stage == "confirmation":
        command(python, "experiments/Timeless_T4T7_validate.py", "--stage", "confirmation")

    print(f"T4-T7 {stage} complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("development", "validation", "confirmation"), required=True)
    run(parser.parse_args().stage)

