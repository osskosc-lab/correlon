"""Stage-ordered T8 runner with freeze and single-open confirmation guards."""

from __future__ import annotations

import argparse
import subprocess
import sys

from timeless_t8_common import (
    CONFIRMATION_MARKER, CONFIRMATION_PATH, RESULTS, ROOT, assert_freeze,
)


def command(*args: str) -> None:
    print("$", " ".join(args), flush=True)
    subprocess.run(list(args), cwd=ROOT, check=True)


def run(stage: str) -> None:
    python = sys.executable
    if stage == "development":
        command(python, "-m", "unittest", "experiments/test_timeless_T8.py", "-v")
    elif stage == "confirmation":
        assert_freeze()
        if CONFIRMATION_MARKER.exists() or CONFIRMATION_PATH.exists() or (RESULTS / "timeless_T8_adversarial.csv").exists():
            raise RuntimeError("T8 confirmation has already been opened")
        CONFIRMATION_MARKER.write_text("opened\n", encoding="utf-8")

    command(python, "experiments/Timeless_T8_scaling.py", "--stage", stage)
    command(python, "experiments/Timeless_T8_interventions.py", "--stage", stage)

    if stage == "validation":
        command(python, "experiments/Timeless_T8_validate.py", "--stage", "freeze")
    elif stage == "confirmation":
        command(python, "experiments/Timeless_T8_adversary.py")
        command(python, "experiments/Timeless_T8_validate.py", "--stage", "confirmation")
    print("T8", stage, "complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("development", "validation", "confirmation"), required=True)
    run(parser.parse_args().stage)

