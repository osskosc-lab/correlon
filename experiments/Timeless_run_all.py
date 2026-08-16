"""Run the Timeless Correlon program in preregistered order.

The runner preserves an existing confirmation file and will not overwrite it.
Use --force for development/validation reruns only; confirmation remains single-open.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def run_command(args: list[str]) -> None:
    print("$", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def maybe_run(path: Path, command: list[str], force: bool) -> None:
    if path.exists() and not force:
        print(f"preserve existing {path}")
        return
    run_command(command)


def main(force: bool = False) -> None:
    python = sys.executable
    maybe_run(RESULTS / "timeless_T0_raw.csv", [python, "experiments/Timeless_T0_sameC_no_go.py"], force)
    maybe_run(RESULTS / "timeless_T1_raw.csv", [python, "experiments/Timeless_T1_gaussian_gibbs.py"], force)
    maybe_run(RESULTS / "timeless_T2_raw.csv", [python, "experiments/Timeless_T2_modular_flow.py"], force)
    maybe_run(RESULTS / "timeless_T3_development_raw.csv", [python, "experiments/Timeless_T3_relation_state_bridge.py", "--stage", "development"], force)
    maybe_run(RESULTS / "timeless_T3_validation_raw.csv", [python, "experiments/Timeless_T3_relation_state_bridge.py", "--stage", "validation"], force)

    confirmation = RESULTS / "timeless_T3_confirmation_raw.csv"
    if confirmation.exists():
        print(f"preserve existing confirmation (single-open rule): {confirmation}")
    else:
        run_command([python, "experiments/Timeless_T3_relation_state_bridge.py", "--stage", "confirmation"])

    run_command([python, "experiments/Timeless_validate.py"])
    run_command([python, "experiments/Timeless_make_report.py"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="rerun T0-T2 and development/validation; never overwrite confirmation")
    args = parser.parse_args()
    main(force=args.force)
