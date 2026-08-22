"""Command-line entry point for the frozen Correlon Zero experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from .analysis import (
    make_figures,
    run_adversarial_and_finalize,
    run_confirmatory,
    run_pilot,
    write_reports,
)
from .utils import (
    ROOT,
    ensure_output_dirs,
    git_blob,
    implementation_manifest,
    load_config,
    sha256_text_normalized,
    write_json,
)
from .validation import run_independent_validation


DEFAULT_CONFIG = "experiments/correlon_zero/config.json"


def validate_frozen_config(config: dict) -> None:
    if config["protocol"] != "CORRELON_ZERO_ADVERSARIAL_INVARIANCE_FALSIFICATION_v1.0":
        raise AssertionError("unexpected protocol identifier")
    if sha256_text_normalized("CORRELON_ZERO_PREREGISTRATION.md") != config["preregistration_sha256"]:
        raise AssertionError("preregistration hash mismatch")
    if git_blob("experiments/Phase4_correlon_core_persistence.py") != config["phase4_blob"]:
        raise AssertionError("Phase-4 source blob mismatch")
    if len(config["preserve_transformations"]) != 7 or len(config["destroy_transformations"]) != 8:
        raise AssertionError("frozen transformation battery is incomplete")
    if len(config["targets"]) != 3 or len(config["negative_classes"]) != 10:
        raise AssertionError("frozen generator battery is incomplete")
    registered = set(config["namespaces"])
    required = set(config["targets"] + config["negative_classes"])
    required.update(config["preserve_transformations"] + config["destroy_transformations"])
    required.add("metric_null")
    if not required.issubset(registered):
        raise AssertionError(f"missing deterministic namespaces: {sorted(required - registered)}")


def run_tests(config: dict) -> None:
    validate_frozen_config(config)
    exit_code = pytest.main([str(ROOT / "tests" / "test_correlon_zero_transforms.py"), "-q"])
    if exit_code != pytest.ExitCode.OK:
        raise SystemExit(int(exit_code))
    manifest = implementation_manifest(config)
    write_json("results/correlon_zero_implementation_manifest.json", manifest)
    print(f"tests passed; implementation hash={manifest['combined_sha256']}", flush=True)


def run_report_stage(config: dict) -> None:
    figures = make_figures(config)
    validation = run_independent_validation(config, figures)
    chart_map_path = ROOT / config["outputs"]["results_dir"] / "correlon_zero_chart_map.csv"
    if chart_map_path.exists():
        import pandas as pd

        chart_map = pd.read_csv(chart_map_path)
        chart_map["qa"] = "passed independent pixel and finite-value inspection"
        chart_map.to_csv(chart_map_path, index=False)
    reports = write_reports(config, figures)
    print(f"independent validation passed={validation['all_checks_pass']}", flush=True)
    print("reports: " + ", ".join(reports[:3]), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--stage",
        required=True,
        choices=["test", "pilot", "confirmatory", "adversarial", "report", "full"],
    )
    args = parser.parse_args()
    config = load_config(Path(args.config))
    ensure_output_dirs(config)
    validate_frozen_config(config)
    if args.stage == "test":
        run_tests(config)
    elif args.stage == "pilot":
        run_pilot(config)
    elif args.stage == "confirmatory":
        run_confirmatory(config)
    elif args.stage == "adversarial":
        run_adversarial_and_finalize(config)
    elif args.stage == "report":
        run_report_stage(config)
    elif args.stage == "full":
        run_tests(config)
        run_confirmatory(config)
        run_adversarial_and_finalize(config)
        run_report_stage(config)


if __name__ == "__main__":
    main()
