"""Shared deterministic I/O and provenance helpers for Correlon Zero."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config["_config_path"] = str(config_path.resolve())
    return config


def ensure_output_dirs(config: dict[str, Any]) -> None:
    for key in ("results_dir", "figures_dir", "cemetery_dir"):
        (ROOT / config["outputs"][key]).mkdir(parents=True, exist_ok=True)


def seed_for(base_seed: int, namespace: str, config: dict[str, Any], offset: int = 0) -> int:
    if namespace not in config["namespaces"]:
        raise KeyError(f"unregistered deterministic namespace: {namespace}")
    modulus = 2**63 - 25
    return int((int(base_seed) + int(config["namespaces"][namespace]) * 1_000_003 + int(offset)) % modulus)


def write_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    temporary.replace(output_path)


def sha256_file(path: str | Path) -> str:
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = ROOT / file_path
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text_normalized(path: str | Path) -> str:
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = ROOT / file_path
    text = file_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_blob(path: str | Path) -> str:
    file_path = Path(path)
    if file_path.is_absolute():
        file_path = file_path.relative_to(ROOT)
    return subprocess.check_output(
        ["git", "hash-object", str(file_path).replace("\\", "/")], cwd=ROOT, text=True
    ).strip()


def git_context() -> dict[str, str]:
    def run(args: list[str]) -> str:
        return subprocess.check_output(args, cwd=ROOT, text=True).strip()

    return {
        "branch": run(["git", "branch", "--show-current"]),
        "commit": run(["git", "rev-parse", "HEAD"]),
    }


def implementation_manifest(config: dict[str, Any]) -> dict[str, Any]:
    files = [
        "CORRELON_ZERO_PREREGISTRATION.md",
        "experiments/correlon_zero/config.json",
        "experiments/correlon_zero/generators.py",
        "experiments/correlon_zero/transforms.py",
        "experiments/correlon_zero/metrics.py",
        "experiments/correlon_zero/baselines.py",
        "experiments/correlon_zero/adversary.py",
        "experiments/correlon_zero/analysis.py",
        "experiments/correlon_zero/validation.py",
        "experiments/correlon_zero/run_correlon_zero.py",
        "tests/test_correlon_zero_transforms.py",
    ]
    hashes = {name: sha256_file(name) for name in files}
    canonical = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "protocol": config["protocol"],
        "version": config["version"],
        "git": git_context(),
        "files": hashes,
        "combined_sha256": hashlib.sha256(canonical).hexdigest(),
    }
