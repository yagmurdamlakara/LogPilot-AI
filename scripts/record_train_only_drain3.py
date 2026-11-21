"""Create a reproducibility manifest for completed train-only Drain3 runs.

Run from the repository root after pulling the latest main:
    python scripts/record_train_only_drain3.py [--output PATH]

This script only reads existing experiment outputs and local environment metadata.
It does not parse HDFS.log, rebuild features, or train models. Existing output
files are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "experiments" / "train_only_drain3" / "summary.json"
AUDIT = ROOT / "reports" / "model_evaluation" / "train_only_drain3_split_audit.json"
CONFIG = ROOT / "configs" / "drain3_train_only.ini"
DATASET = ROOT / "data" / "processed" / "hdfs_full_labeled.csv"
OUTPUT = ROOT / "reports" / "model_evaluation" / "train_only_drain3_manifest.json"

EXPECTED_SEEDS = (11, 22, 33, 44, 55)
REQUIRED_ARTIFACTS = (
    "metrics.json",
    "test_predictions.csv",
    "baseline_models.joblib",
    "drain3_templates.json",
    "experiment_report.json",
)
PACKAGES = ("drain3", "pandas", "scikit-learn", "numpy", "joblib", "scipy")
CHUNK_SIZE = 1024 * 1024


def command_output(*args: str, allow_empty: bool = False) -> str:
    """Return Git metadata or fail explicitly when it cannot be recorded."""
    try:
        result = subprocess.check_output(
            args, cwd=ROOT, stderr=subprocess.PIPE, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Could not capture source-control metadata: {args!r}") from exc
    if not result and not allow_empty:
        raise RuntimeError(f"Source-control command returned empty output: {args!r}")
    return result


def file_record(path: Path) -> dict[str, Any]:
    """Return relative path, byte size and SHA-256 from one streaming read."""
    if not path.is_file():
        raise FileNotFoundError(f"Required file is missing: {path}")

    digest = hashlib.sha256()
    size_bytes = 0
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
                size_bytes += len(chunk)
                digest.update(chunk)
    except OSError as exc:
        raise OSError(f"Could not read required file {path}: {exc}") from exc

    try:
        relative_path = path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Manifest input must be inside repository root: {path}") from exc
    return {"path": relative_path, "size_bytes": size_bytes, "sha256": digest.hexdigest()}


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def validate_seed_entries(entries: Any, *, source: str) -> dict[int, dict[str, Any]]:
    if not isinstance(entries, list):
        raise ValueError(f"{source} must contain a seeds list.")

    indexed: dict[int, dict[str, Any]] = {}
    for position, item in enumerate(entries):
        if not isinstance(item, dict):
            raise ValueError(f"{source} seed entry {position} must be an object.")
        seed = item.get("seed")
        if type(seed) is not int:
            raise ValueError(f"{source} seed entry {position} must have an integer seed.")
        if seed not in EXPECTED_SEEDS:
            raise ValueError(f"{source} contains unexpected seed {seed}.")
        if seed in indexed:
            raise ValueError(f"{source} contains duplicate seed {seed}.")
        indexed[seed] = item

    missing = sorted(set(EXPECTED_SEEDS) - indexed.keys())
    if missing:
        raise ValueError(f"{source} is missing expected seeds: {missing}.")
    return indexed


def validate_summary_seed(item: dict[str, Any], seed: int) -> None:
    integer_fields = ("train_blocks", "validation_blocks", "test_blocks")
    for field in integer_fields:
        if type(item.get(field)) is not int:
            raise ValueError(f"Summary seed {seed} field {field!r} must be an integer.")

    parser_fit = item.get("parser_fit")
    if not isinstance(parser_fit, dict) or type(parser_fit.get("trained_clusters")) is not int:
        raise ValueError(f"Summary seed {seed} parser_fit.trained_clusters must be an integer.")

    feature_build = item.get("frozen_feature_build")
    if not isinstance(feature_build, dict):
        raise ValueError(f"Summary seed {seed} frozen_feature_build must be an object.")
    for field in ("matched_log_lines", "unmatched_log_lines"):
        if type(feature_build.get(field)) is not int:
            raise ValueError(
                f"Summary seed {seed} frozen_feature_build.{field} must be an integer."
            )

    unmatched_fraction = item.get("unmatched_fraction")
    if type(unmatched_fraction) not in (int, float) or not math.isfinite(unmatched_fraction):
        raise ValueError(f"Summary seed {seed} unmatched_fraction must be a finite number.")
    if not isinstance(item.get("metrics"), dict):
        raise ValueError(f"Summary seed {seed} metrics must be an object.")


def build_manifest() -> dict[str, Any]:
    for path in (SUMMARY, AUDIT, CONFIG, DATASET):
        if not path.is_file():
            raise FileNotFoundError(f"Required input file is missing: {path}")

    summary = load_json(SUMMARY)
    audit = load_json(AUDIT)
    if type(audit.get("status")) is not str:
        raise ValueError("Split audit status must be a string.")
    if audit.get("status") != "PASS":
        raise RuntimeError(
            f"Split audit must be PASS before manifest creation; got {audit.get('status')}"
        )

    if type(audit.get("rows")) is not int:
        raise ValueError("Split audit rows must be an integer.")
    summary_seeds = validate_seed_entries(summary.get("seeds"), source="Experiment summary")
    audit_seeds = validate_seed_entries(audit.get("seeds"), source="Split audit")
    for seed in EXPECTED_SEEDS:
        status = audit_seeds[seed].get("status")
        if type(status) is not str:
            raise ValueError(f"Split audit seed {seed} status must be a string.")
    if any(audit_seeds[seed]["status"] != "PASS" for seed in EXPECTED_SEEDS):
        raise RuntimeError("Every seed in the split audit must have status PASS.")

    seed_records = []
    for seed in sorted(EXPECTED_SEEDS):
        seed_result = summary_seeds[seed]
        validate_summary_seed(seed_result, seed)
        seed_dir = SUMMARY.parent / f"seed_{seed}"
        artifacts = [file_record(seed_dir / name) for name in REQUIRED_ARTIFACTS]
        seed_record = {
            "seed": seed,
            "train_blocks": seed_result["train_blocks"],
            "validation_blocks": seed_result["validation_blocks"],
            "test_blocks": seed_result["test_blocks"],
            "trained_drain_clusters": seed_result["parser_fit"]["trained_clusters"],
            "matched_log_lines": seed_result["frozen_feature_build"]["matched_log_lines"],
            "unmatched_log_lines": seed_result["frozen_feature_build"]["unmatched_log_lines"],
            "unmatched_fraction": seed_result["unmatched_fraction"],
            "metrics": seed_result["metrics"],
            "artifacts": artifacts,
        }
        seed_records.append(seed_record)

    config_text = CONFIG.read_text(encoding="utf-8")
    return {
        "schema_version": 2,
        "experiment": "train_only_drain3_frozen_inference",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_control": {
            "repository": "yagmurdamlakara/LogPilot-AI",
            "branch": command_output("git", "branch", "--show-current"),
            "commit": command_output("git", "rev-parse", "HEAD"),
            "working_tree_status": command_output(
                "git", "status", "--short", allow_empty=True
            ),
        },
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": package_versions(),
        },
        "dataset": {
            **file_record(DATASET),
            "rows": int(audit["rows"]),
            "modified_time": datetime.fromtimestamp(
                DATASET.stat().st_mtime, timezone.utc
            ).isoformat(),
            "identity_scope": "hash_calculated_at_manifest_recording_time",
        },
        "split_audit": {
            "status": audit["status"],
            "seed_count": len(audit_seeds),
            "all_seed_statuses": {
                str(seed): audit_seeds[seed]["status"] for seed in sorted(EXPECTED_SEEDS)
            },
            "report_path": AUDIT.relative_to(ROOT).as_posix(),
        },
        "drain3_configuration": {
            "historical_run": {
                "drain3_version": "0.9.11",
                "config_source": "TemplateMinerConfig defaults",
                "ini_loaded": False,
                "effective_defaults": {
                    "profiling_enabled": False,
                    "profiling_report_sec": 60,
                    "snapshot_interval_minutes": 5,
                    "snapshot_compressed": True,
                    "drain_engine": "Drain",
                    "similarity_threshold": 0.4,
                    "depth": 4,
                    "max_children": 100,
                    "max_clusters": None,
                    "extra_delimiters": [],
                    "parametrize_numeric_tokens": True,
                    "masking_rules": [],
                    "mask_prefix": "<",
                    "mask_suffix": ">",
                    "parameter_extraction_cache_capacity": 3000,
                },
                "note": (
                    "The completed historical run constructed TemplateMiner() without "
                    "a drain3.ini, so Drain3 0.9.11 defaults applied."
                ),
            },
            "future_run": {
                **file_record(CONFIG),
                "text": config_text,
            },
        },
        "experiment_settings": {
            "seeds": list(EXPECTED_SEEDS),
            "train_fraction": 0.6,
            "validation_fraction": 0.2,
            "test_fraction": 0.2,
            "split_strategy": "random stratified block-level split",
            "parser_fit_scope": "training-block log lines only",
            "validation_test_parser_behavior": "frozen TemplateMiner.match; no cluster updates",
            "unmatched_message_handling": "__UNMATCHED__ event feature",
            "standard_deviation": "sample standard deviation (ddof=1)",
        },
        "seed_results": seed_records,
        "limitations": [
            "The split is random and block-level, not chronological.",
            "This audit does not establish host-level independence.",
            "These hashes identify bytes read when this manifest was recorded; they do not prove the byte identity of files at the historical experiment run time.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT,
        help="Manifest output path (existing files are never overwritten).",
    )
    args = parser.parse_args(argv)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output = output.resolve()
    try:
        output.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"Manifest output must stay inside repository root: {output}") from exc
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing manifest: {output}. Choose a new --output path."
        )

    manifest = build_manifest()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8", newline="\n") as target:
            json.dump(manifest, target, indent=2, ensure_ascii=False)
            target.write("\n")
    except FileExistsError as exc:
        raise FileExistsError(f"Refusing to overwrite existing manifest: {output}") from exc

    print(f"Manifest saved: {output}")
    print(f"Git commit: {manifest['source_control']['commit']}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Drain3: {manifest['environment']['packages']['drain3']}")
    print(f"Split audit: {manifest['split_audit']['status']}")
    print(f"Seeds recorded: {len(manifest['seed_results'])}")
    print("No raw-log processing or model training was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
