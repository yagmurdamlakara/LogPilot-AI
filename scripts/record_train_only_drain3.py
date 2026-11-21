"""Create a reproducibility manifest for completed train-only Drain3 runs.

Run from the repository root after pulling the latest main:
    python scripts/record_train_only_drain3.py

This script only reads existing experiment outputs and local environment metadata.
It does not parse HDFS.log, rebuild features, or train models.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "experiments" / "train_only_drain3" / "summary.json"
AUDIT = ROOT / "reports" / "model_evaluation" / "train_only_drain3_split_audit.json"
CONFIG = ROOT / "configs" / "drain3_train_only.ini"
DATASET = ROOT / "data" / "processed" / "hdfs_full_labeled.csv"
OUTPUT = ROOT / "reports" / "model_evaluation" / "train_only_drain3_manifest.json"

PACKAGES = (
    "drain3",
    "pandas",
    "scikit-learn",
    "numpy",
    "joblib",
    "scipy",
)


def command_output(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            args, cwd=ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def main() -> int:
    missing = [path for path in (SUMMARY, AUDIT, CONFIG, DATASET) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required input files are missing: " + ", ".join(str(path) for path in missing)
        )

    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS":
        raise RuntimeError(
            f"Split audit must be PASS before manifest creation; got {audit.get('status')}"
        )

    seed_records = []
    for seed_result in summary["seeds"]:
        seed = int(seed_result["seed"])
        seed_dir = SUMMARY.parent / f"seed_{seed}"
        required = (
            "metrics.json",
            "test_predictions.csv",
            "baseline_models.joblib",
            "drain3_templates.json",
            "experiment_report.json",
        )
        missing_artifacts = [name for name in required if not (seed_dir / name).is_file()]
        if missing_artifacts:
            raise FileNotFoundError(
                f"Seed {seed} is missing artifacts: {', '.join(missing_artifacts)}"
            )
        seed_records.append(
            {
                "seed": seed,
                "train_blocks": seed_result["train_blocks"],
                "validation_blocks": seed_result["validation_blocks"],
                "test_blocks": seed_result["test_blocks"],
                "trained_drain_clusters": seed_result["parser_fit"]["trained_clusters"],
                "matched_log_lines": seed_result["frozen_feature_build"]["matched_log_lines"],
                "unmatched_log_lines": seed_result["frozen_feature_build"]["unmatched_log_lines"],
                "unmatched_fraction": seed_result["unmatched_fraction"],
                "metrics": seed_result["metrics"],
                "artifacts": list(required),
            }
        )

    dataset_stat = DATASET.stat()
    config_text = CONFIG.read_text(encoding="utf-8")
    manifest = {
        "experiment": "train_only_drain3_frozen_inference",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_control": {
            "repository": "yagmurdamlakara/LogPilot-AI",
            "branch": command_output("git", "branch", "--show-current"),
            "commit": command_output("git", "rev-parse", "HEAD"),
            "working_tree_status": command_output("git", "status", "--short"),
        },
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": package_versions(),
        },
        "dataset": {
            "path": str(DATASET.relative_to(ROOT)),
            "rows": int(audit["rows"]),
            "size_bytes": dataset_stat.st_size,
            "modified_time": datetime.fromtimestamp(
                dataset_stat.st_mtime, timezone.utc
            ).isoformat(),
        },
        "split_audit": {
            "status": audit["status"],
            "seed_count": len(audit["seeds"]),
            "all_seed_statuses": {
                str(item["seed"]): item["status"] for item in audit["seeds"]
            },
            "report_path": str(AUDIT.relative_to(ROOT)),
        },
        "drain3_configuration": {
            "historical_run_note": (
                "The completed five-seed run was executed with TemplateMiner() "
                "and no drain3.ini in the working directory. Drain3 0.9.11 therefore "
                "used TemplateMinerConfig defaults. The committed INI below explicitly "
                "records those defaults for future runs; it was not loaded by the "
                "historical run itself."
            ),
            "future_run_config_path": str(CONFIG.relative_to(ROOT)),
            "future_run_config_sha256": sha256(CONFIG),
            "future_run_config_text": config_text,
        },
        "experiment_settings": {
            "seeds": [item["seed"] for item in seed_records],
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
            "The historical run's effective defaults are documented; the explicit INI is for future reproducible runs.",
        ],
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Manifest saved: {OUTPUT}")
    print(f"Git commit: {manifest['source_control']['commit']}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Drain3: {manifest['environment']['packages']['drain3']}")
    print(f"Split audit: {manifest['split_audit']['status']}")
    print(f"Seeds recorded: {len(seed_records)}")
    print("No raw-log processing or model training was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
