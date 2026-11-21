"""Audit train/validation/test separation for the train-only Drain3 experiment.

Run from the repository root:
    python scripts/audit_train_only_drain3.py

This reconstructs the stratified block split used by run_train_only_drain3.py,
checks pairwise block_id overlap, compares saved test prediction IDs and labels
with the reconstructed test split, and validates per-seed experiment reports.
It does not test temporal/host independence or prove absence of all leakage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "processed" / "hdfs_full_labeled.csv"
DEFAULT_EXPERIMENTS = ROOT / "experiments" / "train_only_drain3"
DEFAULT_SEEDS = (11, 22, 33, 44, 55)


def reconstruct_splits(
    frame: pd.DataFrame, seed: int, test_size: float, validation_size: float
) -> dict[str, pd.Index]:
    train_val_idx, test_idx = train_test_split(
        frame.index,
        test_size=test_size,
        random_state=seed,
        stratify=frame["label"],
    )
    relative_validation_size = validation_size / (1.0 - test_size)
    train_idx, validation_idx = train_test_split(
        train_val_idx,
        test_size=relative_validation_size,
        random_state=seed,
        stratify=frame.loc[train_val_idx, "label"],
    )
    return {"train": train_idx, "validation": validation_idx, "test": test_idx}


def audit_seed(
    frame: pd.DataFrame,
    seed: int,
    experiments_dir: Path,
    test_size: float,
    validation_size: float,
) -> dict[str, object]:
    split = reconstruct_splits(frame, seed, test_size, validation_size)
    ids = {
        name: set(frame.loc[index, "block_id"].astype(str))
        for name, index in split.items()
    }
    overlaps = {
        "train_validation": len(ids["train"] & ids["validation"]),
        "train_test": len(ids["train"] & ids["test"]),
        "validation_test": len(ids["validation"] & ids["test"]),
    }
    seed_dir = experiments_dir / f"seed_{seed}"
    prediction_path = seed_dir / "test_predictions.csv"
    report_path = seed_dir / "experiment_report.json"
    result: dict[str, object] = {
        "seed": seed,
        "split_sizes": {name: len(index) for name, index in split.items()},
        "block_id_overlaps": overlaps,
        "prediction_id_symmetric_difference": None,
        "prediction_actual_label_mismatches": None,
        "report_size_mismatches": [],
        "errors": [],
        "warnings": [],
    }

    if any(overlaps.values()):
        result["errors"].append("Block IDs overlap between splits.")

    if not prediction_path.is_file():
        result["warnings"].append(f"Prediction file not found: {prediction_path}")
    else:
        predictions = pd.read_csv(
            prediction_path, dtype={"block_id": "string"}
        )
        required = {"block_id", "actual_label"}
        if not required.issubset(predictions.columns):
            result["errors"].append(
                "test_predictions.csv must contain block_id and actual_label."
            )
        else:
            prediction_ids = predictions["block_id"].dropna().astype(str)
            result["prediction_rows"] = int(len(predictions))
            result["prediction_unique_block_ids"] = int(prediction_ids.nunique())
            difference = len(set(prediction_ids) ^ ids["test"])
            result["prediction_id_symmetric_difference"] = difference

            expected_labels = dict(
                zip(frame["block_id"].astype(str), frame["label"].astype(int))
            )
            actual_labels = pd.to_numeric(
                predictions["actual_label"], errors="coerce"
            )
            mismatches = 0
            for block_id, actual in zip(
                predictions["block_id"].astype(str), actual_labels
            ):
                if block_id not in expected_labels or pd.isna(actual):
                    mismatches += 1
                elif int(actual) != expected_labels[block_id]:
                    mismatches += 1
            result["prediction_actual_label_mismatches"] = mismatches

            if len(predictions) != len(ids["test"]):
                result["errors"].append(
                    "Saved prediction row count differs from reconstructed test split."
                )
            if difference:
                result["errors"].append(
                    "Saved prediction IDs do not match reconstructed test IDs."
                )
            if prediction_ids.duplicated().any():
                result["errors"].append(
                    "Saved test predictions contain duplicate block IDs."
                )
            if mismatches:
                result["errors"].append(
                    "Saved prediction actual labels differ from dataset labels."
                )

    if not report_path.is_file():
        result["warnings"].append(f"Experiment report not found: {report_path}")
    else:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        expected_sizes = {
            "train_blocks": len(split["train"]),
            "validation_blocks": len(split["validation"]),
            "test_blocks": len(split["test"]),
        }
        for key, expected in expected_sizes.items():
            if report.get(key) != expected:
                result["report_size_mismatches"].append(
                    {"field": key, "expected": expected, "actual": report.get(key)}
                )
        if report.get("seed") != seed:
            result["errors"].append("Experiment report seed does not match directory.")
        if result["report_size_mismatches"]:
            result["errors"].append(
                "Experiment report split sizes differ from reconstructed split."
            )

    result["status"] = "FAIL" if result["errors"] else (
        "REVIEW" if result["warnings"] else "PASS"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--experiments-dir", type=Path, default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "model_evaluation" / "train_only_drain3_split_audit.json",
    )
    args = parser.parse_args()

    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.data}")
    if not 0 < args.test_size < 1 or not 0 < args.validation_size < 1:
        parser.error("test-size and validation-size must be between 0 and 1.")
    if args.test_size + args.validation_size >= 1:
        parser.error("test-size + validation-size must be less than 1.")

    frame = pd.read_csv(args.data, dtype={"block_id": "string"})
    required = {"block_id", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    frame = frame.dropna(subset=["block_id", "label"]).copy()
    frame["block_id"] = frame["block_id"].astype("string").str.strip()
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    if frame["block_id"].isna().any() or frame["block_id"].duplicated().any():
        raise ValueError("block_id values must be present and unique.")
    if set(frame["label"].unique()) != {0, 1}:
        raise ValueError("Labels must contain both 0=Normal and 1=Anomaly.")

    results = [
        audit_seed(frame, seed, args.experiments_dir, args.test_size, args.validation_size)
        for seed in args.seeds
    ]
    status = "FAIL" if any(item["status"] == "FAIL" for item in results) else (
        "REVIEW" if any(item["status"] == "REVIEW" for item in results) else "PASS"
    )
    report = {
        "experiment": "train_only_drain3_frozen_inference",
        "dataset": str(args.data),
        "rows": int(len(frame)),
        "seeds": results,
        "status": status,
        "interpretation": (
            "This verifies pairwise block-ID separation, saved test prediction IDs "
            "and labels, and reported split sizes against a reconstruction of the "
            "train-only Drain3 split. It does not establish temporal or host-level "
            "independence and does not prove absence of every possible leakage path."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Overall train-only Drain3 split audit: {status}")
    print(f"Dataset rows: {len(frame):,} | Seeds: {', '.join(map(str, args.seeds))}")
    for item in results:
        print(
            f"Seed {item['seed']}: {item['status']} | "
            f"train={item['split_sizes']['train']:,}, "
            f"validation={item['split_sizes']['validation']:,}, "
            f"test={item['split_sizes']['test']:,} | "
            f"ID overlaps={item['block_id_overlaps']} | "
            f"prediction ID differences="
            f"{item['prediction_id_symmetric_difference']}"
        )
        for error in item["errors"]:
            print(f"  ERROR: {error}")
        for warning in item["warnings"]:
            print(f"  WARNING: {warning}")
    print(f"Saved report: {args.output}")
    return 1 if status == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
