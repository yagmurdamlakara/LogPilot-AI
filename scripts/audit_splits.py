"""Reconstruct and audit the exact train/validation/test splits used by baseline.py.

Run from the repository root:
    python scripts/audit_splits.py
    python scripts/audit_splits.py --seeds 11 22 33 44 55

The split procedure intentionally mirrors src/models/baseline.py. It also
compares reconstructed test IDs and labels with saved test_predictions.csv
files when those files are present.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "processed" / "hdfs_full_labeled.csv"
DEFAULT_EXPERIMENTS = ROOT / "experiments"
DEFAULT_SEEDS = (11, 22, 33, 44, 55)


def class_counts(labels: pd.Series) -> dict[str, int]:
    counts = labels.value_counts().sort_index()
    return {str(int(label)): int(count) for label, count in counts.items()}


def reconstruct_splits(
    frame: pd.DataFrame, *, seed: int, test_size: float, validation_size: float
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
    splits = reconstruct_splits(
        frame, seed=seed, test_size=test_size, validation_size=validation_size
    )
    id_sets = {
        name: set(frame.loc[index, "block_id"].astype(str))
        for name, index in splits.items()
    }
    overlaps = {
        "train_validation": len(id_sets["train"] & id_sets["validation"]),
        "train_test": len(id_sets["train"] & id_sets["test"]),
        "validation_test": len(id_sets["validation"] & id_sets["test"]),
    }

    result: dict[str, object] = {
        "seed": seed,
        "split_sizes": {name: len(index) for name, index in splits.items()},
        "anomaly_counts": {
            name: int(frame.loc[index, "label"].sum()) for name, index in splits.items()
        },
        "class_counts": {
            name: class_counts(frame.loc[index, "label"]) for name, index in splits.items()
        },
        "block_id_overlaps": overlaps,
        "test_prediction_file": None,
        "errors": [],
        "warnings": [],
    }
    if any(overlaps.values()):
        result["errors"].append("Block IDs overlap between splits.")

    prediction_path = experiments_dir / f"baseline_seed_{seed}" / "test_predictions.csv"
    if prediction_path.is_file():
        predictions = pd.read_csv(prediction_path, dtype={"block_id": "string"})
        if "block_id" not in predictions.columns or "actual_label" not in predictions.columns:
            result["errors"].append(
                f"{prediction_path} must contain block_id and actual_label columns."
            )
        else:
            predicted_ids = predictions["block_id"].dropna().astype(str)
            expected_ids = id_sets["test"]
            result["test_prediction_file"] = str(prediction_path)
            result["prediction_rows"] = int(len(predictions))
            result["prediction_unique_block_ids"] = int(predicted_ids.nunique())
            result["prediction_id_symmetric_difference"] = len(
                set(predicted_ids) ^ expected_ids
            )
            expected_labels = frame.set_index(frame["block_id"].astype(str))["label"]
            actual_by_id = dict(zip(
                predictions["block_id"].astype(str),
                pd.to_numeric(predictions["actual_label"], errors="coerce"),
            ))
            label_mismatches = sum(
                1 for block_id in expected_ids
                if block_id in actual_by_id
                and actual_by_id[block_id] != expected_labels.loc[block_id]
            )
            result["prediction_actual_label_mismatches"] = int(label_mismatches)
            if len(predictions) != len(expected_ids):
                result["errors"].append("Saved prediction row count differs from reconstructed test split.")
            if result["prediction_id_symmetric_difference"]:
                result["errors"].append("Saved test prediction IDs do not match reconstructed test IDs.")
            if predicted_ids.duplicated().any():
                result["errors"].append("Saved test predictions contain duplicate block IDs.")
            if label_mismatches:
                result["errors"].append("Saved test actual labels differ from dataset labels.")
    else:
        result["warnings"].append(f"Prediction file not found: {prediction_path}")

    result["status"] = "FAIL" if result["errors"] else (
        "REVIEW" if result["warnings"] else "PASS"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--experiments-dir", type=Path, default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "model_evaluation" / "split_audit.json",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.input}")
    if not 0 < args.test_size < 1 or not 0 < args.validation_size < 1:
        parser.error("test-size and validation-size must be between 0 and 1.")
    if args.test_size + args.validation_size >= 1:
        parser.error("test-size + validation-size must be less than 1.")

    frame = pd.read_csv(args.input, dtype={"block_id": "string"})
    required = {"block_id", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    frame = frame.dropna(subset=["label"]).copy()
    frame["block_id"] = frame["block_id"].astype("string").str.strip()
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    if frame["block_id"].isna().any() or frame["block_id"].duplicated().any():
        raise ValueError("block_id values must be present and unique.")
    if set(frame["label"].unique()) != {0, 1}:
        raise ValueError("Labels must contain both 0=Normal and 1=Anomaly.")
    if frame["label"].value_counts().min() < 3:
        raise ValueError("At least three samples per class are required.")

    results = [
        audit_seed(frame, seed, args.experiments_dir, args.test_size, args.validation_size)
        for seed in args.seeds
    ]
    report = {
        "dataset": str(args.input),
        "rows": int(len(frame)),
        "test_size": args.test_size,
        "validation_size": args.validation_size,
        "train_size": 1.0 - args.test_size - args.validation_size,
        "seeds": results,
        "status": "FAIL" if any(item["status"] == "FAIL" for item in results) else (
            "REVIEW" if any(item["status"] == "REVIEW" for item in results) else "PASS"
        ),
        "interpretation": (
            "This verifies the saved test IDs against a reconstruction of the current "
            "baseline.py split procedure. It does not establish independence across time, "
            "hosts, or parser-training data."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Overall split audit: {report['status']}")
    print(f"Dataset rows: {len(frame):,} | Seeds: {', '.join(map(str, args.seeds))}")
    for item in results:
        print(
            f"Seed {item['seed']}: {item['status']} | "
            f"train={item['split_sizes']['train']:,}, "
            f"validation={item['split_sizes']['validation']:,}, "
            f"test={item['split_sizes']['test']:,} | "
            f"ID overlaps={item['block_id_overlaps']} | "
            f"prediction ID differences={item.get('prediction_id_symmetric_difference', 'N/A')}"
        )
        for error in item["errors"]:
            print(f"  ERROR: {error}")
        for warning in item["warnings"]:
            print(f"  WARNING: {warning}")
    print(f"Saved report: {args.output}")
    return 1 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
