"""Audit the labeled HDFS block feature table for common leakage risks.

Run from the repository root:
    python scripts/audit_dataset_leakage.py

This is a diagnostic audit, not a proof that the dataset is leakage-free.
It checks identifiers, labels, suspicious feature names, duplicate columns,
and unusually strong single-feature association with the target.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "processed" / "hdfs_full_labeled.csv"
ID_CANDIDATES = ("block_id", "blockid", "block_id".lower())
LABEL_CANDIDATES = ("label", "anomaly_label", "target", "is_anomaly")
SUSPICIOUS_NAME_PATTERNS = (
    r"(^|_)label($|_)",
    r"(^|_)target($|_)",
    r"(^|_)anomaly($|_)",
    r"(^|_)ground.?truth($|_)",
    r"(^|_)is.?abnormal($|_)",
)


def find_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lookup = {str(column).strip().lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--high-auc-threshold", type=float, default=0.999)
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {args.input}. Pass the labeled CSV path explicitly."
        )

    frame = pd.read_csv(args.input, dtype={"block_id": "string"})
    id_column = find_column(list(frame.columns), ID_CANDIDATES)
    label_column = find_column(list(frame.columns), LABEL_CANDIDATES)

    report: dict[str, object] = {
        "input": str(args.input),
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "checks": {},
        "warnings": [],
        "errors": [],
    }

    if id_column is None:
        report["errors"].append("Block identifier column not found.")
    else:
        ids = frame[id_column].astype("string").str.strip()
        report["checks"]["null_block_ids"] = int(ids.isna().sum())
        report["checks"]["duplicate_block_ids"] = int(ids.duplicated().sum())
        if ids.isna().any() or ids.duplicated().any():
            report["errors"].append("Block IDs must be non-null and unique.")

    if label_column is None:
        report["errors"].append("Target label column not found.")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 2

    labels = pd.to_numeric(frame[label_column], errors="coerce")
    report["checks"]["null_or_non_numeric_labels"] = int(labels.isna().sum())
    report["checks"]["label_values"] = sorted(labels.dropna().unique().tolist())
    report["checks"]["class_counts"] = {
        str(int(key)): int(value) for key, value in labels.value_counts().sort_index().items()
    }
    if labels.isna().any() or set(labels.dropna().unique()) != {0, 1}:
        report["errors"].append("Labels must be complete and encoded as 0=Normal, 1=Anomaly.")

    feature_columns = [
        column for column in frame.columns if column not in {id_column, label_column}
    ]
    suspicious = [
        column for column in feature_columns
        if any(re.search(pattern, str(column).lower()) for pattern in SUSPICIOUS_NAME_PATTERNS)
    ]
    report["checks"]["feature_count_excluding_id_and_label"] = len(feature_columns)
    report["checks"]["suspicious_feature_names"] = suspicious
    if suspicious:
        report["warnings"].append(
            "Review suspicious feature names manually; names alone do not prove leakage."
        )

    numeric = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    non_numeric = [column for column in feature_columns if numeric[column].isna().any()]
    report["checks"]["features_with_non_numeric_or_missing_values"] = non_numeric
    if non_numeric:
        report["warnings"].append(
            "Some features contain missing/non-numeric values; inspect preprocessing."
        )

    duplicated_feature_columns = []
    seen: dict[tuple[object, ...], str] = {}
    for column in numeric.columns:
        signature = tuple(numeric[column].fillna("__NA__").tolist())
        if signature in seen:
            duplicated_feature_columns.append([seen[signature], str(column)])
        else:
            seen[signature] = str(column)
    report["checks"]["duplicate_feature_columns"] = duplicated_feature_columns
    if duplicated_feature_columns:
        report["warnings"].append("Identical feature columns were found.")

    high_auc = []
    if not labels.isna().any() and set(labels.unique()) == {0, 1}:
        for column in numeric.columns:
            values = numeric[column]
            if values.isna().any() or values.nunique(dropna=True) < 2:
                continue
            try:
                auc = float(roc_auc_score(labels, values))
                reverse_auc = max(auc, 1.0 - auc)
                if reverse_auc >= args.high_auc_threshold:
                    high_auc.append({"feature": str(column), "auc_or_reverse_auc": reverse_auc})
            except ValueError:
                continue
    report["checks"]["very_high_univariate_auc_features"] = high_auc
    if high_auc:
        report["warnings"].append(
            "Very high single-feature AUC may be legitimate or indicate leakage; investigate provenance."
        )

    report["status"] = "ERRORS_FOUND" if report["errors"] else (
        "WARNINGS_FOUND" if report["warnings"] else "BASIC_CHECKS_PASSED"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
