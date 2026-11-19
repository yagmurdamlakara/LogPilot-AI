"""Merge HDFS block-level ground-truth labels with event-count features."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _find_column(frame: pd.DataFrame, candidates: tuple[str, ...], role: str) -> str:
    lookup = {str(column).strip().lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    raise ValueError(
        f"Could not find {role} column. Expected one of: {', '.join(candidates)}. "
        f"Available columns: {', '.join(map(str, frame.columns))}"
    )


def normalize_labels(values: pd.Series) -> pd.Series:
    """Normalize common HDFS labels to 0=Normal and 1=Anomaly."""
    normalized = values.astype("string").str.strip().str.lower()
    mapping = {
        "normal": 0,
        "0": 0,
        "false": 0,
        "anomaly": 1,
        "abnormal": 1,
        "1": 1,
        "true": 1,
    }
    mapped = normalized.map(mapping)
    if mapped.isna().any():
        unknown = sorted(normalized[mapped.isna()].dropna().unique().tolist())
        raise ValueError(f"Unsupported label values: {unknown}")
    return mapped.astype("int64")


def merge_block_labels(
    features: pd.DataFrame, labels: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Join feature rows to ground-truth labels by block ID.

    Unlabelled feature blocks are excluded from the returned modelling table
    and counted in the report. Duplicate IDs in either input are rejected
    rather than silently multiplying or overwriting rows.
    """
    feature_id = _find_column(features, ("block_id", "BlockId", "blockId"), "feature block ID")
    label_id = _find_column(labels, ("BlockId", "block_id", "blockId"), "label block ID")
    label_value = _find_column(labels, ("Label", "label", "anomaly_label"), "label")

    feature_frame = features.copy()
    label_frame = labels[[label_id, label_value]].copy()

    feature_frame[feature_id] = feature_frame[feature_id].astype("string").str.strip()
    label_frame[label_id] = label_frame[label_id].astype("string").str.strip()
    feature_frame = feature_frame.dropna(subset=[feature_id])
    label_frame = label_frame.dropna(subset=[label_id])

    if feature_frame[feature_id].duplicated().any():
        raise ValueError("Feature matrix contains duplicate block IDs.")
    if label_frame[label_id].duplicated().any():
        raise ValueError("Label file contains duplicate block IDs.")

    label_frame["label"] = normalize_labels(label_frame[label_value])
    label_frame = label_frame[[label_id, "label"]].rename(columns={label_id: feature_id})

    merged = feature_frame.merge(label_frame, on=feature_id, how="left", validate="one_to_one")
    unmatched = int(merged["label"].isna().sum())
    matched = merged.loc[merged["label"].notna()].copy()
    matched["label"] = matched["label"].astype("int64")

    report = {
        "feature_blocks": int(len(feature_frame)),
        "label_records": int(len(label_frame)),
        "matched_blocks": int(len(matched)),
        "unmatched_feature_blocks": unmatched,
        "normal_blocks": int((matched["label"] == 0).sum()),
        "anomaly_blocks": int((matched["label"] == 1).sum()),
    }
    return matched.reset_index(drop=True), report


def main() -> None:
    cli = argparse.ArgumentParser(
        description="Merge HDFS anomaly labels into a block event feature matrix."
    )
    cli.add_argument("features", type=Path, help="Block-level event feature CSV")
    cli.add_argument("labels", type=Path, help="HDFS anomaly_label.csv")
    cli.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/hdfs_2k_labeled.csv"),
    )
    args = cli.parse_args()

    features = pd.read_csv(args.features, dtype={"block_id": "string"})
    labels = pd.read_csv(args.labels, dtype="string")
    merged, report = merge_block_labels(features, labels)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False, encoding="utf-8")

    for key, value in report.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
