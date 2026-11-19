"""Build block-level event-count features from parsed HDFS logs."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"block_id", "event_id"}


def _event_sort_key(value: str) -> tuple[int, int | str]:
    """Sort numeric Drain cluster IDs numerically, other IDs lexically."""
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)


def build_block_event_matrix(records: pd.DataFrame) -> pd.DataFrame:
    """Return one row per block and one column per event ID.

    Each cell contains the number of times that event template appeared in
    that block. Rows without a block ID or event ID cannot contribute and
    are excluded. The input DataFrame is not modified.
    """
    missing = REQUIRED_COLUMNS.difference(records.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    usable = records.loc[:, ["block_id", "event_id"]].copy()
    usable = usable.dropna(subset=["block_id", "event_id"])
    usable["block_id"] = usable["block_id"].astype(str)
    usable["event_id"] = usable["event_id"].astype(str)
    usable = usable[
        (usable["block_id"].str.len() > 0) & (usable["event_id"].str.len() > 0)
    ]

    if usable.empty:
        empty = pd.DataFrame()
        empty.index.name = "block_id"
        return empty

    matrix = pd.crosstab(usable["block_id"], usable["event_id"])
    ordered_events = sorted(matrix.columns, key=_event_sort_key)
    matrix = matrix.reindex(columns=ordered_events)
    matrix.index.name = "block_id"
    matrix.columns.name = "event_id"
    return matrix.astype("int64")


def save_feature_matrix(matrix: pd.DataFrame, output_path: str | Path) -> Path:
    """Save a block-event matrix as CSV, keeping block IDs as the first column."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(path, index=True, index_label="block_id", encoding="utf-8")
    return path


def main() -> None:
    cli = argparse.ArgumentParser(
        description="Create a block-level event-count matrix from parsed HDFS CSV."
    )
    cli.add_argument("input", type=Path, help="Parsed log CSV from the HDFS parser")
    cli.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/hdfs_2k_block_features.csv"),
        help="Output feature-matrix CSV path",
    )
    args = cli.parse_args()

    records = pd.read_csv(
        args.input,
        dtype={"block_id": "string", "event_id": "string"},
    )
    matrix = build_block_event_matrix(records)
    output_path = save_feature_matrix(matrix, args.output)

    print(f"Input log rows: {len(records)}")
    print(f"Blocks represented: {len(matrix)}")
    print(f"Event features: {len(matrix.columns)}")
    print(f"Feature values: {int(matrix.to_numpy().sum())}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
