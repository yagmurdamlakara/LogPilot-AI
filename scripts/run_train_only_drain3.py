"""Run an inductive, train-only Drain3 experiment for HDFS block anomaly detection.

For each seed, this script:
1. Reconstructs the exact stratified train/validation/test block split used by baseline.py.
2. Fits Drain3 only on raw log lines belonging to training blocks.
3. Re-reads the raw log and uses TemplateMiner.match() (no updates/new clusters)
   for every block, mapping unmatched messages to a dedicated event bucket.
4. Builds block/event count features and runs the unchanged baseline models.

This is intentionally separate from the existing full-corpus Drain3 experiments.
It can be expensive: each seed requires two complete passes over HDFS.log.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

import pandas as pd
from drain3.template_miner import TemplateMiner
from sklearn.model_selection import train_test_split

from src.models.baseline import train_baselines
from src.preprocessing.log_parser import HDFSLogParser

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "processed" / "hdfs_full_labeled.csv"
DEFAULT_LOG = ROOT / "data" / "raw" / "HDFS" / "HDFS.log"
DEFAULT_OUTPUT = ROOT / "experiments" / "train_only_drain3"
UNMATCHED_EVENT = "__UNMATCHED__"


def split_block_ids(
    frame: pd.DataFrame, seed: int, test_size: float, validation_size: float
) -> tuple[set[str], set[str], set[str]]:
    train_val_idx, test_idx = train_test_split(
        frame.index, test_size=test_size, random_state=seed, stratify=frame["label"]
    )
    relative_validation_size = validation_size / (1.0 - test_size)
    train_idx, validation_idx = train_test_split(
        train_val_idx,
        test_size=relative_validation_size,
        random_state=seed,
        stratify=frame.loc[train_val_idx, "label"],
    )
    return (
        set(frame.loc[train_idx, "block_id"].astype(str)),
        set(frame.loc[validation_idx, "block_id"].astype(str)),
        set(frame.loc[test_idx, "block_id"].astype(str)),
    )


def train_parser_on_training_blocks(
    raw_log: Path, train_ids: set[str], miner: TemplateMiner,
    progress_every: int = 500_000,
) -> dict[str, int]:
    lines_seen = 0
    train_block_lines = 0
    print("Parser fit started: training-block log lines only.", flush=True)
    with raw_log.open("r", encoding="utf-8", errors="replace") as source:
        for raw_line in source:
            message = raw_line.strip()
            if not message:
                continue
            lines_seen += 1
            block_id = HDFSLogParser.extract_block_id(message)
            if block_id in train_ids:
                miner.add_log_message(message)
                train_block_lines += 1
            if lines_seen % progress_every == 0:
                print(
                    f"Parser fit scanned {lines_seen:,} log lines; "
                    f"used {train_block_lines:,} training-block lines; "
                    f"clusters={len(miner.drain.clusters):,}",
                    flush=True,
                )
    print(
        f"Parser fit finished: scanned {lines_seen:,} lines; "
        f"used {train_block_lines:,} training-block lines; "
        f"clusters={len(miner.drain.clusters):,}",
        flush=True,
    )
    return {
        "raw_lines_scanned_for_parser_fit": lines_seen,
        "training_block_log_lines": train_block_lines,
        "trained_clusters": len(miner.drain.clusters),
    }


def build_frozen_features(
    raw_log: Path, known_ids: set[str], miner: TemplateMiner, database: Path,
    progress_every: int = 500_000,
) -> dict[str, int]:
    database.parent.mkdir(parents=True, exist_ok=True)
    database.unlink(missing_ok=True)
    connection = sqlite3.connect(database)
    connection.execute(
        """
        CREATE TABLE counts (
            block_id TEXT NOT NULL,
            event_id TEXT NOT NULL,
            event_count INTEGER NOT NULL,
            PRIMARY KEY (block_id, event_id)
        )
        """
    )
    batch: Counter[tuple[str, str]] = Counter()
    scanned = 0
    matched = 0
    unmatched = 0
    block_lines = 0

    def flush() -> None:
        if not batch:
            return
        connection.executemany(
            """
            INSERT INTO counts (block_id, event_id, event_count)
            VALUES (?, ?, ?)
            ON CONFLICT(block_id, event_id)
            DO UPDATE SET event_count = event_count + excluded.event_count
            """,
            [(block_id, event_id, count) for (block_id, event_id), count in batch.items()],
        )
        batch.clear()
        connection.commit()

    try:
        with raw_log.open("r", encoding="utf-8", errors="replace") as source:
            for raw_line in source:
                message = raw_line.strip()
                if not message:
                    continue
                scanned += 1
                block_id = HDFSLogParser.extract_block_id(message)
                if block_id not in known_ids:
                    continue
                block_lines += 1
                cluster = miner.match(message, full_search_strategy="fallback")
                if cluster is None:
                    event_id = UNMATCHED_EVENT
                    unmatched += 1
                else:
                    event_id = str(cluster.cluster_id)
                    matched += 1
                batch[(block_id, event_id)] += 1
                if len(batch) >= 100_000:
                    flush()
                if scanned % progress_every == 0:
                    print(
                        f"Frozen feature pass scanned {scanned:,} log lines; "
                        f"labelled block lines={block_lines:,}; "
                        f"matched={matched:,}; unmatched={unmatched:,}",
                        flush=True,
                    )
        flush()
        print(
            f"Frozen feature pass finished: scanned {scanned:,} lines; "
            f"labelled block lines={block_lines:,}; matched={matched:,}; "
            f"unmatched={unmatched:,}",
            flush=True,
        )
        distinct_blocks = connection.execute(
            "SELECT COUNT(DISTINCT block_id) FROM counts"
        ).fetchone()[0]
        distinct_events = connection.execute(
            "SELECT COUNT(DISTINCT event_id) FROM counts"
        ).fetchone()[0]
        return {
            "raw_lines_scanned_for_feature_build": scanned,
            "labelled_block_log_lines": block_lines,
            "matched_log_lines": matched,
            "unmatched_log_lines": unmatched,
            "blocks_with_features": int(distinct_blocks),
            "event_features": int(distinct_events),
        }
    finally:
        connection.close()


def load_feature_frame(database: Path, labels: pd.DataFrame) -> pd.DataFrame:
    with sqlite3.connect(database) as connection:
        long_features = pd.read_sql_query(
            "SELECT block_id, event_id, event_count FROM counts", connection
        )
    wide = long_features.pivot(
        index="block_id", columns="event_id", values="event_count"
    ).fillna(0)
    wide.columns = [str(column) for column in wide.columns]
    wide = wide.reset_index()
    return labels.merge(wide, on="block_id", how="left", validate="one_to_one").fillna(0)


def save_templates(miner: TemplateMiner, path: Path) -> None:
    templates = [
        {
            "event_id": str(cluster.cluster_id),
            "template": cluster.get_template(),
            "cluster_size_after_training": int(cluster.size),
        }
        for cluster in miner.drain.clusters
    ]
    templates.sort(key=lambda item: int(item["event_id"]))
    path.write_text(json.dumps(templates, indent=2, ensure_ascii=False), encoding="utf-8")


def run_seed(
    frame: pd.DataFrame,
    raw_log: Path,
    output_root: Path,
    seed: int,
    test_size: float,
    validation_size: float,
) -> dict[str, object]:
    train_ids, validation_ids, test_ids = split_block_ids(
        frame, seed, test_size, validation_size
    )
    miner = TemplateMiner()
    parser_stats = train_parser_on_training_blocks(raw_log, train_ids, miner)

    seed_dir = output_root / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    feature_db = seed_dir / "frozen_features.sqlite3"
    feature_stats = build_frozen_features(
        raw_log, set(frame["block_id"].astype(str)), miner, feature_db
    )
    feature_frame = load_feature_frame(
        feature_db, frame[["block_id", "label"]].copy()
    )
    feature_frame["block_id"] = feature_frame["block_id"].astype("string")
    feature_frame["label"] = feature_frame["label"].astype(int)

    # Guard that the exact same block-level partition is still reproducible.
    expected = {"train": train_ids, "validation": validation_ids, "test": test_ids}
    actual = split_block_ids(frame, seed, test_size, validation_size)
    if expected != dict(zip(("train", "validation", "test"), actual)):
        raise RuntimeError("Reconstructed block split changed unexpectedly.")

    models, predictions, metrics = train_baselines(
        feature_frame,
        test_size=test_size,
        validation_size=validation_size,
        random_state=seed,
    )
    import joblib

    joblib.dump(models, seed_dir / "baseline_models.joblib")
    predictions.to_csv(seed_dir / "test_predictions.csv", index=False)
    with (seed_dir / "metrics.json").open("w", encoding="utf-8") as output:
        json.dump(metrics, output, indent=2)
    save_templates(miner, seed_dir / "drain3_templates.json")

    result = {
        "seed": seed,
        "parser_fit": parser_stats,
        "frozen_feature_build": feature_stats,
        "train_blocks": len(train_ids),
        "validation_blocks": len(validation_ids),
        "test_blocks": len(test_ids),
        "unmatched_fraction": (
            feature_stats["unmatched_log_lines"] /
            max(feature_stats["labelled_block_log_lines"], 1)
        ),
        "metrics": metrics["models"],
        "output_dir": str(seed_dir),
    }
    with (seed_dir / "experiment_report.json").open("w", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
    feature_db.unlink(missing_ok=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--raw-log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.2)
    args = parser.parse_args()

    if not args.data.is_file():
        raise FileNotFoundError(f"Labelled dataset not found: {args.data}")
    if not args.raw_log.is_file():
        raise FileNotFoundError(f"Raw HDFS log not found: {args.raw_log}")
    if args.test_size <= 0 or args.validation_size <= 0 or args.test_size + args.validation_size >= 1:
        parser.error("Split fractions must be positive and sum to less than 1.")

    frame = pd.read_csv(args.data, dtype={"block_id": "string"})
    if not {"block_id", "label"}.issubset(frame.columns):
        raise ValueError("Input CSV must contain block_id and label columns.")
    frame = frame[["block_id", "label"]].dropna().copy()
    frame["block_id"] = frame["block_id"].astype("string").str.strip()
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    if frame["block_id"].isna().any() or frame["block_id"].duplicated().any():
        raise ValueError("block_id values must be non-null and unique.")
    if set(frame["label"].unique()) != {0, 1}:
        raise ValueError("Labels must be encoded as 0=Normal and 1=Anomaly.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    for seed in args.seeds:
        print(f"\n=== Train-only Drain3 experiment: seed {seed} ===", flush=True)
        result = run_seed(
            frame, args.raw_log, args.output_dir, seed,
            args.test_size, args.validation_size,
        )
        all_results.append(result)
        print(
            f"Seed {seed} complete | unmatched log fraction="
            f"{result['unmatched_fraction']:.4%}",
            flush=True,
        )

    summary = {
        "experiment": "train_only_drain3_frozen_inference",
        "seeds": all_results,
        "note": (
            "Drain3 is fitted only on log lines whose block IDs belong to the "
            "training split. Validation/test inference uses match() and cannot "
            "create or update clusters. Unmatched messages are counted in "
            "a dedicated __UNMATCHED__ feature."
        ),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSaved experiment summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
