"""Memory-bounded preprocessing for the complete HDFS_v1 log file.

The Drain3 miner is kept alive for the entire input file. Block/event counts
are accumulated in SQLite so the full per-line log does not need to reside in
RAM. The generated matrix contains one row per block and one column per
Drain3 cluster ID.
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from drain3.template_miner import TemplateMiner

from src.preprocessing.log_parser import HDFSLogParser


def _event_sort_key(value: str) -> tuple[int, int | str]:
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)


def _flush_batch(
    connection: sqlite3.Connection,
    counts: Counter[tuple[str, str]],
    templates: dict[str, tuple[str, int]],
) -> None:
    if counts:
        connection.executemany(
            """
            INSERT INTO block_event_counts (block_id, event_id, event_count)
            VALUES (?, ?, ?)
            ON CONFLICT(block_id, event_id)
            DO UPDATE SET event_count = event_count + excluded.event_count
            """,
            [(block_id, event_id, count)
             for (block_id, event_id), count in counts.items()],
        )
        counts.clear()

    if templates:
        connection.executemany(
            """
            INSERT INTO event_templates (event_id, template, cluster_size)
            VALUES (?, ?, ?)
            ON CONFLICT(event_id)
            DO UPDATE SET
                template = excluded.template,
                cluster_size = excluded.cluster_size
            """,
            [
                (event_id, template, cluster_size)
                for event_id, (template, cluster_size) in templates.items()
            ],
        )
        templates.clear()

    connection.commit()


def parse_and_aggregate(
    input_path: str | Path,
    work_db_path: str | Path,
    *,
    miner: Any | None = None,
    batch_size: int = 10_000,
    progress_every: int = 500_000,
    encoding: str = "utf-8",
) -> dict[str, int]:
    """Parse a complete log file and aggregate block/event counts on disk."""
    source = Path(input_path)
    database = Path(work_db_path)
    database.parent.mkdir(parents=True, exist_ok=True)

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if progress_every < 1:
        raise ValueError("progress_every must be at least 1.")
    if not source.is_file():
        raise FileNotFoundError(f"Input log file not found: {source}")

    # A fresh run must not accidentally mix counts from an earlier run.
    if database.exists():
        database.unlink()

    template_miner = miner if miner is not None else TemplateMiner()
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        """
        CREATE TABLE block_event_counts (
            block_id TEXT NOT NULL,
            event_id TEXT NOT NULL,
            event_count INTEGER NOT NULL,
            PRIMARY KEY (block_id, event_id)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE event_templates (
            event_id TEXT PRIMARY KEY,
            template TEXT NOT NULL,
            cluster_size INTEGER NOT NULL
        )
        """
    )

    counts: Counter[tuple[str, str]] = Counter()
    templates: dict[str, tuple[str, int]] = {}
    parsed_lines = 0
    block_lines = 0

    try:
        with source.open("r", encoding=encoding, errors="replace") as log_file:
            for raw_line in log_file:
                message = raw_line.strip()
                if not message:
                    continue

                result = template_miner.add_log_message(message)
                parsed_lines += 1

                event_id = str(result["cluster_id"])
                templates[event_id] = (
                    str(result["template_mined"]),
                    int(result["cluster_size"]),
                )

                block_id = HDFSLogParser.extract_block_id(message)
                if block_id is not None:
                    counts[(block_id, event_id)] += 1
                    block_lines += 1

                if parsed_lines % batch_size == 0:
                    _flush_batch(connection, counts, templates)

                if parsed_lines % progress_every == 0:
                    print(
                        f"Processed lines: {parsed_lines:,} | "
                        f"lines with block IDs: {block_lines:,}"
                    )

        _flush_batch(connection, counts, templates)

        unique_blocks = connection.execute(
            "SELECT COUNT(DISTINCT block_id) FROM block_event_counts"
        ).fetchone()[0]
        unique_events = connection.execute(
            "SELECT COUNT(DISTINCT event_id) FROM block_event_counts"
        ).fetchone()[0]
        feature_values = connection.execute(
            "SELECT COALESCE(SUM(event_count), 0) FROM block_event_counts"
        ).fetchone()[0]

        return {
            "parsed_lines": parsed_lines,
            "lines_with_block_ids": block_lines,
            "unique_blocks": int(unique_blocks),
            "unique_events": int(unique_events),
            "feature_values": int(feature_values),
        }
    finally:
        connection.close()


def export_feature_matrix(
    work_db_path: str | Path, output_path: str | Path
) -> Path:
    """Export SQLite block/event counts as a wide CSV feature matrix."""
    database = Path(work_db_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(database) as connection:
        event_ids = [
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT event_id FROM block_event_counts"
            )
        ]
        event_ids.sort(key=_event_sort_key)
        event_positions = {event_id: index for index, event_id in enumerate(event_ids)}

        with output.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["block_id", *event_ids])

            cursor = connection.execute(
                """
                SELECT block_id, event_id, event_count
                FROM block_event_counts
                ORDER BY block_id, event_id
                """
            )

            current_block: str | None = None
            row_values: list[int] = []

            for block_id, event_id, event_count in cursor:
                if block_id != current_block:
                    if current_block is not None:
                        writer.writerow([current_block, *row_values])
                    current_block = block_id
                    row_values = [0] * len(event_ids)

                row_values[event_positions[event_id]] = int(event_count)

            if current_block is not None:
                writer.writerow([current_block, *row_values])

    return output


def export_event_templates(
    work_db_path: str | Path, output_path: str | Path
) -> Path:
    """Export the latest known template metadata for each Drain3 cluster."""
    database = Path(work_db_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(database) as connection, output.open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["event_id", "event_template", "cluster_size"])
        rows = connection.execute(
            "SELECT event_id, template, cluster_size FROM event_templates"
        )
        for row in sorted(rows, key=lambda item: _event_sort_key(item[0])):
            writer.writerow(row)

    return output


def main() -> None:
    cli = argparse.ArgumentParser(
        description="Memory-bounded Drain3 preprocessing for the full HDFS log."
    )
    cli.add_argument("input", type=Path, help="Path to the raw HDFS.log file")
    cli.add_argument(
        "--output-matrix",
        type=Path,
        default=Path("data/processed/hdfs_full_block_features.csv"),
    )
    cli.add_argument(
        "--output-templates",
        type=Path,
        default=Path("data/processed/hdfs_full_event_templates.csv"),
    )
    cli.add_argument(
        "--work-db",
        type=Path,
        default=Path("data/processed/.hdfs_feature_work.sqlite3"),
        help="Temporary SQLite aggregation database; overwritten on each run.",
    )
    cli.add_argument("--batch-size", type=int, default=10_000)
    cli.add_argument("--progress-every", type=int, default=500_000)
    args = cli.parse_args()

    stats = parse_and_aggregate(
        args.input,
        args.work_db,
        batch_size=args.batch_size,
        progress_every=args.progress_every,
    )
    matrix_path = export_feature_matrix(args.work_db, args.output_matrix)
    templates_path = export_event_templates(args.work_db, args.output_templates)

    for key, value in stats.items():
        print(f"{key.replace('_', ' ').title()}: {value:,}")
    print(f"Saved feature matrix: {matrix_path}")
    print(f"Saved event templates: {templates_path}")

    # The CSV outputs are the durable artifacts; remove the temporary DB.
    Path(args.work_db).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
