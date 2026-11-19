"""Parse HDFS log lines into block-aware Drain3 event records."""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
from drain3.template_miner import TemplateMiner


BLOCK_ID_PATTERN = re.compile(r"\bblk_-?\d+\b")
TIMESTAMP_PATTERN = re.compile(r"^\s*(\d{6}\s+\d{6}|\d{4}-\d{2}-\d{2}[T ]\S+)")

OUTPUT_COLUMNS = [
    "line_number", "timestamp", "block_id", "log_message",
    "event_id", "event_template", "cluster_size",
]


class HDFSLogParser:
    """Stateful HDFS log parser backed by a Drain3 template miner."""

    def __init__(self, template_miner: TemplateMiner | None = None) -> None:
        # Reuse one miner for the complete file so templates are shared.
        self.template_miner = (
            template_miner if template_miner is not None else TemplateMiner()
        )

    @staticmethod
    def extract_block_id(log_message: str) -> str | None:
        """Return the first HDFS block ID found in a log line, if any."""
        match = BLOCK_ID_PATTERN.search(log_message)
        return match.group(0) if match else None

    @staticmethod
    def extract_timestamp(log_message: str) -> str | None:
        """Extract common HDFS timestamps without changing the raw message."""
        match = TIMESTAMP_PATTERN.search(log_message)
        return match.group(1).strip() if match else None

    def parse_lines(self, lines: Iterable[str]) -> pd.DataFrame:
        """Parse lines into records; retain lines that have no block ID.

        Blank lines are skipped. Original line numbers are preserved.
        """
        records: list[dict[str, object]] = []
        for line_number, raw_line in enumerate(lines, start=1):
            log_message = raw_line.strip()
            if not log_message:
                continue

            drain_result = self.template_miner.add_log_message(log_message)
            records.append(
                {
                    "line_number": line_number,
                    "timestamp": self.extract_timestamp(log_message),
                    "block_id": self.extract_block_id(log_message),
                    "log_message": log_message,
                    "event_id": str(drain_result["cluster_id"]),
                    "event_template": drain_result["template_mined"],
                    "cluster_size": int(drain_result["cluster_size"]),
                }
            )

        return pd.DataFrame.from_records(records, columns=OUTPUT_COLUMNS)

    def parse_file(
        self, input_path: str | Path, encoding: str = "utf-8"
    ) -> pd.DataFrame:
        """Read and parse a text log file."""
        path = Path(input_path)
        with path.open("r", encoding=encoding, errors="replace") as log_file:
            return self.parse_lines(log_file)

    @staticmethod
    def save_csv(records: pd.DataFrame, output_path: str | Path) -> Path:
        """Save parsed records as UTF-8 CSV and return the output path."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        records.to_csv(path, index=False, encoding="utf-8")
        return path


def main() -> None:
    """Parse one HDFS log file from the command line."""
    cli = argparse.ArgumentParser(
        description="Parse HDFS logs into Drain3 event records."
    )
    cli.add_argument("input", type=Path, help="Path to the input HDFS log file")
    cli.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/parsed_hdfs_logs.csv"),
    )
    args = cli.parse_args()

    parser = HDFSLogParser()
    records = parser.parse_file(args.input)
    output_path = parser.save_csv(records, args.output)
    block_line_count = int(records["block_id"].notna().sum())
    unique_block_count = int(records["block_id"].nunique())
    unique_event_count = int(records["event_id"].nunique())
    print(f"Parsed lines: {len(records)}")
    print(f"Lines with block IDs: {block_line_count}")
    print(f"Unique block IDs: {unique_block_count}")
    print(f"Unique event IDs: {unique_event_count}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
