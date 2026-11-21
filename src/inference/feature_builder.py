"""Build frozen Drain3 event-count features for HDFS blocks."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable

import pandas as pd
from drain3.template_miner import TemplateMiner

from src.preprocessing.log_parser import HDFSLogParser

UNMATCHED_EVENT = "__UNMATCHED__"


class FrozenBlockFeatureBuilder:
    """Aggregate block/event counts without modifying the fitted Drain3 state."""

    def __init__(self, miner: TemplateMiner, feature_columns: list[str]) -> None:
        if not feature_columns or UNMATCHED_EVENT not in feature_columns:
            raise ValueError("Feature schema must include the __UNMATCHED__ bucket.")
        self.miner = miner
        self.feature_columns = list(feature_columns)
        self.counts: Counter[tuple[str, str]] = Counter()
        self.lines_seen = 0
        self.block_lines = 0
        self.unmatched_lines = 0

    def add_line(self, raw_line: str) -> None:
        message = raw_line.strip()
        if not message:
            return
        self.lines_seen += 1
        block_id = HDFSLogParser.extract_block_id(message)
        if block_id is None:
            return
        self.block_lines += 1
        cluster = self.miner.match(message, full_search_strategy="fallback")
        event_id = str(cluster.cluster_id) if cluster is not None else UNMATCHED_EVENT
        if cluster is None:
            self.unmatched_lines += 1
        if event_id not in self.feature_columns:
            # Any unexpected ID indicates a parser/schema mismatch; never silently
            # drop it or append a new model feature.
            raise ValueError(f"Drain3 returned event {event_id!r}, absent from model schema.")
        self.counts[(block_id, event_id)] += 1

    def add_lines(self, lines: Iterable[str]) -> None:
        for line in lines:
            self.add_line(line)

    def to_frame(self) -> pd.DataFrame:
        block_ids = sorted({block_id for block_id, _event in self.counts})
        frame = pd.DataFrame(0, index=block_ids, columns=self.feature_columns, dtype="int64")
        frame.index.name = "block_id"
        for (block_id, event_id), count in self.counts.items():
            frame.at[block_id, event_id] = count
        return frame.reset_index()

    def stats(self) -> dict[str, int]:
        return {
            "lines_seen": self.lines_seen,
            "lines_with_block_ids": self.block_lines,
            "blocks_with_features": len({block_id for block_id, _ in self.counts}),
            "unmatched_lines": self.unmatched_lines,
        }
