import csv
from pathlib import Path

from src.preprocessing.full_dataset import (
    export_event_templates,
    export_feature_matrix,
    parse_and_aggregate,
)


class FakeMiner:
    def add_log_message(self, message):
        if "Receiving" in message:
            event_id, template = "1", "Receiving block <*>" 
        elif "Received" in message:
            event_id, template = "2", "Received block <*>"
        else:
            event_id, template = "3", "Heartbeat received"
        return {
            "cluster_id": event_id,
            "template_mined": template,
            "cluster_size": 1,
        }


def test_streaming_pipeline_aggregates_blocks_without_loading_full_file(tmp_path: Path):
    source = tmp_path / "HDFS.log"
    source.write_text(
        "081109 203518 INFO dfs: Receiving block blk_1\n"
        "081109 203519 INFO dfs: Received block blk_1\n"
        "081109 203520 INFO dfs: Receiving block blk_2\n"
        "081109 203521 INFO dfs: Heartbeat received\n"
        "\n",
        encoding="utf-8",
    )
    database = tmp_path / "work.sqlite3"
    matrix_path = tmp_path / "features.csv"
    templates_path = tmp_path / "templates.csv"

    stats = parse_and_aggregate(
        source,
        database,
        miner=FakeMiner(),
        batch_size=2,
        progress_every=100,
    )
    export_feature_matrix(database, matrix_path)
    export_event_templates(database, templates_path)

    assert stats == {
        "parsed_lines": 4,
        "lines_with_block_ids": 3,
        "unique_blocks": 2,
        "unique_events": 2,
        "feature_values": 3,
    }

    with matrix_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert rows == [
        {"block_id": "blk_1", "1": "1", "2": "1"},
        {"block_id": "blk_2", "1": "1", "2": "0"},
    ]

    with templates_path.open(newline="", encoding="utf-8") as file:
        templates = list(csv.DictReader(file))

    assert [row["event_id"] for row in templates] == ["1", "2", "3"]


def test_streaming_pipeline_rejects_invalid_batch_size(tmp_path: Path):
    source = tmp_path / "HDFS.log"
    source.write_text("log line\n", encoding="utf-8")

    try:
        parse_and_aggregate(source, tmp_path / "work.sqlite3", batch_size=0, miner=FakeMiner())
    except ValueError as error:
        assert "batch_size" in str(error)
    else:
        raise AssertionError("Expected ValueError for batch_size=0")
