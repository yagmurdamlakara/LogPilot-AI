from pathlib import Path

import pandas as pd

from src.preprocessing.log_parser import HDFSLogParser, OUTPUT_COLUMNS


def test_extract_block_id_from_hdfs_message():
    message = "081109 203518 INFO dfs.DataNode: Receiving block blk_-1608999687919862906"
    assert HDFSLogParser.extract_block_id(message) == "blk_-1608999687919862906"


def test_extract_block_id_returns_none_when_missing():
    assert HDFSLogParser.extract_block_id("DataNode started successfully") is None


def test_parse_lines_preserves_block_ids_and_raw_messages():
    lines = [
        "081109 203518 INFO dfs.DataNode: Receiving block blk_-1608999687919862906",
        "081109 203519 INFO dfs.DataNode: Received block blk_123456789",
        "081109 203520 INFO dfs.DataNode: Heartbeat received",
    ]
    result = HDFSLogParser().parse_lines(lines)

    assert list(result.columns) == OUTPUT_COLUMNS
    assert len(result) == 3
    assert result.loc[0, "block_id"] == "blk_-1608999687919862906"
    assert result.loc[1, "block_id"] == "blk_123456789"
    assert pd.isna(result.loc[2, "block_id"])
    assert result.loc[0, "log_message"] == lines[0]
    assert result["event_id"].notna().all()
    assert result["event_template"].notna().all()


def test_parse_lines_skips_blank_lines_and_keeps_original_line_numbers():
    result = HDFSLogParser().parse_lines(
        [
            "081109 203518 INFO dfs.DataNode: Receiving block blk_123",
            "",
            "081109 203520 INFO dfs.DataNode: Received block blk_456",
        ]
    )
    assert len(result) == 2
    assert result["line_number"].tolist() == [1, 3]


def test_parse_file_and_save_csv(tmp_path: Path):
    input_path = tmp_path / "HDFS.log"
    output_path = tmp_path / "parsed" / "HDFS.csv"
    input_path.write_text(
        "081109 203518 INFO dfs.DataNode: Receiving block blk_123\n",
        encoding="utf-8",
    )

    parser = HDFSLogParser()
    records = parser.parse_file(input_path)
    saved_path = parser.save_csv(records, output_path)

    assert saved_path.exists()
    loaded = pd.read_csv(saved_path)
    assert loaded.loc[0, "block_id"] == "blk_123"
    # CSV readers may infer the numeric-looking Drain cluster ID as an int.
    # Compare normalized strings because EventId is an identifier, not a measure.
    assert str(loaded.loc[0, "event_id"]) == str(records.loc[0, "event_id"])
