import pandas as pd
import pytest

from src.feature_extraction.block_features import (
    build_block_event_matrix,
    save_feature_matrix,
)


def test_builds_event_count_matrix_per_block():
    records = pd.DataFrame(
        {
            "block_id": ["blk_1", "blk_1", "blk_1", "blk_2", "blk_2"],
            "event_id": ["2", "1", "2", "1", "3"],
        }
    )

    matrix = build_block_event_matrix(records)

    assert matrix.index.name == "block_id"
    assert matrix.columns.tolist() == ["1", "2", "3"]
    assert matrix.loc["blk_1"].tolist() == [1, 2, 0]
    assert matrix.loc["blk_2"].tolist() == [1, 0, 1]
    assert matrix.to_numpy().sum() == 5


def test_drops_rows_without_block_or_event_id():
    records = pd.DataFrame(
        {
            "block_id": ["blk_1", None, "blk_2"],
            "event_id": ["1", "2", None],
        }
    )

    matrix = build_block_event_matrix(records)

    assert matrix.index.tolist() == ["blk_1"]
    assert matrix.loc["blk_1", "1"] == 1


def test_empty_valid_input_returns_empty_matrix():
    records = pd.DataFrame(
        {"block_id": [None], "event_id": ["1"]}
    )

    matrix = build_block_event_matrix(records)

    assert matrix.empty
    assert matrix.index.name == "block_id"


def test_missing_required_column_raises_value_error():
    with pytest.raises(ValueError, match="event_id"):
        build_block_event_matrix(pd.DataFrame({"block_id": ["blk_1"]}))


def test_save_feature_matrix_round_trip(tmp_path):
    records = pd.DataFrame(
        {"block_id": ["blk_1", "blk_1", "blk_2"], "event_id": ["1", "2", "2"]}
    )
    matrix = build_block_event_matrix(records)
    output = save_feature_matrix(matrix, tmp_path / "features" / "matrix.csv")

    loaded = pd.read_csv(output, dtype={"block_id": "string"})
    assert output.exists()
    assert loaded["block_id"].tolist() == ["blk_1", "blk_2"]
    assert loaded["1"].tolist() == [1, 0]
    assert loaded["2"].tolist() == [1, 1]
