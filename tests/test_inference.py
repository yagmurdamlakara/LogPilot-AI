import pandas as pd
import pytest
from drain3.template_miner import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

from src.inference.feature_builder import FrozenBlockFeatureBuilder, UNMATCHED_EVENT


def make_miner():
    config = TemplateMinerConfig()
    miner = TemplateMiner(config=config)
    miner.add_log_message(
        "081109 203518 INFO dfs.DataNode: Receiving block blk_111 from host-01"
    )
    return miner


def test_frozen_builder_counts_by_block_without_updating_parser():
    miner = make_miner()
    cluster_count_before = len(miner.drain.clusters)
    feature_columns = [str(cluster.cluster_id) for cluster in miner.drain.clusters]
    feature_columns.append(UNMATCHED_EVENT)

    builder = FrozenBlockFeatureBuilder(miner, feature_columns)
    builder.add_lines([
        "081109 203518 INFO dfs.DataNode: Receiving block blk_222 from host-01",
        "081109 203519 INFO dfs.DataNode: Receiving block blk_222 from host-01",
    ])

    frame = builder.to_frame()
    assert frame["block_id"].tolist() == ["blk_222"]
    event_columns = [column for column in frame.columns if column != "block_id"]
    assert int(frame[event_columns].to_numpy().sum()) == 2
    assert len(miner.drain.clusters) == cluster_count_before
    assert builder.stats()["lines_with_block_ids"] == 2


def test_builder_rejects_schema_without_unmatched_bucket():
    miner = make_miner()
    with pytest.raises(ValueError, match="__UNMATCHED__"):
        FrozenBlockFeatureBuilder(miner, ["1"])


def test_builder_rejects_unexpected_event_id(monkeypatch):
    miner = make_miner()
    builder = FrozenBlockFeatureBuilder(miner, ["not-the-event-id", UNMATCHED_EVENT])
    monkeypatch.setattr(miner, "match", lambda *args, **kwargs: type(
        "Cluster", (), {"cluster_id": 999}
    )())
    with pytest.raises(ValueError, match="absent from model schema"):
        builder.add_line(
            "081109 203518 INFO dfs.DataNode: Receiving block blk_333 from host-01"
        )


def test_empty_input_returns_empty_frame():
    miner = make_miner()
    feature_columns = [str(cluster.cluster_id) for cluster in miner.drain.clusters]
    feature_columns.append(UNMATCHED_EVENT)
    builder = FrozenBlockFeatureBuilder(miner, feature_columns)
    builder.add_lines(["no block id in this message"])
    frame = builder.to_frame()
    assert frame.empty
    assert list(frame.columns) == ["block_id", *feature_columns]
