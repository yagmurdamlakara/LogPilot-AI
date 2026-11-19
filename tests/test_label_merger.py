import pandas as pd
import pytest

from src.preprocessing.label_merger import merge_block_labels, normalize_labels


def test_merges_labels_by_block_id_and_reports_distribution():
    features = pd.DataFrame(
        {"block_id": ["blk_1", "blk_2", "blk_3"], "1": [2, 0, 1], "2": [0, 3, 1]}
    )
    labels = pd.DataFrame(
        {"BlockId": ["blk_1", "blk_2", "blk_4"], "Label": ["Normal", "Anomaly", "Normal"]}
    )

    merged, report = merge_block_labels(features, labels)

    assert merged["block_id"].tolist() == ["blk_1", "blk_2"]
    assert merged["label"].tolist() == [0, 1]
    assert report == {
        "feature_blocks": 3,
        "label_records": 3,
        "matched_blocks": 2,
        "unmatched_feature_blocks": 1,
        "normal_blocks": 1,
        "anomaly_blocks": 1,
    }


def test_normalizes_numeric_and_text_labels():
    values = pd.Series(["Normal", "Anomaly", "0", "1", "abnormal"])
    assert normalize_labels(values).tolist() == [0, 1, 0, 1, 1]


def test_rejects_unknown_label_values():
    with pytest.raises(ValueError, match="Unsupported label values"):
        normalize_labels(pd.Series(["Normal", "Unknown"]))


def test_rejects_duplicate_feature_block_ids():
    features = pd.DataFrame({"block_id": ["blk_1", "blk_1"], "1": [1, 2]})
    labels = pd.DataFrame({"BlockId": ["blk_1"], "Label": ["Normal"]})
    with pytest.raises(ValueError, match="duplicate block IDs"):
        merge_block_labels(features, labels)


def test_rejects_duplicate_label_block_ids():
    features = pd.DataFrame({"block_id": ["blk_1"], "1": [1]})
    labels = pd.DataFrame(
        {"BlockId": ["blk_1", "blk_1"], "Label": ["Normal", "Anomaly"]}
    )
    with pytest.raises(ValueError, match="duplicate block IDs"):
        merge_block_labels(features, labels)
