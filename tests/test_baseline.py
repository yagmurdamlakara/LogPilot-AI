import numpy as np
import pandas as pd
import pytest

from src.models.baseline import evaluate_predictions, select_f1_threshold, train_baselines


def test_evaluate_predictions_uses_anomaly_as_positive_class():
    metrics = evaluate_predictions([0, 0, 1, 1], [0, 1, 1, 1])
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == pytest.approx(0.8)
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]


def test_select_f1_threshold_chooses_threshold_from_validation_scores():
    threshold, metrics = select_f1_threshold(
        [0, 0, 0, 1, 1],
        [0.1, 0.2, 0.3, 0.7, 0.9],
    )
    assert threshold == pytest.approx(0.7)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_select_f1_threshold_rejects_missing_class():
    with pytest.raises(ValueError, match="both classes"):
        select_f1_threshold([0, 0, 0], [0.1, 0.2, 0.3])


def test_select_f1_threshold_rejects_non_finite_scores():
    with pytest.raises(ValueError, match="finite"):
        select_f1_threshold([0, 1], [0.1, np.nan])


def test_train_baselines_returns_tuned_models_and_predictions():
    rows = []
    for index in range(100):
        anomaly = int(index % 5 == 0)
        rows.append({
            "block_id": f"blk_{index}",
            "event_1": index % 3 + anomaly * 8,
            "event_2": index % 7 + anomaly * 5,
            "label": anomaly,
        })
    models, predictions, results = train_baselines(pd.DataFrame(rows), random_state=17)
    assert set(models) == {"isolation_forest", "logistic_regression"}
    assert len(predictions) == 20
    assert results["train_blocks"] == 60
    assert results["validation_blocks"] == 20
    assert results["test_blocks"] == 20
    assert results["train_anomalies"] == 12
    assert results["validation_anomalies"] == 4
    assert results["test_anomalies"] == 4
    assert "threshold" in results["models"]["logistic_regression"]
    assert "logistic_regression_anomaly_score" in predictions


def test_train_baselines_requires_label_column():
    with pytest.raises(ValueError, match="label"):
        train_baselines(pd.DataFrame({"event_1": [0, 1]}))


def test_train_baselines_requires_both_classes():
    data = pd.DataFrame({"event_1": [0, 1, 2, 3], "label": [0, 0, 0, 0]})
    with pytest.raises(ValueError, match="Both classes"):
        train_baselines(data)


def test_train_baselines_rejects_invalid_split_sizes():
    data = pd.DataFrame({"event_1": range(10), "label": [0, 1] * 5})
    with pytest.raises(ValueError, match="less than 1"):
        train_baselines(data, test_size=0.6, validation_size=0.4)
