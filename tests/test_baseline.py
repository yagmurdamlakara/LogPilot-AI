import pandas as pd
import pytest

from src.models.baseline import evaluate_predictions, train_baselines


def test_evaluate_predictions_uses_anomaly_as_positive_class():
    metrics = evaluate_predictions([0, 0, 1, 1], [0, 1, 1, 1])
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == pytest.approx(0.8)
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]


def test_train_baselines_returns_both_models_and_predictions():
    rows = []
    for index in range(60):
        anomaly = int(index % 5 == 0)
        rows.append({
            "block_id": f"blk_{index}",
            "event_1": index % 3 + anomaly * 8,
            "event_2": index % 7 + anomaly * 5,
            "label": anomaly,
        })
    models, predictions, results = train_baselines(pd.DataFrame(rows), random_state=17)
    assert set(models) == {"isolation_forest", "logistic_regression"}
    assert len(predictions) == 12
    assert results["train_blocks"] == 48
    assert results["test_blocks"] == 12
    assert results["test_anomalies"] == 2
    assert "f1" in results["models"]["logistic_regression"]


def test_train_baselines_requires_label_column():
    with pytest.raises(ValueError, match="label"):
        train_baselines(pd.DataFrame({"event_1": [0, 1]}))


def test_train_baselines_requires_both_classes():
    data = pd.DataFrame({"event_1": [0, 1, 2, 3], "label": [0, 0, 0, 0]})
    with pytest.raises(ValueError, match="Both classes"):
        train_baselines(data)
