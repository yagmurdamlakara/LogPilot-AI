"""Train and evaluate baseline HDFS block anomaly detectors."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def evaluate_predictions(y_true, y_pred) -> dict[str, object]:
    """Calculate anomaly-class metrics (1=Anomaly) and a fixed-label confusion matrix."""
    return {
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def train_baselines(data: pd.DataFrame, *, test_size: float = 0.2, random_state: int = 42):
    """Train Isolation Forest and Logistic Regression on one stratified block split.

    The held-out test set is not used for fitting or threshold selection.
    Isolation Forest uses its built-in auto threshold. Logistic Regression
    uses class weighting to account for imbalanced labels.
    """
    if "label" not in data.columns:
        raise ValueError("Input data must contain a label column.")
    feature_columns = [c for c in data.columns if c not in {"label", "block_id"}]
    if not feature_columns:
        raise ValueError("No feature columns found.")

    frame = data.dropna(subset=["label"]).copy()
    X = frame[feature_columns].apply(pd.to_numeric, errors="raise")
    y = frame["label"].astype(int)
    if set(y.unique()) != {0, 1}:
        raise ValueError("Both classes are required; labels must be 0=Normal and 1=Anomaly.")
    if y.value_counts().min() < 2:
        raise ValueError("At least two samples per class are required for stratified splitting.")

    train_idx, test_idx = train_test_split(
        frame.index, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    models = {
        "isolation_forest": IsolationForest(
            n_estimators=200, contamination="auto", random_state=random_state, n_jobs=-1
        ),
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                class_weight="balanced", max_iter=2000, random_state=random_state
            )),
        ]),
    }
    results = {
        "random_state": random_state,
        "test_size": test_size,
        "feature_count": len(feature_columns),
        "train_blocks": int(len(train_idx)),
        "test_blocks": int(len(test_idx)),
        "train_anomalies": int(y_train.sum()),
        "test_anomalies": int(y_test.sum()),
        "models": {},
    }
    predictions = pd.DataFrame(index=test_idx)
    if "block_id" in frame.columns:
        predictions["block_id"] = frame.loc[test_idx, "block_id"].astype(str)
    predictions["actual_label"] = y_test

    for name, model in models.items():
        if name == "logistic_regression":
            model.fit(X_train, y_train)
        else:
            model.fit(X_train)
        started = time.perf_counter()
        raw = model.predict(X_test)
        elapsed = time.perf_counter() - started
        predicted = (raw == -1).astype(int) if name == "isolation_forest" else raw.astype(int)

        metrics = evaluate_predictions(y_test, predicted)
        metrics["test_inference_ms_per_block"] = float(elapsed * 1000 / max(len(X_test), 1))
        results["models"][name] = metrics
        predictions[f"{name}_prediction"] = predicted

    return models, predictions.reset_index(drop=True), results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LogPilot AI baseline models.")
    parser.add_argument("input", type=Path, help="Labelled block feature CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/baseline"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    data = pd.read_csv(args.input, dtype={"block_id": "string"})
    models, predictions, results = train_baselines(
        data, test_size=args.test_size, random_state=args.random_state
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, args.output_dir / "baseline_models.joblib")
    predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(f"Train blocks: {results['train_blocks']}")
    print(f"Test blocks: {results['test_blocks']}")
    print(f"Train anomalies: {results['train_anomalies']}")
    print(f"Test anomalies: {results['test_anomalies']}")
    for name, metrics in results["models"].items():
        print(
            f"{name}: precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, f1={metrics['f1']:.4f}, "
            f"inference_ms_per_block={metrics['test_inference_ms_per_block']:.4f}"
        )
        print(f"  confusion_matrix [ [TN, FP], [FN, TP] ]: {metrics['confusion_matrix']}")
    print(f"Saved results: {args.output_dir}")


if __name__ == "__main__":
    main()
