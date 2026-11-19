"""Train, tune and evaluate HDFS block anomaly detectors."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
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


def select_f1_threshold(y_true, anomaly_scores) -> tuple[float, dict[str, object]]:
    """Choose a score threshold on validation data only; higher scores mean more anomalous.

    Ties are resolved by higher precision, then the higher threshold (fewer alerts).
    """
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(anomaly_scores, dtype=float)
    if y.ndim != 1 or scores.ndim != 1 or len(y) != len(scores) or len(y) == 0:
        raise ValueError("y_true and anomaly_scores must be non-empty 1D arrays of equal length.")
    if not np.isfinite(scores).all():
        raise ValueError("Anomaly scores must contain only finite values.")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Validation labels must contain both classes: 0=Normal and 1=Anomaly.")

    # Include a threshold above the maximum score, representing no positive predictions.
    candidates = np.unique(np.concatenate([scores, [np.nextafter(scores.max(), np.inf)]]))
    best_threshold = float(candidates[0])
    best_metrics: dict[str, object] | None = None
    best_key = (-1.0, -1.0, -np.inf)

    for threshold in candidates:
        predicted = (scores >= threshold).astype(int)
        metrics = evaluate_predictions(y, predicted)
        key = (float(metrics["f1"]), float(metrics["precision"]), float(threshold))
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = metrics

    assert best_metrics is not None
    return best_threshold, best_metrics


def train_baselines(
    data: pd.DataFrame,
    *,
    test_size: float = 0.2,
    validation_size: float = 0.2,
    random_state: int = 42,
):
    """Train two baselines, tune thresholds on validation, and evaluate once on test."""
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
    if y.value_counts().min() < 3:
        raise ValueError("At least three samples per class are required for train/validation/test splitting.")
    if not 0 < test_size < 1 or not 0 < validation_size < 1:
        raise ValueError("test_size and validation_size must be between 0 and 1.")
    if test_size + validation_size >= 1:
        raise ValueError("test_size + validation_size must be less than 1.")

    train_val_idx, test_idx = train_test_split(
        frame.index, test_size=test_size, random_state=random_state, stratify=y
    )
    relative_validation_size = validation_size / (1.0 - test_size)
    train_idx, validation_idx = train_test_split(
        train_val_idx,
        test_size=relative_validation_size,
        random_state=random_state,
        stratify=y.loc[train_val_idx],
    )

    X_train, X_validation, X_test = X.loc[train_idx], X.loc[validation_idx], X.loc[test_idx]
    y_train, y_validation, y_test = y.loc[train_idx], y.loc[validation_idx], y.loc[test_idx]

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

    results: dict[str, object] = {
        "random_state": random_state,
        "train_fraction": 1.0 - test_size - validation_size,
        "validation_fraction": validation_size,
        "test_fraction": test_size,
        "feature_count": len(feature_columns),
        "train_blocks": int(len(train_idx)),
        "validation_blocks": int(len(validation_idx)),
        "test_blocks": int(len(test_idx)),
        "train_anomalies": int(y_train.sum()),
        "validation_anomalies": int(y_validation.sum()),
        "test_anomalies": int(y_test.sum()),
        "models": {},
    }

    predictions = pd.DataFrame({
        "block_id": frame.loc[test_idx, "block_id"].astype(str).to_numpy()
        if "block_id" in frame.columns else test_idx.astype(str),
        "actual_label": y_test.to_numpy(),
    })

    for name, model in models.items():
        if name == "logistic_regression":
            model.fit(X_train, y_train)
            validation_scores = model.predict_proba(X_validation)[:, 1]
            test_scores = model.predict_proba(X_test)[:, 1]
        else:
            model.fit(X_train)
            # Isolation Forest returns lower scores for more anomalous samples.
            validation_scores = -model.score_samples(X_validation)
            test_scores = -model.score_samples(X_test)

        threshold, validation_metrics = select_f1_threshold(y_validation, validation_scores)
        test_predictions = (test_scores >= threshold).astype(int)
        metrics = evaluate_predictions(y_test, test_predictions)
        metrics["threshold"] = threshold
        metrics["validation_metrics_at_selected_threshold"] = validation_metrics

        started = time.perf_counter()
        if name == "logistic_regression":
            model.predict_proba(X_test)
        else:
            model.score_samples(X_test)
        elapsed = time.perf_counter() - started
        metrics["test_inference_ms_per_block"] = float(elapsed * 1000 / max(len(X_test), 1))
        results["models"][name] = metrics

        predictions[f"{name}_anomaly_score"] = test_scores
        predictions[f"{name}_prediction"] = test_predictions

    return models, predictions.reset_index(drop=True), results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and tune LogPilot AI baseline models.")
    parser.add_argument("input", type=Path, help="Labelled block feature CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/baseline"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    data = pd.read_csv(args.input, dtype={"block_id": "string"})
    models, predictions, results = train_baselines(
        data,
        test_size=args.test_size,
        validation_size=args.validation_size,
        random_state=args.random_state,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, args.output_dir / "baseline_models.joblib")
    predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    for split in ("train", "validation", "test"):
        print(
            f"{split.title()} blocks: {results[f'{split}_blocks']} | "
            f"anomalies: {results[f'{split}_anomalies']}"
        )
    for name, metrics in results["models"].items():
        validation = metrics["validation_metrics_at_selected_threshold"]
        print(
            f"{name}: threshold={metrics['threshold']:.6f} | "
            f"validation_f1={validation['f1']:.4f} | "
            f"test_precision={metrics['precision']:.4f} | "
            f"test_recall={metrics['recall']:.4f} | test_f1={metrics['f1']:.4f} | "
            f"inference_ms_per_block={metrics['test_inference_ms_per_block']:.4f}"
        )
        print(f"  test confusion_matrix [ [TN, FP], [FN, TP] ]: {metrics['confusion_matrix']}")
    print(f"Saved results: {args.output_dir}")


if __name__ == "__main__":
    main()
