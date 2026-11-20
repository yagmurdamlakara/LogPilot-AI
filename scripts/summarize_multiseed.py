"""Aggregate five-seed baseline metrics for LogPilot AI.

Run from the repository root:
    python scripts/summarize_multiseed.py

Reads experiments/baseline_seed_{11,22,33,44,55}/metrics.json and writes
experiments/summary/multi_seed_results.csv. Uses sample standard deviation
(ddof=1), matching the project's reporting convention.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (11, 22, 33, 44, 55)
MODELS = ("isolation_forest", "logistic_regression")
METRICS = ("precision", "recall", "f1", "test_inference_ms_per_block")
OUTPUT = ROOT / "experiments" / "summary" / "multi_seed_results.csv"


def sample_mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("Cannot aggregate an empty metric list.")
    mean = sum(values) / len(values)
    if len(values) < 2:
        raise ValueError("At least two seed values are required for sample std.")
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def main() -> None:
    collected: dict[str, dict[str, list[float]]] = {
        model: {metric: [] for metric in METRICS} for model in MODELS
    }

    for seed in SEEDS:
        metrics_path = ROOT / "experiments" / f"baseline_seed_{seed}" / "metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        actual_seed = payload.get("random_state")
        if actual_seed != seed:
            raise ValueError(
                f"{metrics_path}: expected random_state={seed}, got {actual_seed!r}"
            )

        for model in MODELS:
            model_payload = payload.get("models", {}).get(model)
            if model_payload is None:
                raise KeyError(f"{metrics_path}: missing model {model!r}")

            for metric in METRICS:
                if metric == "test_inference_ms_per_block":
                    value = model_payload.get(metric)
                else:
                    value = model_payload.get(metric)
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(
                        f"{metrics_path}: invalid {model}.{metric}: {value!r}"
                    )
                collected[model][metric].append(float(value))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["model", "metric", "mean", "std_sample", "unit", "seed_count"]
        )
        for model in MODELS:
            for metric in METRICS:
                mean, std = sample_mean_std(collected[model][metric])
                unit = "ms/block" if metric.endswith("ms_per_block") else "score"
                writer.writerow(
                    [model, metric, f"{mean:.16g}", f"{std:.16g}", unit, len(SEEDS)]
                )

    print(f"Saved: {OUTPUT.relative_to(ROOT)}")
    print(f"Seeds: {', '.join(map(str, SEEDS))}")
    print("Standard deviation: sample standard deviation (ddof=1)")


if __name__ == "__main__":
    main()
