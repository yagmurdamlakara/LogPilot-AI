"""Run real Ollama-backed LLM explanation quality scenarios.

The bundled scenarios are synthetic behavioral fixtures, not real HDFS incidents
and not a substitute for the project's human review of 15-20 anomalous blocks.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.llm.analyzer import HDFSLogAnalyzer


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCENARIOS = Path(__file__).with_name("scenarios.json")
DEFAULT_OUTPUT = ROOT / "reports" / "llm_evaluation" / "latest.json"


def load_scenarios(path: Path) -> list[dict[str, Any]]:
    """Load and minimally validate the scenario fixture file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    scenarios = payload.get("scenarios") if isinstance(payload, dict) else None
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("Scenario file must contain a non-empty 'scenarios' list.")

    required = {
        "id", "block_id", "prediction", "anomaly_score",
        "log_lines", "expected_confidence",
    }
    seen_ids: set[str] = set()
    for index, scenario in enumerate(scenarios):
        if not isinstance(scenario, dict) or not required.issubset(scenario):
            raise ValueError(f"Scenario at index {index} is missing required fields.")
        if scenario["id"] in seen_ids:
            raise ValueError(f"Duplicate scenario id: {scenario['id']}")
        seen_ids.add(scenario["id"])
        if scenario["prediction"] not in (0, 1):
            raise ValueError(f"Invalid prediction in scenario {scenario['id']}.")
        if not isinstance(scenario["log_lines"], list) or not scenario["log_lines"]:
            raise ValueError(f"Scenario {scenario['id']} must contain log_lines.")
        if not isinstance(scenario["expected_confidence"], list):
            raise ValueError(f"Scenario {scenario['id']} expected_confidence must be a list.")
    return scenarios


def evaluate_scenario(
    analyzer: HDFSLogAnalyzer,
    scenario: dict[str, Any],
) -> dict[str, Any]:
    """Run one scenario and record behavior without silently hiding failures."""
    started = time.perf_counter()
    result: dict[str, Any] = {
        "scenario_id": scenario["id"],
        "description": scenario.get("description", ""),
        "purpose": scenario.get("purpose", ""),
        "status": "error",
    }
    try:
        explanation = analyzer.explain_block(
            block_id=scenario["block_id"],
            prediction=scenario["prediction"],
            anomaly_score=scenario["anomaly_score"],
            log_lines=scenario["log_lines"],
        )
        confidence = explanation["confidence"].strip().lower()
        expected_confidence = scenario["expected_confidence"]
        classifier_unchanged = (
            explanation["classifier_prediction"] == scenario["prediction"]
            and explanation["classifier_anomaly_score"] == float(scenario["anomaly_score"])
            and explanation["block_id"] == scenario["block_id"]
        )
        confidence_in_range = confidence in expected_confidence
        classifier_label = explanation.get(
            "classifier_label",
            "Anomaly" if scenario["prediction"] == 1 else "Normal",
        )

        result.update({
            "status": "completed",
            "classifier_unchanged": classifier_unchanged,
            "confidence": confidence,
            "expected_confidence": expected_confidence,
            "confidence_in_expected_range": confidence_in_range,
            "evidence": explanation["evidence"],
            "summary": explanation["summary"],
            "possible_causes": explanation["possible_causes"],
            "recommendations": explanation["recommendations"],
            "limitations": explanation["limitations"],
            "classifier_prediction": explanation["classifier_prediction"],
            "classifier_label": classifier_label,
            "classifier_anomaly_score": explanation["classifier_anomaly_score"],
        })
    except Exception as exc:  # Preserve a failed scenario and continue the batch.
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["latency_seconds"] = round(time.perf_counter() - started, 3)
    return result


def run_evaluation(
    scenarios: list[dict[str, Any]],
    *,
    analyzer: HDFSLogAnalyzer | None = None,
) -> dict[str, Any]:
    """Evaluate all scenarios sequentially using one analyzer/client."""
    analyzer = analyzer or HDFSLogAnalyzer()
    results = [evaluate_scenario(analyzer, item) for item in scenarios]
    completed = [item for item in results if item["status"] == "completed"]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scenario_count": len(results),
        "completed_count": len(completed),
        "error_count": len(results) - len(completed),
        "classifier_preservation_pass_count": sum(
            item.get("classifier_unchanged") is True for item in completed
        ),
        "confidence_expectation_pass_count": sum(
            item.get("confidence_in_expected_range") is True for item in completed
        ),
        "results": results,
        "note": (
            "Synthetic scenario results are behavioral checks only. "
            "They are not real HDFS incident labels or human-rated explanation quality."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    try:
        scenarios = load_scenarios(args.scenarios)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"Scenario loading failed: {exc}", file=sys.stderr)
        return 2

    report = run_evaluation(scenarios)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=== LOGPILOT LLM QUALITY EVALUATION ===")
    print(f"Scenarios: {report['scenario_count']}")
    print(f"Completed: {report['completed_count']}")
    print(f"Errors: {report['error_count']}")
    print(
        "Classifier preserved: "
        f"{report['classifier_preservation_pass_count']}/{report['completed_count']}"
    )
    print(
        "Confidence expectation met: "
        f"{report['confidence_expectation_pass_count']}/{report['completed_count']}"
    )
    for item in report["results"]:
        status = item["status"].upper()
        confidence_status = (
            "confidence=OK"
            if item.get("confidence_in_expected_range")
            else f"confidence={item.get('confidence', 'N/A')} "
                 f"(expected {item.get('expected_confidence', [])})"
        )
        print(f"- {item['scenario_id']}: {status}; {confidence_status}")
        if item.get("error"):
            print(f"  Error: {item['error']}")
    print(f"Report: {args.output}")
    return 0 if report["error_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
