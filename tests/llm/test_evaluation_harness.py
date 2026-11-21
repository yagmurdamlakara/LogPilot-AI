"""Offline tests for the LLM quality evaluation harness."""
from pathlib import Path

from tests.llm.evaluation.evaluate_llm import (
    evaluate_scenario,
    load_scenarios,
    run_evaluation,
)


SCENARIOS_PATH = (
    Path(__file__).parent / "evaluation" / "scenarios.json"
)


class FakeAnalyzer:
    def __init__(self, *, confidence="düşük", prediction_override=None):
        self.confidence = confidence
        self.prediction_override = prediction_override

    def explain_block(self, *, block_id, prediction, anomaly_score, log_lines):
        return {
            "confidence": self.confidence,
            "classifier_prediction": (
                prediction if self.prediction_override is None
                else self.prediction_override
            ),
            "classifier_anomaly_score": float(anomaly_score),
            "block_id": block_id,
            "evidence": [log_lines[0]],
            "summary": "Sentetik senaryo açıklaması.",
            "possible_causes": [],
            "recommendations": [],
            "limitations": ["Bu test sahte analyzer kullanır."],
        }


def test_scenario_file_contains_five_unique_valid_fixtures():
    scenarios = load_scenarios(SCENARIOS_PATH)

    assert len(scenarios) == 5
    assert len({item["id"] for item in scenarios}) == 5
    assert {item["prediction"] for item in scenarios} == {0, 1}


def test_evaluation_records_confidence_expectation_and_classifier_integrity():
    scenario = load_scenarios(SCENARIOS_PATH)[0]

    result = evaluate_scenario(FakeAnalyzer(confidence="düşük"), scenario)

    assert result["status"] == "completed"
    assert result["classifier_unchanged"] is True
    assert result["confidence_in_expected_range"] is True


def test_evaluation_flags_unexpected_confidence_without_hiding_result():
    scenario = load_scenarios(SCENARIOS_PATH)[0]

    result = evaluate_scenario(FakeAnalyzer(confidence="yüksek"), scenario)

    assert result["status"] == "completed"
    assert result["confidence_in_expected_range"] is False
    assert result["confidence"] == "yüksek"


def test_evaluation_flags_classifier_output_mutation():
    scenario = load_scenarios(SCENARIOS_PATH)[0]

    result = evaluate_scenario(
        FakeAnalyzer(confidence="düşük", prediction_override=0),
        scenario,
    )

    assert result["classifier_unchanged"] is False


def test_batch_evaluation_continues_after_a_scenario_error():
    class FailingAnalyzer:
        def explain_block(self, **kwargs):
            raise RuntimeError("simulated local model failure")

    scenarios = load_scenarios(SCENARIOS_PATH)[:2]
    report = run_evaluation(scenarios, analyzer=FailingAnalyzer())

    assert report["scenario_count"] == 2
    assert report["completed_count"] == 0
    assert report["error_count"] == 2
    assert all(item["status"] == "error" for item in report["results"])
