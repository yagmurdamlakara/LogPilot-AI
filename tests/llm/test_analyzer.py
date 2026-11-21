"""Automated tests for the LLM explanation layer.

These tests use a fake client and never require a running Ollama service.
"""
import json

import pytest

from src.llm.analyzer import HDFSLogAnalyzer
from src.llm.schemas import ExplanationValidationError, parse_explanation


LOG_LINES = [
    "081111 085914 27 INFO dfs.FSNamesystem: BLOCK* NameSystem.delete:",
    "blk_-1030832046197982436 is added to invalidSet",
]


def valid_payload(evidence=None):
    return {
        "summary": "Loglarda blok geçersiz kümesine eklenmiş görünüyor.",
        "evidence": evidence if evidence is not None else [2],
        "possible_causes": ["Blok geçersizleştirme işlemi; nedeni mevcut satırlardan kesinleştirilemez."],
        "recommendations": ["İlgili block için önceki ve sonraki log kayıtlarını incele."],
        "limitations": ["Verilen iki satır tek başına kök nedeni göstermiyor."],
        "confidence": "düşük",
    }


class FakeLLMClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate_json(self, *, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        return self.response


def test_analyzer_preserves_classifier_result_and_block_id():
    client = FakeLLMClient(json.dumps(valid_payload(), ensure_ascii=False))
    analyzer = HDFSLogAnalyzer(client=client)

    result = analyzer.explain_block(
        block_id="blk_-1030832046197982436",
        prediction=1,
        anomaly_score=0.87,
        log_lines=LOG_LINES,
    )

    assert result["classifier_prediction"] == 1
    assert result["classifier_label"] == "Anomaly"
    assert result["classifier_anomaly_score"] == pytest.approx(0.87)
    assert result["block_id"] == "blk_-1030832046197982436"
    assert result["evidence"] == [LOG_LINES[1]]
    assert len(client.calls) == 1


def test_analyzer_preserves_normal_prediction():
    client = FakeLLMClient(json.dumps(valid_payload(), ensure_ascii=False))
    analyzer = HDFSLogAnalyzer(client=client)

    result = analyzer.explain_block(
        block_id="blk_test",
        prediction=0,
        anomaly_score=0.12,
        log_lines=LOG_LINES,
    )

    assert result["classifier_prediction"] == 0
    assert result["classifier_label"] == "Normal"
    assert result["classifier_anomaly_score"] == pytest.approx(0.12)


def test_analyzer_rejects_out_of_range_evidence_index():
    payload = valid_payload(evidence=[3])
    analyzer = HDFSLogAnalyzer(
        client=FakeLLMClient(json.dumps(payload, ensure_ascii=False))
    )

    with pytest.raises(ExplanationValidationError, match="aralığı dışında"):
        analyzer.explain_block(
            block_id="blk_test",
            prediction=1,
            anomaly_score=0.9,
            log_lines=LOG_LINES,
        )


def test_analyzer_rejects_missing_required_fields():
    payload = valid_payload()
    del payload["possible_causes"]
    analyzer = HDFSLogAnalyzer(
        client=FakeLLMClient(json.dumps(payload, ensure_ascii=False))
    )

    with pytest.raises(ExplanationValidationError, match="zorunlu alanlar eksik"):
        analyzer.explain_block(
            block_id="blk_test",
            prediction=1,
            anomaly_score=0.9,
            log_lines=LOG_LINES,
        )


@pytest.mark.parametrize(
    ("prediction", "score", "lines"),
    [
        (2, 0.5, LOG_LINES),
        (1, -0.1, LOG_LINES),
        (1, 1.1, LOG_LINES),
        (1, 0.5, []),
    ],
)
def test_analyzer_rejects_invalid_input_before_calling_client(prediction, score, lines):
    client = FakeLLMClient(json.dumps(valid_payload(), ensure_ascii=False))
    analyzer = HDFSLogAnalyzer(client=client)

    with pytest.raises(ValueError):
        analyzer.explain_block(
            block_id="blk_test",
            prediction=prediction,
            anomaly_score=score,
            log_lines=lines,
        )

    assert client.calls == []


def test_parse_explanation_rejects_invalid_json():
    with pytest.raises(ExplanationValidationError, match="geçerli JSON değil"):
        parse_explanation("{not-json", LOG_LINES)


def test_parse_explanation_rejects_invalid_confidence():
    payload = valid_payload()
    payload["confidence"] = "kesin"

    with pytest.raises(ExplanationValidationError, match="confidence"):
        parse_explanation(json.dumps(payload, ensure_ascii=False), LOG_LINES)
