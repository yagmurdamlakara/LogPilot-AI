"""Regression tests for evidence and confidence guidance in the LLM prompt."""
import json

from src.llm.prompts import SYSTEM_PROMPT, build_user_prompt


def test_prompt_calibrates_confidence_from_log_evidence_not_classifier_score():
    assert "yalnızca sınıflandırıcı skoruna göre belirleme" in SYSTEM_PROMPT
    assert "log kanıtlarıyla ne ölçüde desteklendiğine göre belirle" in SYSTEM_PROMPT
    assert "log satırı varsa confidence değerini düşük seç" in SYSTEM_PROMPT
    assert "confidence değerini orta seç" in SYSTEM_PROMPT
    assert "Yüksek güveni yalnızca birden fazla açık" in SYSTEM_PROMPT
    assert "genel heartbeat, INFO veya rutin durum mesajları" in SYSTEM_PROMPT


def test_prompt_prevents_definitive_root_cause_claims():
    assert "daima olasılık olarak ifade et" in SYSTEM_PROMPT
    assert "kesin kök neden iddiası anlamına gelmez" in SYSTEM_PROMPT


def test_prompt_limits_recommendations_to_evidence_based_checks():
    assert "kanıtsız yapılandırma değişikliği veya sistem güncellemesi önerme" in SYSTEM_PROMPT


def test_user_prompt_numbers_source_evidence_and_keeps_classifier_result():
    supplied = [
        "synthetic fixture: block operation started",
        "synthetic fixture: block operation completed",
    ]
    prompt = build_user_prompt(
        block_id="blk_quality_fixture",
        prediction=1,
        anomaly_score=0.99,
        log_lines=supplied,
    )

    payload_text = prompt.split("\n", 1)[1]
    payload = json.loads(payload_text)

    assert payload["log_evidence"] == [
        {"line_number": 1, "text": supplied[0]},
        {"line_number": 2, "text": supplied[1]},
    ]
    assert payload["classifier_prediction"]["value"] == 1
    assert payload["classifier_prediction"]["anomaly_score"] == 0.99
