"""Validation helpers for structured local-LLM explanations.

The LLM is an explanatory layer only. These helpers validate its output and
never modify or recalculate the classifier prediction.
"""
from __future__ import annotations

import json
from typing import Any


REQUIRED_FIELDS = {
    "summary",
    "evidence",
    "possible_causes",
    "recommendations",
    "limitations",
    "confidence",
}


class ExplanationValidationError(ValueError):
    """Raised when the local LLM returns an invalid explanation."""


def calibrate_confidence(model_confidence: str, evidence_count: int) -> str:
    """Apply a conservative evidence-volume ceiling to model confidence.

    One or two selected records cannot support more than low confidence.
    Three records can support at most medium confidence. High confidence is
    left possible only when at least four distinct source records were selected.
    This is a conservative guardrail, not a semantic proof of root cause.
    """
    confidence = model_confidence.strip().lower()
    if evidence_count <= 2:
        return "düşük"
    if evidence_count == 3 and confidence == "yüksek":
        return "orta"
    return confidence


def parse_explanation(raw: str, supplied_log_lines: list[str]) -> dict[str, Any]:
    """Parse output and resolve selected 1-based evidence indexes to source lines."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ExplanationValidationError(
            f"LLM yanıtı geçerli JSON değil: {exc.msg}"
        ) from exc

    if not isinstance(payload, dict):
        raise ExplanationValidationError("LLM yanıtı JSON nesnesi olmalıdır.")

    missing = REQUIRED_FIELDS.difference(payload)
    if missing:
        raise ExplanationValidationError(
            "LLM yanıtında zorunlu alanlar eksik: " + ", ".join(sorted(missing))
        )

    for field in ("summary", "confidence"):
        if not isinstance(payload[field], str) or not payload[field].strip():
            raise ExplanationValidationError(f"'{field}' alanı boş olmayan metin olmalıdır.")

    evidence_indexes = payload["evidence"]
    if not isinstance(evidence_indexes, list) or not all(
        isinstance(item, int) and not isinstance(item, bool)
        for item in evidence_indexes
    ):
        raise ExplanationValidationError(
            "'evidence' alanı 1'den başlayan tamsayı satır numaralarından oluşmalıdır."
        )

    invalid_indexes = [
        item for item in evidence_indexes
        if item < 1 or item > len(supplied_log_lines)
    ]
    if invalid_indexes:
        raise ExplanationValidationError(
            "LLM, verilen log aralığı dışında kanıt satır numarası seçti: "
            + json.dumps(invalid_indexes[:5])
        )
    if len(set(evidence_indexes)) != len(evidence_indexes):
        raise ExplanationValidationError("LLM aynı kanıt satırını birden fazla seçti.")

    for field in ("possible_causes", "recommendations", "limitations"):
        value = payload[field]
        if not isinstance(value, list) or not all(
            isinstance(item, str) and item.strip() for item in value
        ):
            raise ExplanationValidationError(
                f"'{field}' alanı boş olmayan metinlerden oluşan bir liste olmalıdır."
            )

    allowed_confidence = {"düşük", "orta", "yüksek"}
    model_confidence = payload["confidence"].strip().lower()
    if model_confidence not in allowed_confidence:
        raise ExplanationValidationError(
            "'confidence' alanı düşük, orta veya yüksek olmalıdır."
        )

    normalized = dict(payload)
    normalized["evidence"] = [
        supplied_log_lines[index - 1] for index in evidence_indexes
    ]
    normalized["model_confidence"] = model_confidence
    normalized["confidence"] = calibrate_confidence(
        model_confidence,
        len(evidence_indexes),
    )
    return {
        **{field: normalized[field] for field in REQUIRED_FIELDS},
        "model_confidence": normalized["model_confidence"],
    }
