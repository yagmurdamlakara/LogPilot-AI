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


def parse_explanation(raw: str, supplied_log_lines: list[str]) -> dict[str, Any]:
    """Parse and validate a JSON explanation against the supplied log evidence."""
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

    for field in ("evidence", "possible_causes", "recommendations", "limitations"):
        value = payload[field]
        if not isinstance(value, list) or not all(
            isinstance(item, str) and item.strip() for item in value
        ):
            raise ExplanationValidationError(
                f"'{field}' alanı boş olmayan metinlerden oluşan bir liste olmalıdır."
            )

    # Evidence must be copied verbatim from the supplied records. This prevents
    # invented or paraphrased log lines from being presented as source evidence.
    allowed_lines = set(supplied_log_lines)
    invalid_evidence = [
        item for item in payload["evidence"] if item not in allowed_lines
    ]
    if invalid_evidence:
        # Include a bounded diagnostic excerpt so evaluation reports reveal
        # whether the model paraphrased, decorated, or invented a log record.
        diagnostic = [item[:240] for item in invalid_evidence[:3]]
        raise ExplanationValidationError(
            "LLM, kendisine verilmeyen veya birebir eşleşmeyen log kanıtı üretti. "
            f"Eşleşmeyen kanıt örnekleri: {json.dumps(diagnostic, ensure_ascii=False)}"
        )

    allowed_confidence = {"düşük", "orta", "yüksek"}
    if payload["confidence"].strip().lower() not in allowed_confidence:
        raise ExplanationValidationError(
            "'confidence' alanı düşük, orta veya yüksek olmalıdır."
        )

    # Return only the documented fields; ignore any extra model-generated keys.
    return {field: payload[field] for field in REQUIRED_FIELDS}
