"""Create structured explanations for existing LogPilot predictions."""
from __future__ import annotations

from typing import Any

from src.llm.client import OllamaClient
from src.llm.prompts import SYSTEM_PROMPT, build_user_prompt
from src.llm.schemas import parse_explanation


class HDFSLogAnalyzer:
    """Explain a fixed classifier result using bounded raw log evidence."""

    def __init__(self, client: OllamaClient | None = None) -> None:
        self.client = client or OllamaClient()

    def explain_block(
        self,
        *,
        block_id: str,
        prediction: int,
        anomaly_score: float,
        log_lines: list[str],
    ) -> dict[str, Any]:
        if not block_id.strip():
            raise ValueError("block_id boş olamaz.")
        if prediction not in (0, 1):
            raise ValueError("prediction yalnızca 0 veya 1 olabilir.")
        if not 0.0 <= float(anomaly_score) <= 1.0:
            raise ValueError("anomaly_score 0 ile 1 arasında olmalıdır.")
        if not log_lines:
            raise ValueError("Açıklama için en az bir ham log satırı gereklidir.")

        # Bound prompt size to avoid accidentally sending an entire large log.
        bounded_lines = [line.strip() for line in log_lines[:100] if line.strip()]
        if not bounded_lines:
            raise ValueError("Geçerli ham log satırı bulunamadı.")

        user_prompt = build_user_prompt(
            block_id=block_id,
            prediction=prediction,
            anomaly_score=anomaly_score,
            log_lines=bounded_lines,
        )
        raw_response = self.client.generate_json(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        explanation = parse_explanation(raw_response, bounded_lines)

        # Preserve classifier output separately so the UI can display it unchanged.
        explanation["classifier_prediction"] = int(prediction)
        explanation["classifier_label"] = "Anomaly" if prediction == 1 else "Normal"
        explanation["classifier_anomaly_score"] = float(anomaly_score)
        explanation["block_id"] = block_id
        return explanation
