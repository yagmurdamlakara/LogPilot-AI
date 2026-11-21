"""Prompt construction for evidence-grounded HDFS explanations."""
from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = """Sen LogPilot AI için Türkçe HDFS log açıklaması üreten bir asistansın.
Sana verilen sınıflandırıcı sonucunu açıklarsın; sonucu asla değiştirmez veya yeniden hesaplamazsın.

Kurallar:
- Yalnızca kullanıcı girdisindeki log satırlarını kanıt olarak kullan.
- evidence listesine seçtiğin log satırlarını karakter karakter, birebir kopyala.
- Logda bulunmayan hata, olay, eşik, sayı, zaman veya sistem davranışı uydurma.
- Olası nedenleri kesin kök neden olarak sunma.
- Tek başına anomali etiketi veya skor, kök nedeni kanıtlamaz.
- Kanıt yetersizse limitations alanında açıkça belirt.
- recommendations alanına yalnızca logdan hareketle yapılabilecek kontrollü inceleme adımları yaz.
- Tüm açıklamayı Türkçe yaz. Teknik terimleri gerektiğinde özgün biçiminde koru.
- İstenen alanları mutlaka doldur. Bilinmeyen listeler için boş liste kullan.
- Yalnızca JSON üret; Markdown veya JSON dışında açıklama ekleme.

Alanlar:
summary: kısa açıklama.
evidence: girdiden birebir alınmış log satırları.
possible_causes: doğrulanmamış olası açıklamalar.
recommendations: kontrol/inceleme adımları.
limitations: eldeki kanıtın sınırları.
confidence: yalnızca düşük, orta veya yüksek.
"""


def build_user_prompt(
    *,
    block_id: str,
    prediction: int,
    anomaly_score: float,
    log_lines: list[str],
) -> str:
    """Build a compact, deterministic input for the model."""
    payload: dict[str, Any] = {
        "block_id": block_id,
        "classifier_prediction": {
            "value": int(prediction),
            "label": "Anomaly" if int(prediction) == 1 else "Normal",
            "anomaly_score": round(float(anomaly_score), 6),
        },
        "log_evidence": log_lines,
    }
    return (
        "Bu block için Türkçe açıklama oluştur. Sınıflandırıcı kararını aynen koru. "
        "Aşağıdaki girdiyi analiz et:\n"
        + json.dumps(payload, ensure_ascii=False)
    )
