"""Minimal standard-library client for the local Ollama chat API."""
from __future__ import annotations

import json
import urllib.error
import urllib.request


class OllamaError(RuntimeError):
    """Raised when Ollama is unavailable or returns an invalid response."""


EXPLANATION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "evidence": {
            "type": "array",
            "items": {"type": "integer", "minimum": 1},
        },
        "possible_causes": {"type": "array", "items": {"type": "string"}},
        "recommendations": {"type": "array", "items": {"type": "string"}},
        "limitations": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["düşük", "orta", "yüksek"]},
    },
    "required": [
        "summary",
        "evidence",
        "possible_causes",
        "recommendations",
        "limitations",
        "confidence",
    ],
    "additionalProperties": False,
}


class OllamaClient:
    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3:4b",
        timeout: int = 300,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate_json(self, *, system_prompt: str, user_prompt: str) -> str:
        """Request a schema-constrained JSON response from local Ollama."""
        body = {
            "model": self.model,
            "stream": False,
            "format": EXPLANATION_JSON_SCHEMA,
            "think": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 512,
                "num_ctx": 4096,
            },
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OllamaError(
                f"Ollama HTTP hatası ({exc.code}): {detail[:1000]}"
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise OllamaError(
                "Ollama servisine bağlanılamadı veya zaman aşımı oluştu. "
                "Ollama'nın çalıştığını ve modelin yüklü olduğunu kontrol edin."
            ) from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise OllamaError("Ollama geçerli bir API yanıtı döndürmedi.") from exc

        message = result.get("message") or {}
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            thinking = message.get("thinking")
            detail = (
                f" Modelin thinking alanı: {thinking[:300]}"
                if isinstance(thinking, str) and thinking else ""
            )
            raise OllamaError("Ollama boş bir içerik döndürdü." + detail)
        return content
