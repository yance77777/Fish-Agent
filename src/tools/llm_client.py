"""OpenAI-compatible chat client used by Fish-Agent nodes."""
from __future__ import annotations

import base64
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


@dataclass
class LLMResponse:
    content: Any


class LLMClient:
    """Small OpenAI-compatible client.

    Configure with OPENAI_API_KEY and optionally OPENAI_BASE_URL. Doubao or
    other OpenAI-compatible gateways can be used by pointing OPENAI_BASE_URL at
    their /v1 endpoint.
    """

    def __init__(self, ctx: Any = None, api_key: str | None = None, base_url: str | None = None) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("ARK_API_KEY")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL") or "").rstrip("/")
        self.timeout = float(os.getenv("FISH_AGENT_LLM_TIMEOUT", "60"))

    def invoke(
        self,
        *,
        messages: list[Any],
        model: str,
        temperature: float = 0.2,
        max_completion_tokens: int = 1000,
    ) -> LLMResponse:
        if not self.api_key or not self.base_url:
            raise RuntimeError("LLM is not configured. Set OPENAI_API_KEY and OPENAI_BASE_URL.")

        url = self.base_url
        if not url.endswith("/chat/completions"):
            url = f"{url}/chat/completions"

        payload = {
            "model": model,
            "messages": [self._message_to_dict(message) for message in messages],
            "temperature": temperature,
            "max_tokens": max_completion_tokens,
        }
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return LLMResponse(content=data["choices"][0]["message"].get("content", ""))

    def _message_to_dict(self, message: Any) -> dict[str, Any]:
        role = getattr(message, "type", None) or getattr(message, "role", None) or "user"
        if role == "human":
            role = "user"
        if role == "ai":
            role = "assistant"
        return {"role": role, "content": self._content_to_api(getattr(message, "content", message))}

    def _content_to_api(self, content: Any) -> Any:
        if not isinstance(content, list):
            return content

        normalized: list[dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                normalized.append({"type": "text", "text": str(item)})
                continue

            if item.get("type") != "image_url":
                normalized.append(item)
                continue

            image_url = item.get("image_url", {})
            url = image_url.get("url") if isinstance(image_url, dict) else str(image_url)
            normalized.append({"type": "image_url", "image_url": {"url": self._normalize_image_url(url)}})
        return normalized

    def _normalize_image_url(self, url: str) -> str:
        if url.startswith(("http://", "https://", "data:")):
            return url

        path = Path(url)
        if not path.exists():
            return url

        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"
