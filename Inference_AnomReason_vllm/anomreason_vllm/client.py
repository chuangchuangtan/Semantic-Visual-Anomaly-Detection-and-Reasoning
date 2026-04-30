"""OpenAI-compatible client for vLLM chat-completions endpoints."""

from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from PIL import Image

class OpenAICompatClient:
    """Minimal client for vLLM's OpenAI-compatible `/v1/chat/completions` API."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: float = 600.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )

    def chat_with_image(self, image_path: str | Path, prompt: str) -> str:
        image_b64 = encode_image_to_base64(image_path)
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    ],
                }
            ],
        }
        response = self.session.post(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload),
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()


# def encode_image_to_base64(path: str | Path) -> str:
#     with Path(path).open("rb") as f:
#         return base64.b64encode(f.read()).decode("ascii")

def encode_image_to_base64(
    path: str | Path,
    max_size: int = 1024,
    quality: int = 85,
) -> str:
    path = Path(path)
    with Image.open(path) as img:
        width, height = img.size
        if max(width, height) <= max_size and img.format == "JPEG":
            return base64.b64encode(path.read_bytes()).decode("ascii")
        img = img.convert("RGB")
        if max(width, height) > max_size:
            img.thumbnail((max_size, max_size))
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")