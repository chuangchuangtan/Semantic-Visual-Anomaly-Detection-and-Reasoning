"""Step-level response cache.

Key = sha256 over (model, step, canonical messages, sampling params,
response_format, extra_body). Messages embed the image data URLs, so the
image content is covered by the hash automatically. Value = the response
payload (content, finish_reason, usage). File-per-key with atomic rename.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from typing import Any, Optional


class StepCache:
    def __init__(self, cache_dir: str, *, enabled: bool = True) -> None:
        self.cache_dir = cache_dir
        self.enabled = enabled and bool(cache_dir)
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        if self.enabled:
            os.makedirs(self.cache_dir, exist_ok=True)

    @staticmethod
    def make_key(
        *,
        model: str,
        step: str,
        messages: Any,
        max_tokens: int,
        temperature: float,
        top_p: float,
        frequency_penalty: float,
        presence_penalty: float,
        stop: Any,
        response_format: Optional[dict],
        extra_body: Optional[dict],
        sample: Optional[int] = None,
    ) -> str:
        payload = {
            "model": model,
            "step": step,
            "messages": messages,
            "sample": sample,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stop": stop,
            "response_format": response_format,
            "extra_body": extra_body,
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> str:
        # two-level fan-out to avoid one huge directory
        return os.path.join(self.cache_dir, key[:2], key + ".json")

    def get(self, key: str) -> Optional[dict]:
        if not self.enabled:
            return None
        path = self._path(key)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "content" in data:
                with self._lock:
                    self.hits += 1
                return data
        except FileNotFoundError:
            pass
        except Exception:
            pass
        with self._lock:
            self.misses += 1
        return None

    def put(self, key: str, value: dict) -> None:
        if not self.enabled:
            return
        path = self._path(key)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(value, f, ensure_ascii=False)
            os.replace(tmp, path)
        except Exception:
            tmp = locals().get("tmp")
            if isinstance(tmp, str) and os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except Exception:
                    pass

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses, "dir": self.cache_dir}
