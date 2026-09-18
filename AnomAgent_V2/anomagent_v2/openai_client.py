from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from openai import OpenAI

from .config import OpenAIConfig, RetryConfig
from .step_cache import StepCache


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, finish_reason, content):
        self.finish_reason = finish_reason
        self.message = _Msg(content)


class _CachedResponse:
    """Minimal response shim for cache hits (content + usage + finish_reason)."""

    from_cache = True

    def __init__(self, payload):
        self._payload = payload
        self.usage = payload.get("usage")
        self.choices = [_Choice(payload.get("finish_reason"), payload.get("content"))]

    def to_dict(self):
        return {
            "choices": [{"finish_reason": self._payload.get("finish_reason")}],
            "usage": self._payload.get("usage"),
        }


@dataclass
class UsageTotals:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def add(self, *, prompt_tokens: Optional[int], completion_tokens: Optional[int], total_tokens: Optional[int]) -> None:
        with self._lock:
            if isinstance(prompt_tokens, int):
                self.prompt_tokens += prompt_tokens
            if isinstance(completion_tokens, int):
                self.completion_tokens += completion_tokens
            if isinstance(total_tokens, int):
                self.total_tokens += total_tokens

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


def _extract_usage(response_obj: Any) -> Optional[dict]:
    usage: Optional[dict] = None
    try:
        response_dict = response_obj.to_dict()
        if isinstance(response_dict, dict):
            usage = response_dict.get("usage")
    except Exception:
        usage = None

    if usage is None:
        usage_obj = getattr(response_obj, "usage", None)
        if usage_obj is not None:
            try:
                usage = usage_obj.to_dict()
            except Exception:
                try:
                    usage = usage_obj.model_dump()
                except Exception:
                    usage = None

    return usage if isinstance(usage, dict) else None


class OpenAIChatClient:
    def __init__(
        self,
        *,
        openai_config: OpenAIConfig,
        retry_config: RetryConfig,
        logger: Optional[logging.Logger] = None,
        extra_body: Optional[dict] = None,
        on_call: Optional[Callable[[dict], None]] = None,
        step_cache: Optional[StepCache] = None,
    ) -> None:
        if not openai_config.api_key:
            raise ValueError("Missing OpenAI API key. Set OPENAI_API_KEY or pass api_key explicitly.")

        kwargs: dict[str, Any] = {"api_key": openai_config.api_key}
        if openai_config.base_url:
            kwargs["base_url"] = openai_config.base_url

        self.client = OpenAI(**kwargs)
        self.extra_body = extra_body
        self.on_call = on_call
        self.step_cache = step_cache
        self.model = openai_config.model or "gpt-4o"
        self.retry = retry_config
        self.logger = logger or logging.getLogger(__name__)
        self.usage = UsageTotals()

    def create_chat_completion(
        self,
        *,
        messages,
        max_tokens: int,
        temperature: float,
        top_p: float,
        frequency_penalty: float,
        presence_penalty: float,
        stop=None,
        step: str = "unknown",
        response_format: Optional[dict] = None,
        sample: Optional[int] = None,
    ):
        cache_key: Optional[str] = None
        if self.step_cache is not None and self.step_cache.enabled:
            cache_key = self.step_cache.make_key(
                model=self.model,
                step=step,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop,
                response_format=response_format,
                extra_body=self.extra_body,
                sample=sample,
            )
            cached = self.step_cache.get(cache_key)
            if cached is not None:
                self._notify_call(
                    step=step,
                    response=_CachedResponse(cached),
                    attempt=0,
                    status="cache_hit",
                    latency_ms=0.0,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    error=None,
                )
                return _CachedResponse(cached)

        t0 = time.perf_counter()
        last_error: Optional[Exception] = None
        for attempt in range(1, self.retry.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stop=stop,
                    stream=False,
                    **({"response_format": response_format} if response_format else {}),
                    **({"extra_body": self.extra_body} if self.extra_body else {}),
                )
                self._accumulate_usage(response)
                if cache_key is not None:
                    content = None
                    finish_reason = None
                    try:
                        content = response.choices[0].message.content
                        finish_reason = response.choices[0].finish_reason
                    except Exception:
                        pass
                    self.step_cache.put(
                        cache_key,
                        {
                            "content": content,
                            "finish_reason": finish_reason,
                            "usage": _extract_usage(response) or {},
                            "ts": time.time(),
                        },
                    )
                self._notify_call(
                    step=step,
                    response=response,
                    attempt=attempt,
                    status="ok",
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    error=None,
                )
                return response
            except Exception as e:
                last_error = e
                self._notify_call(
                    step=step,
                    response=None,
                    attempt=attempt,
                    status="error",
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    error=str(e)[:300],
                )
                if attempt >= self.retry.max_retries:
                    raise

                sleep_seconds = min(
                    self.retry.max_seconds,
                    self.retry.min_seconds * (2 ** (attempt - 1)),
                )
                sleep_seconds += random.uniform(0, self.retry.jitter_seconds)
                self.logger.warning(
                    "OpenAI request failed (attempt %s/%s): %s. Retrying in %.1fs...",
                    attempt,
                    self.retry.max_retries,
                    e,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)

        raise last_error  # type: ignore[misc]

    def _notify_call(
        self,
        *,
        step: str,
        response: Any,
        attempt: int,
        status: str,
        latency_ms: float,
        max_tokens: int,
        temperature: float,
        error: Optional[str],
    ) -> None:
        if self.on_call is None:
            return
        usage: dict = {}
        if response is not None:
            usage = _extract_usage(response) or {}
        finish_reason = None
        try:
            if response is not None and response.choices:
                finish_reason = response.choices[0].finish_reason
        except Exception:
            finish_reason = None
        meta = {
            "model": self.model,
            "step": step,
            "attempt": attempt,
            "status": status,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "finish_reason": finish_reason,
            "latency_ms": round(latency_ms, 1),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "error": error,
        }
        try:
            self.on_call(meta)
        except Exception:
            pass

    def _accumulate_usage(self, response_obj: Any) -> None:
        usage = _extract_usage(response_obj)
        if usage is None:
            return

        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        if prompt_tokens is None:
            prompt_tokens = usage.get("input_tokens")
        if completion_tokens is None:
            completion_tokens = usage.get("output_tokens")
        if total_tokens is None and isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
            total_tokens = prompt_tokens + completion_tokens

        self.usage.add(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    def reset_usage(self) -> None:
        self.usage = UsageTotals()
