from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Optional

from openai import OpenAI

from .config import OpenAIConfig, RetryConfig


@dataclass
class UsageTotals:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, *, prompt_tokens: Optional[int], completion_tokens: Optional[int], total_tokens: Optional[int]) -> None:
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
    ) -> None:
        if not openai_config.api_key:
            raise ValueError("Missing OpenAI API key. Set OPENAI_API_KEY or pass api_key explicitly.")

        kwargs: dict[str, Any] = {"api_key": openai_config.api_key}
        if openai_config.base_url:
            kwargs["base_url"] = openai_config.base_url

        self.client = OpenAI(**kwargs)
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
    ):
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
                )
                self._accumulate_usage(response)
                return response
            except Exception as e:
                last_error = e
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
