from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, Optional


@dataclass
class RetryConfig:
    max_retries: int = 3
    min_seconds: float = 1.0
    max_seconds: float = 20.0
    jitter_seconds: float = 0.5

    def __post_init__(self) -> None:
        if self.max_retries < 1:
            raise ValueError("RetryConfig.max_retries must be >= 1")
        if self.min_seconds < 0 or self.max_seconds < 0 or self.jitter_seconds < 0:
            raise ValueError("RetryConfig seconds must be >= 0")
        if self.min_seconds > self.max_seconds:
            raise ValueError("RetryConfig.min_seconds must be <= max_seconds")


@dataclass
class ImageEncodingConfig:
    max_side: int = 2048
    image_format: Literal["png", "jpeg"] = "png"
    jpeg_quality: int = 85

    def __post_init__(self) -> None:
        if self.max_side < 1:
            raise ValueError("ImageEncodingConfig.max_side must be >= 1")
        if self.jpeg_quality < 1 or self.jpeg_quality > 100:
            raise ValueError("ImageEncodingConfig.jpeg_quality must be in [1, 100]")


@dataclass
class OpenAIConfig:
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None


@dataclass
class AnalyzerConfig:
    max_workers: int = 8
    num_get_objects: int = 3
    num_summarizer_step2: int = 3
    max_objects: Optional[int] = None

    image: ImageEncodingConfig = field(default_factory=ImageEncodingConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)

    log_level: str = "INFO"
    log_to_file: bool = True

    def __post_init__(self) -> None:
        if self.max_workers < 1:
            raise ValueError("AnalyzerConfig.max_workers must be >= 1")
        if self.num_get_objects < 1:
            raise ValueError("AnalyzerConfig.num_get_objects must be >= 1")
        if self.num_summarizer_step2 < 1:
            raise ValueError("AnalyzerConfig.num_summarizer_step2 must be >= 1")
        if self.max_objects is not None and self.max_objects < 1:
            raise ValueError("AnalyzerConfig.max_objects must be >= 1")

    def to_dict(self, *, redact_secrets: bool = True) -> dict:
        data = asdict(self)
        if redact_secrets:
            openai = data.get("openai")
            if isinstance(openai, dict) and openai.get("api_key"):
                openai["api_key"] = "***REDACTED***"
        return data
