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
            raise ValueError("RetryConfig.min_seconds must be <= RetryConfig.max_seconds")


@dataclass
class ImageEncodingConfig:
    """Image token governance knobs.

    max_side:      long-edge cap for the 'global' tier (CLI: --image-max-side)
    image_format:  png | jpeg  (jpeg strongly reduces image tokens)
    jpeg_quality:  quality for jpeg encoding
    """

    max_side: int = 2048
    image_format: Literal["png", "jpeg"] = "jpeg"
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
class ExpertConfig:
    """Optional second model endpoint (model routing). Empty = routing disabled."""

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.model)


@dataclass
class AgentConfig:
    """Controller / token-elasticity / dynamic-schedule settings.

    schedule: 'fixed'  -> deterministic T+1+5N+2S+1 pipeline (11+5N with T=S=3;
                           exact V1 call pattern, default)
              'dynamic'-> controller decides the next tool call per step (budget-driven)
    """

    schedule: Literal["fixed", "dynamic"] = "fixed"
    max_tokens: int = 2000  # base per-call cap (V1 parity); doubles elastically on finish_reason=length
    max_tokens_cap: int = 16000  # hard ceiling for elastic expansion
    step3_max_tokens: int = 8000
    step2_max_tokens: int = 8000
    reasoning_max_tokens: int = 4000
    discovery_max_tokens: int = 2000
    discovery_grounding_max_tokens: int = 4000

    # dynamic schedule only
    budget_factor: float = 0.8  # B = factor * baseline_tokens_per_image
    hard_stop_factor: float = 1.5
    max_actions_per_image: int = 0  # 0 -> auto 3N+8
    controller_temperature: float = 0.1
    fallback_baseline_tokens: int = 108000

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("AgentConfig.max_tokens must be >= 1")
        if self.max_tokens_cap < self.max_tokens:
            raise ValueError("AgentConfig.max_tokens_cap must be >= max_tokens")


@dataclass
class AnalyzerConfig:
    max_workers: int = 8  # intra-image parallelism (paper: 8)
    num_get_objects: int = 3  # T: object discovery drafts
    num_summarizer_step2: int = 3  # S: summarizer step2 samples
    num_global_reasoning: int = 3  # Sg: global single-shot reasoning samples (default 3 = same as S; V1 code default: 1)
    output_mode: Literal["text", "json"] = "json"  # json = strict JSON schema steps (P0-2)
    vision_mode: Literal["full", "tiered"] = "tiered"  # tiered = global/medium/local res (P0-3A)
    grounding: bool = True  # bbox crops for local steps (P0-3B; requires output_mode=json)
    crop_verify: bool = True  # drop unreliable crops -> full image (P1-3)
    max_objects: Optional[int] = None
    no_thinking: bool = False  # Qwen3-series: chat_template_kwargs.enable_thinking=False
    image_concurrency: int = 1  # inter-image parallelism (1 = serial paper behavior)
    step_cache: bool = True  # step-level response cache (resume + dedup)

    image: ImageEncodingConfig = field(default_factory=ImageEncodingConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    expert: ExpertConfig = field(default_factory=ExpertConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    log_level: str = "INFO"
    log_to_file: bool = True

    # step -> client routing, e.g. {"summarizer_step2": "expert", "summarizer_step3": "expert"}
    route_steps: dict = field(default_factory=dict)
    base_urls: list = field(default_factory=list)  # multi-instance round-robin (inter-image throughput)

    def __post_init__(self) -> None:
        if self.max_workers < 1:
            raise ValueError("AnalyzerConfig.max_workers must be >= 1")
        if self.num_get_objects < 1:
            raise ValueError("AnalyzerConfig.num_get_objects must be >= 1")
        if self.num_summarizer_step2 < 1:
            raise ValueError("AnalyzerConfig.num_summarizer_step2 must be >= 1")
        if self.num_global_reasoning < 1:
            raise ValueError("AnalyzerConfig.num_global_reasoning must be >= 1")
        if self.max_objects is not None and self.max_objects < 1:
            raise ValueError("AnalyzerConfig.max_objects must be >= 1")
        if self.image_concurrency < 1:
            raise ValueError("AnalyzerConfig.image_concurrency must be >= 1")
        if self.grounding and self.output_mode != "json":
            raise ValueError("grounding requires --output-mode json (bboxes come from the JSON schema)")

    def route_for(self, step: str) -> str:
        if step in self.route_steps:
            return self.route_steps[step]
        for prefix, client in self.route_steps.items():
            if step.startswith(prefix):
                return client
        return "main"

    def to_dict(self, *, redact_secrets: bool = True) -> dict:
        data = asdict(self)
        if redact_secrets:
            for section in ("openai", "expert"):
                cfg = data.get(section)
                if isinstance(cfg, dict) and cfg.get("api_key"):
                    cfg["api_key"] = "***REDACTED***"
        return data
