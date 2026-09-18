from __future__ import annotations

import datetime
import argparse
import os
import json
import re
import time
from typing import Optional

from .config import (
    AgentConfig,
    AnalyzerConfig,
    ExpertConfig,
    ImageEncodingConfig,
    OpenAIConfig,
    RetryConfig,
)
from .runner import ImageRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="anomagent_v2", description="AnomAgent V2: agent-ified anomaly analysis for AI-generated images.")
    parser.add_argument("--input", nargs="+", required=True, help="Image file(s) and/or directory (searched recursively).")
    parser.add_argument("--output-dir", default="outputs", help="Root directory for run artifacts.")
    parser.add_argument("--run-suffix", default="", help="Suffix for the run directory name.")
    parser.add_argument("--continue", dest="is_continue", action="store_true", help="Reuse the latest run dir for --run-suffix and skip images already completed.")
    parser.add_argument("--start", default="", help="Only process images whose basename contains this substring.")
    parser.add_argument("--end", default="", help="Only process images whose basename contains this substring.")

    parser.add_argument("--model", default=None, help="OpenAI model name (defaults to env OPENAI_MODEL or 'gpt-4o').")
    parser.add_argument("--api-key", default=None, help="OpenAI API key (defaults to env OPENAI_API_KEY).")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL (defaults to env OPENAI_BASE_URL).")

    parser.add_argument("--max-workers", type=int, default=8, help="Intra-image parallel calls (paper default 8).")
    parser.add_argument("--num-get-objects", type=int, default=3, help="T: object discovery drafts (paper default 3).")
    parser.add_argument("--num-summarizer-step2", type=int, default=3, help="S: summarizer step2 samples (paper default 3).")
    parser.add_argument("--num-global-reasoning", type=int, default=3,
                        help="Sg: global single-shot reasoning samples (default 3 = same as S; V1 code default: 1).")
    parser.add_argument("--max-objects", type=int, default=None, help="Cap the number of objects (cost knob).")

    parser.add_argument("--image-max-side", type=int, default=2048, help="Long-edge cap for the global image tier.")
    parser.add_argument("--image-format", choices=["png", "jpeg"], default="jpeg", help="Encoding format (jpeg saves image tokens).")
    parser.add_argument("--jpeg-quality", type=int, default=85, help="JPEG quality.")

    parser.add_argument("--output-mode", choices=["text", "json"], default="json", help="Step output protocol: legacy free-text or strict JSON schema.")
    parser.add_argument("--vision-mode", choices=["full", "tiered"], default="tiered", help="Image resolution tiers: full (V1) or tiered (global/medium/local).")
    parser.add_argument("--no-grounding", dest="grounding", action="store_false", help="Disable bbox grounding crops (default: on, requires json mode).")
    parser.add_argument("--no-crop-verify", dest="crop_verify", action="store_false", help="Disable the crop accuracy check (default: on).")
    parser.add_argument("--no-thinking", action="store_true", help="Qwen3-series: disable thinking mode (chat_template_kwargs.enable_thinking=False).")

    parser.add_argument("--image-concurrency", type=int, default=1, help="Inter-image parallel workers (1 = serial paper behavior).")
    parser.add_argument("--base-urls", default=None, help="Comma-separated endpoints; workers round-robin across them.")
    parser.add_argument("--no-step-cache", dest="step_cache", action="store_false", help="Disable the step-level response cache (default: on).")
    parser.add_argument("--step-cache-dir", default=None, help="Step cache directory (default: <run-dir>/step_cache).")

    parser.add_argument("--schedule", choices=["fixed", "dynamic"], default="fixed",
                        help="fixed = deterministic T+1+5N+2S+1 pipeline, same call count as V1 (default); dynamic = controller-driven tool scheduling.")
    parser.add_argument("--budget-ref-run", default=None, help="[dynamic] run dir whose ledger median tokens set the per-image budget baseline.")
    parser.add_argument("--budget-tokens", type=int, default=None, help="[dynamic] explicit per-image baseline token budget.")
    parser.add_argument("--budget-factor", type=float, default=None, help="[dynamic] budget = factor x baseline (default 0.8).")
    parser.add_argument("--max-actions", type=int, default=None, help="[dynamic] action cap (default: 3N+8).")
    parser.add_argument("--max-tokens", type=int, default=2000, help="Base per-call max_tokens; doubled elastically on finish_reason=length.")
    parser.add_argument("--max-tokens-cap", type=int, default=16000, help="Ceiling for elastic max_tokens expansion.")

    parser.add_argument("--expert-base-url", default=None, help="Optional expert-model endpoint (model routing).")
    parser.add_argument("--expert-model", default=None, help="Expert model name.")
    parser.add_argument("--expert-api-key", default=None, help="Expert endpoint API key (defaults to the main key).")
    parser.add_argument("--route-steps", default=None, help="step=client pairs, e.g. 'summarizer_step2=expert,summarizer_step3=expert' (prefix or exact match).")
    parser.add_argument("--log-level", default="INFO", help="Log level.")
    return parser


def _resolve_run_dir(output_dir: str, suffix: str, is_continue: bool) -> str:
    ts_pattern = r"\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}"
    if suffix:
        pattern = re.compile(rf"^run_({ts_pattern})_{re.escape(suffix)}$")
    else:
        pattern = re.compile(rf"^run_({ts_pattern})$")
    if is_continue and os.path.isdir(output_dir):
        candidates = []
        for name in os.listdir(output_dir):
            if not os.path.isdir(os.path.join(output_dir, name)):
                continue
            match = pattern.match(name)
            if match:
                candidates.append((name, match.group(1)))
        if candidates:
            latest = max(candidates, key=lambda x: datetime.datetime.strptime(x[1], "%Y_%m_%d_%H_%M_%S"))[0]
            return os.path.join(output_dir, latest)
    run_dir = f"{output_dir}/run_{time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())}" + (f"_{suffix}" if suffix else "")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass

    route_steps: dict = {}
    if args.route_steps:
        for pair in args.route_steps.split(","):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                raise ValueError(f"--route-steps entries must be step=client: {pair!r}")
            step, client = pair.split("=", 1)
            route_steps[step.strip()] = client.strip()

    config = AnalyzerConfig(
        max_workers=args.max_workers,
        num_get_objects=args.num_get_objects,
        num_summarizer_step2=args.num_summarizer_step2,
        num_global_reasoning=args.num_global_reasoning,
        output_mode=args.output_mode,
        vision_mode=args.vision_mode,
        grounding=args.grounding,
        crop_verify=args.crop_verify,
        max_objects=args.max_objects,
        no_thinking=args.no_thinking,
        image_concurrency=args.image_concurrency,
        step_cache=args.step_cache,
        image=ImageEncodingConfig(
            max_side=args.image_max_side,
            image_format=args.image_format,
            jpeg_quality=args.jpeg_quality,
        ),
        retry=RetryConfig(),
        openai=OpenAIConfig(
            api_key=args.api_key or os.getenv("OPENAI_API_KEY"),
            base_url=args.base_url or os.getenv("OPENAI_BASE_URL"),
            model=args.model or os.getenv("OPENAI_MODEL"),
        ),
        expert=ExpertConfig(
            api_key=args.expert_api_key,
            base_url=args.expert_base_url,
            model=args.expert_model,
        ),
        agent=AgentConfig(
            schedule=args.schedule,
            max_tokens=args.max_tokens,
            max_tokens_cap=args.max_tokens_cap,
            budget_factor=args.budget_factor if args.budget_factor is not None else 0.8,
            max_actions_per_image=args.max_actions if args.max_actions is not None else 0,
        ),
        route_steps=route_steps,
        base_urls=[u.strip() for u in args.base_urls.split(",") if u.strip()] if args.base_urls else [],
        log_level=args.log_level,
    )
    if config.openai.model is None:
        config.openai.model = "gpt-4o"

    run_dir = _resolve_run_dir(args.output_dir, args.run_suffix, args.is_continue)
    print(f"run dir: {run_dir}")
    with open(os.path.join(run_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(
            {**config.to_dict(), "run_dir": run_dir, "started_at": time.strftime("%Y-%m-%d %H:%M:%S")},
            f, ensure_ascii=False, indent=1,
        )

    runner = ImageRunner(run_dir, config, step_cache_dir=args.step_cache_dir)
    runner.run(
        args.input,
        start=args.start,
        end=args.end,
        is_continue=args.is_continue,
        budget_ref_run=args.budget_ref_run,
    )
    failed = runner.failed_files
    print(f"done: {len(runner.responses_structured)} images, {len(failed)} failed -> {run_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
