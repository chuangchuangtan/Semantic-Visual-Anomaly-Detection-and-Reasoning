#!/usr/bin/env python3
"""Run batch semantic-anomaly inference against vLLM OpenAI-compatible servers."""

from __future__ import annotations

import argparse
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anomreason_vllm.client import OpenAICompatClient
from anomreason_vllm.dataset import extract_image_references, resolve_image_paths
from anomreason_vllm.io import append_ndjson, atomic_write_json, completed_paths, load_json, strip_ndjson_prefix
from anomreason_vllm.parsing import DEFAULT_PROMPT, parse_anomaly_response

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch inference for AnomReason-style semantic anomaly detection with vLLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--test-json", required=True, type=Path, help="Dataset JSON to evaluate.")
    parser.add_argument(
        "--image-root",
        "--projection-root",
        dest="image_root",
        required=True,
        type=Path,
        help="Directory containing evaluation images.",
    )
    parser.add_argument(
        "--path-mode",
        choices=("basename", "relative", "absolute"),
        default="relative",
        help="How dataset image references are resolved against --image-root.",
    )
    parser.add_argument("--out", required=True, help="Output NDJSON path. The legacy ndjson: prefix is accepted.")
    parser.add_argument("--error-out", default=None, help="Optional NDJSON path for failed records.")
    parser.add_argument("--missing-out", default=None, type=Path, help="Optional JSON report for missing images.")
    parser.add_argument("--model", required=True, help="Model id reported by vLLM.")
    parser.add_argument("--base-port", type=int, default=8000, help="First vLLM OpenAI-compatible port.")
    parser.add_argument("--instances", "--gpus", dest="instances", type=int, default=1, help="Number of vLLM instances.")
    parser.add_argument("--api-key", default="EMPTY", help="Dummy key accepted by most local vLLM servers.")
    parser.add_argument("--concurrency", type=int, default=4, help="Maximum concurrent image requests.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per image.")
    parser.add_argument("--timeout", type=float, default=600.0, help="HTTP request timeout in seconds.")
    parser.add_argument("--resume", action="store_true", help="Skip successful records already present in --out.")
    parser.add_argument("--resume-all", action="store_true", help="Skip all records already present in --out, including errors.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle image order before inference.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of images for smoke tests.")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--prompt-file", type=Path, default=None, help="Custom prompt text file.")
    parser.add_argument(
        "--raw-response-only",
        action="store_true",
        help="Do not parse model responses; keep only the original response text in outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.instances < 1:
        raise SystemExit("--instances must be >= 1.")
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1.")
    out_path = strip_ndjson_prefix(args.out)
    error_path = strip_ndjson_prefix(args.error_out) if args.error_out else None
    prompt = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else DEFAULT_PROMPT

    dataset = load_json(args.test_json)
    references = extract_image_references(dataset)
    images, missing = resolve_image_paths(references, args.image_root, args.path_mode, skip_missing=True)
    if args.missing_out:
        atomic_write_json(args.missing_out, missing)

    if args.shuffle:
        random.shuffle(images)
    if args.limit is not None:
        images = images[: args.limit]

    done: set[str] = set()
    if args.resume or args.resume_all:
        done = completed_paths(out_path, success_only=not args.resume_all)
    todo = [path for path in images if path not in done]

    print(
        f"Images: {len(images)} | Missing: {len(missing)} | Done: {len(done)} | Remaining: {len(todo)}",
        flush=True,
    )
    if missing and not args.missing_out:
        print("Tip: pass --missing-out runs/.../missing_images.json to keep a missing-image report.", flush=True)
    if not todo:
        return 0

    endpoints = [f"http://127.0.0.1:{args.base_port + i}/v1" for i in range(args.instances)]

    failures = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(
                run_with_retries,
                base_url=endpoints[index % len(endpoints)],
                image_path=image_path,
                prompt=prompt,
                args=args,
            ): image_path
            for index, image_path in enumerate(todo)
        }
        iterator = as_completed(futures)
        if tqdm:
            iterator = tqdm(iterator, total=len(futures), desc="Infer", unit="img")

        for index, future in enumerate(iterator, start=1):
            image_path = futures[future]
            try:
                append_ndjson(out_path, future.result())
            except Exception as exc:  # keep long jobs crash-resistant
                failures += 1
                record = {
                    "status": "error",
                    "path": image_path,
                    "parsed": None if args.raw_response_only else [],
                    "response": None,
                    "error": str(exc),
                }
                append_ndjson(error_path or out_path, record)
                if not tqdm:
                    print(f"FAIL ({index}/{len(futures)}): {image_path} | {exc}", flush=True)
            else:
                if not tqdm:
                    print(f"OK ({index}/{len(futures)}): {image_path}", flush=True)

    print(f"Finished. Output: {out_path} | Failures: {failures}", flush=True)
    return 1 if failures else 0


def run_with_retries(base_url: str, image_path: str, prompt: str, args: argparse.Namespace) -> dict[str, Any]:
    delay = 2.0
    last_error: Exception | None = None
    for attempt in range(1, args.max_retries + 1):
        try:
            client = OpenAICompatClient(
                base_url=base_url,
                api_key=args.api_key,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )
            response_text = client.chat_with_image(image_path, prompt)
            return {
                "status": "ok",
                "path": image_path,
                "parsed": None if args.raw_response_only else parse_anomaly_response(response_text),
                "response": response_text,
                "base_url": base_url,
            }
        except Exception as exc:
            last_error = exc
            if attempt < args.max_retries:
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
    raise RuntimeError(f"{last_error}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
