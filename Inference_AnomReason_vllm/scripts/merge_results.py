#!/usr/bin/env python3
"""Merge inference NDJSON records into the JSON format expected by SemAP/SemF1."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anomreason_vllm.io import atomic_write_json, iter_ndjson
from anomreason_vllm.parsing import parse_anomaly_response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert one or more inference NDJSON files into pred_json for metric evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", action="append", type=Path, help="Input NDJSON file. Can be repeated.")
    parser.add_argument("--pattern", default=None, help="Glob pattern used when --input is omitted.")
    parser.add_argument("--latest", action="store_true", help="Use only the most recently modified matched file.")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output JSON path.")
    parser.add_argument("--include-errors", action="store_true", help="Include failed records as empty lists.")
    parser.add_argument("--prefer-parsed", action="store_true", help="Use record['parsed'] before reparsing response text.")
    parser.add_argument(
        "--output-format",
        choices=("parsed", "response"),
        default="parsed",
        help="Write structured parsed results or the original raw response text.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = collect_inputs(args)
    if not inputs:
        raise SystemExit("No NDJSON inputs found. Pass --input runs/.../predictions.ndjson or --pattern '*.ndjson'.")

    merged: dict[str, object] = {}
    for record in iter_ndjson(inputs):
        image_path = record.get("path")
        if not image_path:
            continue
        if record.get("status") != "ok":
            if args.include_errors:
                merged[str(image_path)] = [] if args.output_format == "parsed" else None
            continue
        if args.output_format == "response":
            merged[str(image_path)] = record.get("response")
        elif args.prefer_parsed and isinstance(record.get("parsed"), list):
            merged[str(image_path)] = record["parsed"]
        else:
            merged[str(image_path)] = parse_anomaly_response(record.get("response"))

    output = args.output or default_output_path(inputs, len(merged))
    atomic_write_json(output, merged)
    print(f"Merged {len(merged)} predictions -> {output}")
    return 0


def collect_inputs(args: argparse.Namespace) -> list[Path]:
    if args.input:
        return args.input
    pattern = args.pattern or "*.ndjson"
    paths = sorted(Path().glob(pattern), key=lambda p: p.stat().st_mtime)
    if args.latest and paths:
        return [paths[-1]]
    return paths


def default_output_path(inputs: list[Path], count: int) -> Path:
    stem = inputs[-1].name.replace(".ndjson", "")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return inputs[-1].with_name(f"{stem}_predictions_{timestamp}_num{count}.json")


if __name__ == "__main__":
    raise SystemExit(main())
