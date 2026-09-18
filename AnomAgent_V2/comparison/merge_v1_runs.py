#!/usr/bin/env python3
"""Merge V1 run dirs produced by disjoint parallel processes into one.

V1 (AnomAgent @ 97fd9cf) processes images serially inside a single process.
To reproduce 20 images faster without touching V1 code behavior, launch 4
processes over disjoint 5-image subsets (each with
ANOMAGENT_V1_ENABLE_LEDGER=1); this tool merges their run dirs into a single
run dir compatible with compare.py (every_steps/, ledger.jsonl,
responses.json, responses_structured.json, failed_files.json,
run_metadata.json, run.log).

Usage:
  python comparison/merge_v1_runs.py --runs <run_p1> <run_p2> <run_p3> <run_p4> --out <merged_dir>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="V1 run dirs over disjoint image sets")
    ap.add_argument("--out", required=True, help="merged run dir to create")
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    if os.path.exists(out):
        sys.exit(f"refusing to overwrite existing dir: {out}")
    os.makedirs(os.path.join(out, "every_steps"), exist_ok=True)

    n_ledger = 0
    for run in args.runs:
        run = os.path.abspath(run)
        if not os.path.isdir(run):
            sys.exit(f"run dir not found: {run}")
        es = os.path.join(run, "every_steps")
        for fn in sorted(os.listdir(es)):
            dst = os.path.join(out, "every_steps", fn)
            if os.path.exists(dst):
                sys.exit(f"duplicate every_steps entry across runs: {fn} (image sets not disjoint?)")
            shutil.copy2(os.path.join(es, fn), dst)
        led = os.path.join(run, "ledger.jsonl")
        if os.path.exists(led):
            with open(led, encoding="utf-8") as src, open(os.path.join(out, "ledger.jsonl"), "a", encoding="utf-8") as dst:
                for line in src:
                    dst.write(line)
                    n_ledger += 1
        for name in ("responses.json", "responses_structured.json", "failed_files.json"):
            p = os.path.join(run, name)
            if not os.path.exists(p):
                continue
            data = json.load(open(p))
            target = os.path.join(out, name)
            existing = json.load(open(target)) if os.path.exists(target) else []
            if not isinstance(existing, list):
                existing = [existing]
            existing.extend(data if isinstance(data, list) else [data])
            json.dump(existing, open(target, "w"), ensure_ascii=False, indent=2)
        md = os.path.join(run, "run_metadata.json")
        if os.path.exists(md):
            meta = json.load(open(md))
            meta["merged_from"] = [os.path.abspath(r) for r in args.runs]
            json.dump(meta, open(os.path.join(out, "run_metadata.json"), "w"), ensure_ascii=False, indent=2)
        with open(os.path.join(out, "run.log"), "a", encoding="utf-8") as log_out:
            log_out.write(f"===== merged from {run} =====\n")
            p = os.path.join(run, "run.log")
            if os.path.exists(p):
                shutil.copyfileobj(open(p, encoding="utf-8"), log_out)

    n_steps = len([f for f in os.listdir(os.path.join(out, "every_steps")) if f.endswith(".json")])
    print(f"merged {len(args.runs)} runs -> {out} (every_steps jsons={n_steps}, ledger lines={n_ledger})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
