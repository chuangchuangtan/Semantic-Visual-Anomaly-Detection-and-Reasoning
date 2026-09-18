#!/usr/bin/env python3
"""V1 (AnomAgent @ 97fd9cf) vs V2 (AnomAgent_V2) A/B on the same 20 images.

Quality: SemAP_SemF1 @ threshold 0.7 with alpha=0.0, i.e. reasoning-only
scores (the metric blends alpha*observed + (1-alpha)*reasoning). Two
calibers are reported:
  micro - pooled TP/FP/FN across all images (by_threshold["0.7"] block)
  macro - per-image means (top-level SemF1 / SemAP in metrics.json)

Efficiency: per-call ledger.jsonl when present (V2 always writes one; V1
only when the measurement-only ledger patch is applied), otherwise V1
calls are ESTIMATED with the deterministic formula 11+5N, which
over-counts images that failed early.

Usage:
  python comparison/compare.py --v1-run <dir> --v2-run <dir> [--gt gt.json] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SEMAP = os.path.join(REPO, "SemAP_SemF1")
PY = os.environ.get("ANOMAGENT_PY", "/opt/data/private/anaconda3/envs/AnomAgent/bin/python")
LEGACY_B = "/opt/data/private/tcc/AnomAgent_plus/baseline"
DEFAULT_GT = os.path.join(REPO, "experiments", "pilot20", "gt_pilot20.json")
LEGACY_GT = os.path.join(LEGACY_B, "gt_pilot20.json")


def image_keys(v1_run: str, v2_run: str) -> list:
    keys = set()
    for run in (v1_run, v2_run):
        es = os.path.join(run, "every_steps")
        if os.path.isdir(es):
            keys.update(fn[:-5] + ".jpg" for fn in os.listdir(es)
                        if fn.endswith(".json") and not fn.endswith(".ledger.jsonl") and ".viz." not in fn)
    return sorted(keys)


def preds_of(run_dir: str, keys: list, tag: str, out_dir: str) -> tuple:
    out = os.path.join(out_dir, f"preds_{tag}.json")
    full = {}
    rs = os.path.join(run_dir, "responses_structured.json")
    if os.path.exists(rs):
        for entry in json.load(open(rs)):
            for path, items in entry.items():
                full[os.path.basename(path)] = items
    es = os.path.join(run_dir, "every_steps")
    if os.path.isdir(es):
        for fn in os.listdir(es):
            if not fn.endswith(".json") or fn.endswith(".ledger.jsonl"):
                continue
            try:
                d = json.load(open(os.path.join(es, fn)))
            except Exception:
                continue
            base = os.path.splitext(fn)[0] + ".jpg"
            if base not in full:
                full[base] = (d.get("final") or {}).get("anomalies") or []
    sub = {k: full.get(k, []) for k in keys}
    json.dump(sub, open(out, "w"), ensure_ascii=False, indent=1)
    return out, len([k for k in keys if k in full])


def ledger_stats(run_dir: str) -> tuple:
    """Per-image tokens/calls from ledger.jsonl. Returns (tok, calls, has_ledger).

    Live API calls are status ok|error; cache_hit lines are replays and cost
    nothing (excluded from both tokens and call counts).
    """
    per_img_tok, per_img_calls = {}, {}
    p = os.path.join(run_dir, "ledger.jsonl")
    if not os.path.exists(p):
        return per_img_tok, per_img_calls, False
    with open(p, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            base = os.path.splitext(r.get("image") or "?")[0] + ".jpg"
            if r.get("status") in ("ok", "error"):
                per_img_calls[base] = per_img_calls.get(base, 0) + 1
                if r.get("status") == "ok":
                    per_img_tok[base] = per_img_tok.get(base, 0) + int(r.get("total_tokens") or 0)
    return per_img_tok, per_img_calls, True


def efficiency_of(run_dir: str, keys: list) -> dict:
    per_img_tok, per_img_calls, has_ledger = ledger_stats(run_dir)
    per_img_nobj, per_img_wall = {}, {}
    es = os.path.join(run_dir, "every_steps")
    if os.path.isdir(es):
        for fn in os.listdir(es):
            if not fn.endswith(".json") or fn.endswith(".ledger.jsonl"):
                continue
            base = os.path.splitext(fn)[0] + ".jpg"
            try:
                d = json.load(open(os.path.join(es, fn)))
            except Exception:
                continue
            u = d.get("usage") or {}
            if base not in per_img_tok:
                per_img_tok[base] = int(u.get("total_tokens") or 0)
            n_obj = len(d.get("objects") or {}) if d.get("objects") else len({e.get("object") for e in (d.get("entries") or []) if e.get("object")})
            per_img_nobj[base] = n_obj
            if base not in per_img_calls and not has_ledger:
                per_img_calls[base] = 11 + 5 * n_obj  # ESTIMATED deterministic V1 formula; over-counts early-failed images
            el = (d.get("final") or {}).get("elapsed_minutes")
            if el:
                per_img_wall[base] = float(el) * 60.0
    tok = [per_img_tok[k] for k in keys if per_img_tok.get(k)]  # skip images with no recorded usage (failed mid-run)
    calls = [per_img_calls[k] for k in keys if k in per_img_calls]
    wall = [per_img_wall[k] for k in keys if k in per_img_wall]
    return {
        "n_token_records": len(tok),
        "calls_are_estimated": not has_ledger,
        "tokens_per_image_median": int(statistics.median(tok)) if tok else None,
        "tokens_per_image_mean": int(statistics.mean(tok)) if tok else None,
        "calls_per_image_mean": round(statistics.mean(calls), 1) if calls else None,
        "wall_seconds_per_image_median": int(statistics.median(wall)) if wall else None,
        "n_objects_mean": round(statistics.mean(list(per_img_nobj.values())), 2) if per_img_nobj else None,
        "total_tokens": sum(tok),
    }


def _failed(run_dir: str) -> list:
    p = os.path.join(run_dir, "failed_files.json")
    if not os.path.exists(p):
        return []
    try:
        f = json.load(open(p))
        return list(f) if isinstance(f, list) else []
    except Exception:
        return []


def semf1(pred_path: str, gt_path: str) -> dict:
    cmd = [PY, "semap_semf1_metric.py", "--pred_json", pred_path, "--gt_json", gt_path,
           "--thresholds", "0.7", "--alpha", "0.0", "--output_root", os.path.join(SEMAP, "runs")]
    r = subprocess.run(cmd, cwd=SEMAP, capture_output=True, text=True)
    out_dir = None
    for line in r.stdout.splitlines():
        if line.startswith("- "):
            out_dir = line[2:].strip()
    mj = os.path.join(out_dir, "metrics.json") if out_dir else None
    if mj and os.path.exists(mj):
        m = json.load(open(mj))
        t = m.get("by_threshold", {}).get("0.7", {})
        return {
            "micro": {  # pooled TP/FP/FN across images
                "SemF1@0.7": t.get("SemF1"),
                "Precision": t.get("Precision"),
                "Recall": t.get("Recall"),
                "TP/FP/FN": [t.get("TP"), t.get("FP"), t.get("FN")],
            },
            "macro": {  # per-image means
                "SemF1@0.7": m.get("SemF1"),
                "SemAP@0.7": m.get("SemAP"),
            },
            "n_images": m.get("num_images"),
        }
    return {"error": (r.stdout or "")[-600:] + (r.stderr or "")[-600:]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1-run", required=True)
    ap.add_argument("--v2-run", required=True)
    ap.add_argument("--gt", default=DEFAULT_GT, help=f"default {DEFAULT_GT}; falls back to {LEGACY_GT}")
    ap.add_argument("--out", default=HERE)
    args = ap.parse_args()
    args.out = os.path.abspath(args.out)
    os.makedirs(args.out, exist_ok=True)

    if not os.path.exists(args.gt) and os.path.exists(LEGACY_GT):
        print(f"[compare] --gt not found, falling back to legacy GT: {LEGACY_GT}")
        args.gt = LEGACY_GT
    if not os.path.exists(args.gt):
        print(f"[compare] GT json not found: {args.gt}")
        return 2

    keys = image_keys(args.v1_run, args.v2_run)
    p1, n1 = preds_of(args.v1_run, keys, "v1", args.out)
    p2, n2 = preds_of(args.v2_run, keys, "v2", args.out)
    m1 = semf1(p1, args.gt)
    m2 = semf1(p2, args.gt)
    e1 = efficiency_of(args.v1_run, keys)
    e2 = efficiency_of(args.v2_run, keys)
    ratio = (e2["tokens_per_image_median"] / e1["tokens_per_image_median"]) if (e1["tokens_per_image_median"] and e2["tokens_per_image_median"]) else None
    verdict = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "v1_run": args.v1_run, "v2_run": args.v2_run,
        "gt": args.gt,
        "metric": "SemAP_SemF1 @0.7, alpha=0 (reasoning-only); micro=pooled, macro=per-image mean",
        "n_images": {"v1": n1, "v2": n2},
        "n_failed": {"v1": len(_failed(args.v1_run)), "v2": len(_failed(args.v2_run))},
        "quality": {"v1": m1, "v2": m2},
        "efficiency": {"v1": e1, "v2": e2, "v2_token_ratio": round(ratio, 3) if ratio else None},
    }
    json.dump(verdict, open(os.path.join(args.out, "verdict.json"), "w"), ensure_ascii=False, indent=1)

    def f(x, spec=".3f"):
        return format(x, spec) if isinstance(x, (int, float)) else "n/a"

    def delta(a, b, spec="+.3f"):
        return f((b - a) if (isinstance(a, (int, float)) and isinstance(b, (int, float))) else None, spec)

    q1, q2 = m1.get("micro", {}), m2.get("micro", {})
    Q1, Q2 = m1.get("macro", {}), m2.get("macro", {})
    lines = [
        "# AnomAgent V1 vs V2 — 20 图 A/B",
        "",
        f"时间: {verdict['generated_at']}  ",
        f"V1: `{os.path.basename(args.v1_run)}` (GitHub AnomAgent @ 97fd9cf, text/regex, PNG 2048, thinking on)  ",
        f"V2: `{os.path.basename(args.v2_run)}` (AnomAgent_V2, json/tiered/grounding/jpeg, no-thinking, concurrency=4)",
        "",
        f"完成度: V1 {n1}/20（失败 {len(_failed(args.v1_run))} 图，空预测计全 FN）；V2 {n2}/20（失败 {len(_failed(args.v2_run))} 图）。  ",
        f"GT: `{args.gt}`  ",
        "",
        "## 质量 (SemAP_SemF1 @0.7, α=0 = reasoning-only；micro=池化, macro=逐图均值)",
        "",
        "**micro（池化 TP/FP/FN）**",
        "",
        "| | V1 | V2 | Δ |",
        "|---|---|---|---|",
        f"| SemF1@0.7 | {f(q1.get('SemF1@0.7'))} | {f(q2.get('SemF1@0.7'))} | {delta(q1.get('SemF1@0.7'), q2.get('SemF1@0.7'))} |",
        f"| Precision | {f(q1.get('Precision'))} | {f(q2.get('Precision'))} | {delta(q1.get('Precision'), q2.get('Precision'))} |",
        f"| Recall | {f(q1.get('Recall'))} | {f(q2.get('Recall'))} | {delta(q1.get('Recall'), q2.get('Recall'))} |",
        f"| TP/FP/FN | {q1.get('TP/FP/FN')} | {q2.get('TP/FP/FN')} | - |",
        "",
        "**macro（逐图均值）**",
        "",
        "| | V1 | V2 | Δ |",
        "|---|---|---|---|",
        f"| SemF1@0.7 | {f(Q1.get('SemF1@0.7'))} | {f(Q2.get('SemF1@0.7'))} | {delta(Q1.get('SemF1@0.7'), Q2.get('SemF1@0.7'))} |",
        f"| SemAP@0.7 | {f(Q1.get('SemAP@0.7'))} | {f(Q2.get('SemAP@0.7'))} | {delta(Q1.get('SemAP@0.7'), Q2.get('SemAP@0.7'))} |",
        "",
        "## 效率 (per-image, 同 20 图)",
        "",
        "| | V1 | V2 | V2/V1 |",
        "|---|---|---|---|",
        f"| token/图 (median) | {e1['tokens_per_image_median']} (n={e1['n_token_records']}) | {e2['tokens_per_image_median']} (n={e2['n_token_records']}) | {f(ratio, '.3f')} |",
        f"| token/图 (mean) | {e1['tokens_per_image_mean']} (n={e1['n_token_records']}) | {e2['tokens_per_image_mean']} (n={e2['n_token_records']}) | {f((e2['tokens_per_image_mean'] / e1['tokens_per_image_mean']) if (e1['tokens_per_image_mean'] and e2['tokens_per_image_mean']) else None, '.3f')} |",
        f"| 调用/图 (mean) | {e1['calls_per_image_mean']}{'（估算 11+5N）' if e1.get('calls_are_estimated') else ''} | {e2['calls_per_image_mean']}{'（估算 11+5N）' if e2.get('calls_are_estimated') else ''} | - |",
        f"| 单图 wall (median, s) | {e1['wall_seconds_per_image_median']} | {e2['wall_seconds_per_image_median']} | - |",
        f"| 物体数/图 | {e1['n_objects_mean']} | {e2['n_objects_mean']} | - |",
        "",
    ]
    open(os.path.join(args.out, "report.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
