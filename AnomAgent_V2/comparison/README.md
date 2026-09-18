# V1 vs V2 — 20-sample A/B

- **V1**: `AnomAgent` at commit `97fd9cf` (the AnomAgent currently on GitHub) — text/regex
  parsing, full-size PNG (2048) re-sent on every call, fixed max_tokens, serial images.
- **V2**: `AnomAgent_V2` (this package) — default config: strict JSON schema, tiered
  resolution + grounding crops + JPEG q85, elastic max_tokens, no-thinking, step cache,
  per-step ledger, `--image-concurrency 4`.
- **Protocol**: identical in both (`--schedule fixed` = the paper pipeline, T=3 drafts,
  1 merge, per-object 5 calls, S=3 verify, S=3 global scan, 1 finish — the "7+5N"
  notation in earlier notes assumed S=1; with the code default S=3 the count is
  T+1+5N+2S+1 = 11+5N; V1 and V2 make exactly the same calls). No relation-pair
  dedup in either (per design decision). Prompts are byte-identical (V2 appends only
  the strict-JSON schema tail in json mode).
- **Data**: first 20 images of the 50-image set; GT = `experiments/pilot20/gt_pilot20.json`
  (input list + images + GT all under `experiments/pilot20/`).
- **Model/endpoint**: same for both (Qwen3.8-27B @ vLLM :5000).
- **Quality**: SemAP_SemF1 @ threshold 0.7, alpha=0 (**reasoning-only**; the metric
  blends alpha*observed + (1-alpha)*reasoning). Two calibers are reported:
  **micro** = pooled TP/FP/FN, **macro** = per-image means (`SemAP_SemF1/semap_semf1_metric.py`).

## compare.py

```bash
python comparison/compare.py --v1-run <V1 run dir> --v2-run <V2 run dir> [--gt gt.json] [--out DIR]
```

Aggregates per-image predictions from both runs, runs the metric, and writes
`preds_v1.json`, `preds_v2.json`, `verdict.json`, `report.md` (dual-caliber quality table +
per-image tokens / calls / wall time, V2/V1 ratios). Token/call accounting uses each run's
per-call `ledger.jsonl` when present (cache_hit lines cost 0); for V1 runs without a ledger
calls are ESTIMATED with 11+5N and flagged `（估算）` in the report. Paths are portable
(repo-relative + `ANOMAGENT_PY` env override); the GT default falls back to the legacy
location with a notice.

## merge_v1_runs.py

```bash
python comparison/merge_v1_runs.py --runs <run_p1> <run_p2> ... --out <merged dir>
```

Merges V1 run dirs produced by disjoint parallel processes (V1 is serial per process; the
9/2 reproduction used 4 x 5-image processes) into one run dir compatible with compare.py.

## Experiment outputs

- 9/1 A/B: `experiments/ab_2026_09_01/` (runs + `comparison_20260901/` with the
  old single-caliber report — regenerated numbers: see `docs/hand-off.md` §10.4/§10.7).
- 9/2 final-code reproduction: `experiments/repro_2026_09_02/` (see `REPRODUCTION.md`).
