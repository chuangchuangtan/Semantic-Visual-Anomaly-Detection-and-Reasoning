# AnomAgent V2

Agent-ified multi-step **semantic anomaly detection & reasoning** for AI-generated
images. Drop-in replacement for `AnomAgent` (V1) in this repository: same analysis
protocol and **byte-identical prompts** (`prompts.py` is the V1 prompt file plus one
controller system prompt that is only used in `--schedule dynamic` mode), rebuilt as a
**controller + 7 tools**, with engineering and efficiency upgrades around it.

| # | Upgrade | Detail | Where |
|---|---------|--------|-------|
| 1 | **Strict JSON-schema outputs** | Steps emit schema-validated JSON (OpenAI-compatible `response_format` / guided decoding) instead of fragile regex parsing; one feedback retry on invalid JSON | `anomagent_v2/schemas.py`, `core.chat_json` |
| 2 | **Image token governance** | Bbox grounding from object discovery → per-object / union crops for local steps; **tiered resolution** (global / medium / local); **JPEG q85** encoding; `--image-max-side` long-edge control (default 2048) | `image_utils.py`; `--vision-mode tiered`, `--grounding`, `--image-format jpeg` |
| 3 | **Crop accuracy check** | Local, zero-LLM gate: no bbox / scene-level (>60% of image) / too small (<1%) / drafts disagree (IoU < 0.30) → fall back to the full image; plus a post-hoc "descriptor must mention the object" re-run (max 1/object) | `agent/crop_verify.py` |
| 4 | **Elastic `max_tokens`** | 2000 per call by default; on `finish_reason == "length"` the cap is doubled (up to `--max-tokens-cap`, default 16000) and retried — no truncated output is ever emitted | `core.chat` |
| 5 | **Agent-ified controller + 7 tools** | `detect_objects / crop / inspect / check_relation / verify / global_scan / finish`. `--schedule fixed` (default) executes the tools in the exact paper order with no controller calls and no budget logic → deterministic, same call pattern as V1. `--schedule dynamic` lets a controller LLM pick the next tool under a per-image token budget (default action cap `3N+8`) | `agent/tools.py`, `agent/controller.py` |
| 6 | **Step-level response cache** | Key = sha256(model, step, **sample slot**, canonical messages, sampling params, `response_format`, `extra_body`) → instant re-runs and fully deterministic replays; self-consistency samples keep distinct slots (see [Deterministic replay](#deterministic-replay-step-cache)) | `step_cache.py` |
| 7 | **Model routing** | Optional second endpoint (`--expert-base-url/--expert-model`) + per-step routing (`--route-steps summarizer_step2=expert,...`) | `config.py`, `core.py` |
| 8 | **Per-step token ledger** | Every API call is logged (run-level `ledger.jsonl` + per-image `<name>.ledger.jsonl`): model/client/step/object, prompt/completion tokens, finish_reason, latency, attempts; aggregated in `run_summary.json` and `tools/ledger_report.py` | `core._record_chat`, `tools/ledger_report.py` |
| 9 | **Inter-image parallelism + multi-endpoint** | `--image-concurrency K`, round-robin across `--base-urls url1,url2`; combined with the step cache this cuts wall time ~5x at equal quality (20-image A/B: 4h35m serial → 57.7 min). Note: the OpenAI `/v1/batches` hook was removed in code review — it was a silent no-op (never wired into the fan-out path) and the deployment endpoint (vLLM) does not implement it; concurrent fan-out + step cache cover the same goal on any OpenAI-compatible endpoint | `runner.py` |
| 10 | **Per-sample HTML visualization** | One self-contained page per image: full step timeline — for every call, the exact image variant sent (crops with bbox overlays on the global view), system/user prompts, raw + parsed outputs, tokens / latency / finish_reason; batch or single sample | `tools/viz_html.py` |

## Call budget (fixed schedule)

`--schedule fixed` (default) runs the 7 tools in the paper order. Per image with N
objects and sampling counts T (discovery drafts), S2 (verify), Sg (global scans):

| Step | Tool | Calls |
|------|------|-------|
| Object discovery | `detect_objects` | T drafts + 1 merge |
| Per object (× N) | `inspect` (descriptor step1+2), `check_relation` (relationship step1+2, per-object summarizer step1) | 5N |
| Evidence verification | `verify` (summarizer step2) | S2 |
| Whole-image safety scan | `global_scan` (single-shot reasoning) | Sg |
| Final report | `finish` (summarizer step3) | 1 |

**Total = T + 1 + 5N + S2 + Sg + 1**, i.e. **11+5N with the defaults** (T=3, S2=3,
Sg=3). This is the exact V1 call pattern under the same sampling config: V1's code
default is Sg=1 (→ 9+5N) and V2 can match it with `--num-global-reasoning 1`.
(The earlier shorthand "7+5N" corresponds to T=3 with S2=Sg=1.) No budget logic and
no unordered relation-pair dedup are introduced — the schedule is config-driven, so
call counts are predictable and reproducible.

`--schedule dynamic` keeps the same 7 tools but lets a controller model decide the
next call per loop iteration (strict-JSON action, one per turn) under
`--budget-factor × baseline` tokens (`--budget-ref-run` or `--budget-tokens`) and an
action cap (`--max-actions`, default `3N+8`). Use it for cost/coverage studies; fixed
is the production default.

## Install

```bash
pip install -e .
```

(any env with `openai>=1.40`, `Pillow`, `python-dotenv`, `tqdm` works; the
`AnomReason_vllm` env at the repo root works as-is. The console script
`anomagent_v2` and `python -m anomagent_v2` are equivalent.)

## Configure

```bash
cp .env.example .env   # set OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL
```

For a local vLLM instance:

```bash
OPENAI_API_KEY=*** \
OPENAI_BASE_URL=http://127.0.0.1:5000/v1 \
OPENAI_MODEL=Qwen3.8-27B
```

CLI flags (`--api-key/--base-url/--model`) override the environment.

## Run

```bash
python -m anomagent_v2 \
  --input path/to/image_or_dir \
  --output-dir outputs \
  --run-suffix myrun \
  --no-thinking \              # Qwen3-series: disable thinking tokens
  --image-concurrency 4        # inter-image parallelism (1 = serial, paper behavior)
```

Common options (defaults in parentheses):

```
--output-mode {json,text}        json (strict-JSON steps; text = V1 free-text)
--vision-mode {tiered,full}      tiered (global/medium/local resolution tiers)
--grounding / --no-grounding     on  (bbox crops for local steps; requires json mode)
--no-crop-verify                 crop accuracy check is on by default
--image-max-side 2048            long-edge cap for the global tier
--image-format {jpeg,png}        jpeg (quality --jpeg-quality 85)
--num-get-objects 3              T: object discovery drafts
--num-summarizer-step2 3         S2: verify (summarizer step2) samples
--num-global-reasoning 3         Sg: global scan samples (V1 code default: 1)
--max-tokens 2000 --max-tokens-cap 16000   elastic per-call cap
--schedule {fixed,dynamic}       fixed (deterministic paper order)
--max-actions 0                  [dynamic] action cap (0 = default 3N+8)
--budget-ref-run DIR / --budget-tokens N   [dynamic] budget baseline
--max-workers 8                  intra-image parallel calls
--max-objects N                  cap object count (cost knob)
--step-cache-dir DIR / --no-step-cache     step cache (default: <run-dir>/step_cache)
--image-concurrency K --base-urls u1,u2    inter-image parallelism / round-robin
--expert-base-url U --expert-model M --expert-api-key K   [routing]
--route-steps "step=expert,..."           [routing] per-step override
--continue                             resume the latest run dir for --run-suffix
--start SUB --end SUB                  filter images by basename substring
--log-level {DEBUG,INFO,WARNING,ERROR}
```

### V1 parity (reproduce the original pipeline with V2 code)

```bash
python -m anomagent_v2 --input ... --output-dir outputs \
  --output-mode text --vision-mode full --no-grounding \
  --image-format png --num-global-reasoning 1
```

This disables every V2 extension (JSON schemas, tiering, grounding crops, JPEG) and
restores V1's free-text steps on full-size PNG with V1's default Sg=1 → identical
call pattern and prompts to `AnomAgent` V1.

### Deterministic replay (step cache)

The step cache is **on by default** at `<run-dir>/step_cache`. A cache key is the
sha256 of (model, step, **sample slot**, canonical messages — which embed the image
data URLs —, sampling params, `response_format`, `extra_body`). The sample slot
(`sample = 0..k-1`) separates the self-consistency samples, so:

- **first run**: the T discovery drafts, S2 verify and Sg global-scan samples each
  get their own key → truly independent samples, as in V1;
- **replay**: re-running the same images/params hits every key → 0 API calls and a
  byte-identical final output.

```bash
# 1) first run (fills the cache)
python -m anomagent_v2 --input img.jpg --output-dir /tmp/demo --run-suffix first ...

# 2) replay (100% cache hits, final output identical)
python -m anomagent_v2 --input img.jpg --output-dir /tmp/demo --run-suffix replay \
  --step-cache-dir /tmp/demo/run_<ts>_first/step_cache ...
```

Verified on a 4-object image (31 = 11 + 5×4 calls): first run 0/31 hits, replay
**31/31 hits in 0.3 s** with `final.anomalies` byte-identical. Use
`--step-cache-dir` to share one cache across ablation runs that only differ in the
flag under test; `--no-step-cache` disables caching entirely.

## Outputs

Each run creates `<output-dir>/run_<timestamp>[_suffix]/`:

- `run.log`, `run_metadata.json` (redacted config), `run_summary.json`
  (`n_images`, `tokens_per_image`, `calls_per_image`, `step_cache_hits`,
  `truncated_length_retries`, `failed`)
- `ledger.jsonl` — per-call token/latency/finish_reason ledger (run-level)
- `every_steps/<image>.json` — V1-compatible per-image state + final anomalies
- `every_steps/<image>.ledger.jsonl` — per-image ledger
- `every_steps/<image>.viz.json` + `every_steps/assets/<image>/` — step I/O for the HTML viz
- `responses.json`, `responses_structured.json`, `failed_files.json` (V1 format,
  consumed by `SemAP_SemF1` / `make_pred.py` unchanged)

## Tools

```bash
# per-step token / latency / finish_reason report
python tools/ledger_report.py --run-dir outputs/run_... [--image NAME]

# per-sample HTML visualization (batch or single)
python tools/viz_html.py --run-dir outputs/run_... --all
python tools/viz_html.py --run-dir outputs/run_... --sample 2025_04_18_....jpg
python tools/viz_html.py --run-dir outputs/run_... --samples a.jpg,b.jpg
# -> outputs/run_.../viz_html/index.html + one self-contained .html per sample
```

## 20-sample A/B vs V1

See `comparison/README.md` and `comparison/report.md`. Headline (same 20 images,
same Qwen3.8-27B endpoint, identical protocol): V1 (text/regex, full-size PNG,
serial) **SemF1@0.7 = 0.537** with 5/20 images failing, vs V2 (default config,
`--image-concurrency 4`) **0.806**, 20/20 images; median tokens/image
164,840 → 150,043 (0.91×), wall time 4h35m (serial) → 57.7 min (×4).
