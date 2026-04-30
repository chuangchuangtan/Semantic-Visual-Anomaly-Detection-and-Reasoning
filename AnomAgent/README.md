# AnomAgentV1

Multi-step anomaly analysis for AI-generated images (OpenAI API).

## Install
You can use the anomreason-vllm Python environment.
```bash
pip install -e .
```

## Configure

Create `.env` from `.env.example` and fill in your key:

```bash
cp .env.example .env
```

Required:
- `OPENAI_API_KEY`

Optional:
- `OPENAI_MODEL` (default: `gpt-4o`)
- `OPENAI_BASE_URL` (for proxies / gateways)


For local vLLM or proxy deployments, set `OPENAI_BASE_URL` to the OpenAI-compatible endpoint and set `OPENAI_MODEL` to the served model id. 
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
VLLM_VIDEO_FETCH_TIMEOUT=120 \
vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 32768 \
  --port 5000 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --api-key YOUR_API_KEY \
```

## Run

Analyze one or more images / folders (folders are searched recursively):

```bash
python3 -m anomagent --input path/to/image_or_dir --output-dir outputs
# or, if installed:
anomagent --input path/to/image_or_dir --output-dir outputs
```

Common options:

```bash
anomagent \
  --input examples \
  --output-dir outputs \
  --model gpt-4o \
  --log-level INFO \
  --max-workers 8 \
  --num-get-objects 3 \
  --num-summarizer-step2 3 \
  --max-objects 12 \
  --image-max-side 2048
```

## Outputs

Each run creates a timestamped directory under `--output-dir`, e.g. `outputs/run_YYYY_MM_DD_HH_MM_SS_*`.

Inside the run directory:
- `run.log` contains runtime logs.
- `run_metadata.json` records the (redacted) run configuration.
- `every_steps/` contains per-image JSON logs (image bytes are stripped; includes `image_info` + token `usage`).
- `responses.json`, `responses_structured.json`, `failed_files.json` are aggregated results.

## Notes

- You must use a **vision-capable** OpenAI model for image inputs.
- This repository is research/experimental code; prompts and output formats may change.
