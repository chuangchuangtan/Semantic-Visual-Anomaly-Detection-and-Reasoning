# AnomAgentV1

Multi-step anomaly analysis for AI-generated images (OpenAI API).

## Install

```bash
pip install -r requirements.txt
# or
pip install -e .
```

## Development

```bash
pip install -e ".[dev]"
pre-commit install
pre-commit run --all-files
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

## License

MIT (see `LICENSE`).
