# AnomReason vLLM Inference

Fast, resumable inference scripts for running an AnomReason model on an image dataset through vLLM's OpenAI-compatible API, then exporting predictions for SemAP / SemF1 evaluation.

The public workflow replaces the old manual sequence:

1. Start vLLM servers.
2. Run dataset inference.
3. Merge NDJSON predictions into JSON.
4. Evaluate with `semap_semf1_metric.py`.


## Repository Layout

```text
anomreason_vllm/                 # Reusable parser, IO, dataset, and client code
scripts/serve_vllm.sh            # Multi-GPU vLLM launcher
scripts/infer_dataset.py         # Resumable batch inference
scripts/merge_results.py         # NDJSON -> pred_json converter
scripts/run_pipeline.sh          # Inference + merge + optional metric wrapper
configs/prompt_ai_generated_yes_no.txt
```


## 1. Start vLLM

```bash
bash scripts/serve_vllm.sh \
  --model /path/to/model \
  --gpus 0,1,2,3 \
  --base-port 8000 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --stagger-seconds 60
```

This starts one vLLM OpenAI-compatible server per GPU:

```text
http://127.0.0.1:8000/v1
http://127.0.0.1:8001/v1
...
```

Logs and PID files are written to `runs/vllm_*/` unless `--log-dir` is provided.


## 2. Run Inference

Use the pipeline wrapper when vLLM is already running:

Table 1: Comparative performance on the AnomReason-Test.

The test JSON should provide the list or mapping of test images.

```bash
bash scripts/run_pipeline.sh \
  --model vllm_model_id \
  --test-json /path/to/AnomReason_After_HITL_test.json \ 
  --image-root /path/to/images \
  --instances 4 \
  --base-port 8000 \
  --concurrency 32 \
  --output-dir runs/my_eval
```


Table 2: Explainable deepfake detection on AnomReason-Deepfake.
```bash
bash scripts/run_pipeline.sh \
  --model /path/to/model \
  --test-json /path/to/AnomReason_Deepfake_Test_list.json \
  --image-root /path/to/images \
  --prompt-file configs/prompt_ai_generated_yes_no.txt \
  --raw-response-only \
  --path-mode relative \
  --instances 4 \
  --base-port 8000 \
  --concurrency 32 \
  --output-dir runs/my_eval_cla
```




Artifacts:

```text
runs/my_eval/predictions.ndjson      # Append-only inference records
runs/my_eval/errors.ndjson           # Failed requests
runs/my_eval/missing_images.json     # Dataset references not found on disk
runs/my_eval/predictions.json        # SemAP/SemF1 pred_json format
```

The inference step is resumable by default in `run_pipeline.sh`. Re-running the same command skips successful records already present in `predictions.ndjson`.

## Dataset JSON Formats

The loader supports common dataset layouts:

```json
["image_001.jpg", "image_002.jpg"]
```

```json
[
  {"image": "image_001.jpg", "label": "..."},
  {"image_path": "subdir/image_002.jpg", "label": "..."}
]
```

```json
{
  "image_001.jpg": [{"Observed Phenomenon": "..."}],
  "image_002.jpg": []
}
```

Path resolution is controlled by `--path-mode`:

```text
basename   image_root / basename(reference)
relative   image_root / reference
absolute   use absolute references directly, otherwise image_root / reference
```

`basename` is the default because many internal annotation files store source paths while evaluation images are projected into one flat folder.

## 3. Merge Only

If inference has already produced NDJSON:

```bash
python3 scripts/merge_results.py \
  --input runs/my_eval/predictions.ndjson \
  --output runs/my_eval/predictions.json \
  --prefer-parsed
```


## 4. Evaluate With SemAP / SemF1

Table 1: Comparative performance on the AnomReason-Test.
```bash
# alpha 1.0: SemAP_Phe
# alpha 0.0: SemAP_Rea
# alpha 0.5: SemAP_Full

CUDA_VISIBLE_DEVICES=0 python3 SemAP_SemF1/semap_semf1_metric.py \
  --pred_json runs/my_eval/predictions.json \
  --gt_json /path/to/AnomReason_After_HITL_test.json \
  --thresholds 0.7,0.8,0.9 \
  --alpha 0.5 \
  --use_gpu
```

Table 2: Explainable deepfake detection on AnomReason-Deepfake.

```bash
bash scripts/postprocess_dual_prompt_results.py \
  --anom-detection-json runs/my_eval/predictions.json \
  --classification-json runs/my_eval_cla/predictions.json \
  --output-dir runs/postprocess_dual_prompt
```


```bash
# alpha 1.0: SemAP_Phe
# alpha 0.0: SemAP_Rea
# alpha 0.5: SemAP_Full

CUDA_VISIBLE_DEVICES=0 python3 SemAP_SemF1/semap_semf1_metric.py \
  --pred_json runs/postprocess_dual_prompt/anom_detection_filtered.json \
  --gt_json /path/to/AnomReason_After_HITL_test.json \
  --thresholds 0.7,0.8,0.9 \
  --alpha 0.5 \
  --use_gpu
```


