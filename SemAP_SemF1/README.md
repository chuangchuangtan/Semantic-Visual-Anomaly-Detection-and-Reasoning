# SemAP / SemF1

This release packages the metric as a cleaner public evaluator instead of an experiment-only script.

The metric uses greedy one-to-one matching between predicted anomaly entries and ground-truth entries. Each pair is scored with fused BERTScore:

`score = alpha * BERTScore(Observed Phenomenon) + (1 - alpha) * BERTScore(Reasoning)`

After matching, thresholded precision is used as the SemAP value in this implementation, and thresholded F1 is used as SemF1.


## Input format

Prediction JSON and ground-truth JSON should both look like this:

```json
{
  "image_001.jpg": [
    {
      "Observed Phenomenon": "A vehicle is parked across two lanes.",
      "Reasoning": "This blocks traffic flow and creates an unsafe road condition."
    },
    {
      "Observed Phenomenon": "A pedestrian is walking outside the crosswalk.",
      "Reasoning": "The pedestrian may enter the vehicle path unexpectedly."
    }
  ]
}
```

Optional fields such as `Name` and `Severity Score` are preserved if present, but the metric only requires:

- `Observed Phenomenon`
- `Reasoning`

## Quick start

You can use the anomreason-vllm Python environment.

Run a single alpha:

```bash
CUDA_VISIBLE_DEVICES=0 python semap_semf1_metric.py \
  --pred_json /path/to/preds.json \
  --gt_json /path/to/ground_truth.json \
  --thresholds 0.7,0.8,0.9 \
  --alpha 0.5 \
  --use_gpu
```

Run multiple alphas in one command:

```bash
CUDA_VISIBLE_DEVICES=0 python semap_semf1_metric.py \
  --pred_json /path/to/preds.json \
  --gt_json /path/to/ground_truth.json \
  --alphas 0.0,0.5,1.0 \
  --thresholds 0.7,0.8,0.9 \
  --use_gpu
```

Filter evaluation to a split file:

```bash
CUDA_VISIBLE_DEVICES=0 python semap_semf1_metric.py \
  --pred_json /path/to/preds.json \
  --gt_json /path/to/ground_truth.json \
  --subset_json /path/to/test_split.json
```

The subset file can be:

- a dict keyed by image id
- a list of image ids
- a list of single-key dicts such as the format used in many experimental split files

## Outputs

Each run creates a directory under `runs/` containing:

- `run.log`
- `run_config.json`
- `metrics.json`
- `matched_results.json`
- `matched_results_pretty.txt`
- snapshots of the release scripts

The metrics file includes:

- per-threshold precision, recall, SemF1, and SemAP
- overall `SemAP` and `SemF1`
- compatibility aliases `mAP` and `mF1`
- per-image metrics

## Files

- `semap_semf1_metric.py`: main public evaluator
- `utiliz.py`: shared helpers for JSON IO, logging, subset parsing, and response parsing
