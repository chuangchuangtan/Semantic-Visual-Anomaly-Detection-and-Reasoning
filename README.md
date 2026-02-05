# Semantic Visual Anomaly Detection and Reasoning in AI-Generated Images

Official implementation of  **Semantic Visual Anomaly Detection and Reasoning in AI-Generated Images**  (ICLR 2026)

This repository contains the code, dataset interface, and evaluation tools for semantic-level anomaly detection and reasoning in AI-generated images.

---

## Overview

Recent AI-generated images often exhibit *semantic anomalies* that go beyond low-level visual artifacts, such as violations of commonsense, physical laws, object relations, or anatomy.

This work introduces:

- **AnomAgent**: a multi-agent pipeline for semantic anomaly annotation
- **AnomReason**: a large-scale benchmark with structured anomaly annotations
- **Semantic evaluation metrics** (SemAP / SemF1) for anomaly detection and reasoning

Each anomaly is represented as a structured quadruple:

- **Name**: concise description of the anomaly  
- **Phenomenon**: what is visually incorrect  
- **Reasoning**: why it violates commonsense or logic  
- **Severity**: anomaly severity score  

---

## Repository Structure

```
.
├── anomagent/
├── anomreason/
├── metrics/
├── scripts/
│   ├── annotate/
│   ├── train/
│   └── eval/
├── examples/
├── docs/
└── README.md
```

---

## Installation

```bash
conda create -n anomreason python=3.10 -y
conda activate anomreason
pip install -r requirements.txt
```

---

## Dataset: AnomReason

**AnomReason** is a benchmark for semantic anomaly detection and reasoning in AI-generated images.

- Images: photorealistic AI-generated images  
- Annotations: structured anomaly quadruples  
- Splits: train / test  

### Annotation Format

```json
{
  "image_id": "xxx",
  "anomalies": [
    {
      "name": "Missing safety harness",
      "phenomenon": "...",
      "reasoning": "...",
      "severity": 20
    }
  ]
}
```

Dataset download links will be released.

---

## Running AnomAgent

```bash
python scripts/annotate/run_anomagent.py \
  --input_dir data/images/test \
  --output outputs/anomagent_results.jsonl
```

---

## Evaluation

Semantic anomaly detection and reasoning are evaluated using **SemAP** and **SemF1**, based on semantic similarity matching.

```bash
python scripts/eval/eval_semantic.py \
  --pred outputs/anomagent_results.jsonl \
  --gt data/anomreason/test.jsonl
```

---

## Training

```bash
python scripts/train/train_lora.py \
  --train data/anomreason/train.jsonl \
  --output checkpoints/exp_name
```

---

## Citation

```bibtex
@misc{tan2025semanticvisualanomalydetection,
      title={Semantic Visual Anomaly Detection and Reasoning in AI-Generated Images}, 
      author={Chuangchuang Tan and Xiang Ming and Jinglu Wang and Renshuai Tao and Bin Li and Yunchao Wei and Yao Zhao and Yan Lu},
      year={2025},
      eprint={2510.10231},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.10231}, 
}
```
