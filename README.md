# Semantic Visual Anomaly Detection and Reasoning in AI-Generated Images

Official implementation of  **Semantic Visual Anomaly Detection and Reasoning in AI-Generated Images**  (ICLR 2026)

This repository contains the code, dataset interface, and evaluation tools for semantic-level anomaly detection and reasoning in AI-generated images.

---
## 📰 News

- [2026-04] 🎉 We release the **AnomReason** dataset. [[Download link](https://drive.google.com/drive/folders/1jsDFvnJkKrl2cr2cWDhj3K_exzh9W8ey?usp=sharing)]
- [2026-04] 🚀 We release the code for **AnomAgent**.
- [2026-04] 📊 We introduce a new semantic matching metric: **SemAP** and **SemF1**.
- [Coming Soon] 🔧 Training and evaluation pipelines will be released.

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


## Dataset: AnomReason

**AnomReason** is a benchmark for semantic anomaly detection and reasoning in AI-generated images.

- Images: photorealistic AI-generated images  
- Annotations: structured anomaly quadruples  
- Splits: train / test  

### Annotation Format

```json
{
  "image_id": [
    {
      "name": "Missing safety harness",
      "phenomenon": "...",
      "reasoning": "...",
      "severity": 20
    }
  ]
}
```

---

## Running AnomAgent

```bash
cd AnomAgent
python3 -m anomagent --input path/to/image_or_dir --output-dir outputs
```

---

## Evaluation
```bash
cd SemAP_SemF1
CUDA_VISIBLE_DEVICES=0 python3 semap_semf1_metric.py \
  --pred_json /path/to/preds.json \
  --gt_json /path/to/ground_truth.json \
  --thresholds 0.7,0.8,0.9 \
  --alpha 0.5 \
  --use_gpu
```

---

## Training

```bash

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
