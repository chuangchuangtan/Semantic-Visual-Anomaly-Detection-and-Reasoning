# Semantic Visual Anomaly Detection and Reasoning in AI-Generated Images

Official implementation of  **Semantic Visual Anomaly Detection and Reasoning in AI-Generated Images**  (ICLR 2026)

This repository contains the code, dataset interface, training-data conversion scripts, inference utilities, and evaluation tools for semantic-level anomaly detection and reasoning in AI-generated images.

---
## 📰 News

- [2026-04] 🎉 We release the **AnomReason** dataset. [[Download link](https://drive.google.com/drive/folders/1jsDFvnJkKrl2cr2cWDhj3K_exzh9W8ey?usp=sharing)]
- [2026-04] 🚀 We release the code for **AnomAgent**.
- [2026-04] 📊 We introduce a new semantic matching metric: **SemAP** and **SemF1**.
- [2026-04] 🔧 We release the code for Training and Evaluation.
- [2026-04] 🚀 We release the Assessment Images.[[Download link](https://drive.google.com/drive/folders/1jsDFvnJkKrl2cr2cWDhj3K_exzh9W8ey?usp=sharing)]

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

## Repository Layout

```text
AnomAgent/                    # OpenAI API based semantic anomaly annotation agent
build_instruction_dataset/    # Qwen3-VL instruction-data conversion scripts
Inference_AnomReason_vllm/    # vLLM batch inference and post-processing utilities
SemAP_SemF1/                  # SemAP / SemF1 evaluator
scripts/                      # Qwen3-VL fine-tuning helper scripts
requirements_vllm.txt         # GPU inference environment requirements
```


---

## Environment
The python environment for AnomAgent, Inference, and Evaluation.
```bash
conda create -n anomreason-vllm python=3.10.18 -y
conda activate anomreason-vllm
pip install -r ./requirements_vllm.txt 
```


## Running AnomAgent

```bash
cd AnomAgent
pip install -e .
python3 -m anomagent --input path/to/image_or_dir --output-dir outputs
```

---
## Instruction Dataset

Run from the repository root:

```bash
python3 build_instruction_dataset/build_instruction_dataset.py \
  --recipe anomaly_only \
  --fake-input AnomReason_After_HITL.json \
  --split-file dataset_split_list.json \
  --split-name train \
  --output outputs_$(date +%Y%m%d_%H%M%S)/anomaly_only_train.json
```


```bash
python3 build_instruction_dataset/build_instruction_dataset.py \
  --recipe authenticity_reasoning \
  --fake-input AnomReason_After_HITL.json \
  --real-input Real_image_caption_Num23968_2025_07_28_14_25_57.json \
  --split-file dataset_split_list.json \
  --split-name train \
  --output outputs_$(date +%Y%m%d_%H%M%S)/authenticity_reasoning_train.json
```


---

## Training

```bash
git clone https://github.com/2U1/Qwen-VL-Series-Finetune
cp scripts/qwen3_finetune_lora.sh  Qwen-VL-Series-Finetune/scripts
cd Qwen-VL-Series-Finetune

conda create -n Qwen-VL-Series-Finetune python av==14.4.0 -c conda-forge -y
pip install -r requirements.txt -f https://download.pytorch.org/whl/cu128
pip install qwen-vl-utils
pip install flash-attn --no-build-isolation
# Set image_folder and data_path in qwen3_finetune_lora.sh

bash scripts/qwen3_finetune_lora.sh
```


---

## Inference

```text
Refer Inference_AnomReason_vllm/README.md for detailed information.

1. Start vLLM servers.
2. Run dataset inference.
3. Merge NDJSON predictions into JSON.
```


## Evaluation
```bash
# alpha 1.0: SemAP_Phe
# alpha 0.0: SemAP_Rea
# alpha 0.5: SemAP_Full

cd SemAP_SemF1
CUDA_VISIBLE_DEVICES=0 python3 semap_semf1_metric.py \
  --pred_json /path/to/preds.json \
  --gt_json /path/to/ground_truth.json \
  --thresholds 0.7,0.8,0.9 \
  --alpha 0.5 \
  --use_gpu
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
