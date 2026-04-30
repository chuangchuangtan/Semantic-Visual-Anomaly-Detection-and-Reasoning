#!/bin/bash

export NCCL_P2P_LEVEL=NVL          
export NCCL_SHM_DISABLE=0          
export CUDA_DEVICE_MAX_CONNECTIONS=1 
export TORCH_NCCL_BLOCKING_WAIT=0
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NVIDIA_TF32_OVERRIDE=1
export TORCH_CUDA_ARCH_LIST="8.6"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=src:$PYTHONPATH

# Download Qwen3-VL-8B-Instruct locally.
# git lfs clone https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct.git
# Breakpoint resume: cd Qwen3-VL-8B-Instruct && git lfs fetch --all
MODEL_NAME="Qwen/Qwen3-VL-8B-Instruct"

GLOBAL_BATCH_SIZE=128
BATCH_PER_DEVICE=8
NUM_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | awk -F"," '{print NF}')
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

lr=2e-5

image_folder='your/image/folder/path'
data_path='authenticity_reasoning_train.json'
data_filename=$(basename "$data_path" .json)

timeflag=$(date +%Y%m%d_%H%M%S)
run_name="qwen3vl-AnomReason_${data_filename}_${timeflag}"
output_dir="./output/${data_filename}_${timeflag}"


mkdir -p $output_dir
cp "$0" "$output_dir/"
cp ${MODEL_NAME}/chat_template.json  ${output_dir}
cp -r ./src "$output_dir/"

deepspeed src/train/train_sft.py \
    --use_liger_kernel True \
    --lora_enable True \
    --use_dora False \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.5 \
    --num_lora_modules -1 \
    --deepspeed scripts/zero3_offload.json \
    --model_id $MODEL_NAME \
    --data_path $data_path \
    --image_folder $image_folder \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --freeze_merger True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((256 * 28 * 28)) \
    --image_max_pixels $((512 * 28 * 28)) \
    --learning_rate $lr \
    --merger_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 0 \
    --save_total_limit 10 \
    --dataloader_num_workers 8


python src/merge_lora_weights.py \
    --model-base ${MODEL_NAME} \
    --model-path $output_dir \
    --save-model-path ${output_dir}-lora-merged \
    --safe-serialization


# cp {your local MODEL_NAME}/*.json ${output_dir}-lora-merged