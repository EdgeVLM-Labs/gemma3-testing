#!/bin/bash

# QVED Finetuning Script for Google/Gemma-3N-E2B
# This script performs Stage 3 (finetuning) only, using the pre-trained Gemma-3N-E2B checkpoint
# The checkpoint already includes pre-trained video and image projectors

# Environment setup
export PYTHONPATH="./:$PYTHONPATH"
export DATASET_DIR="$(pwd)/playground/data"

# Suppress DeepSpeed hostfile warning for single-GPU training
export PDSH_RCMD_TYPE=ssh

# WandB Configuration
export WANDB_PROJECT="gemma3n"
export WANDB_ENTITY="fyp-21"
export WANDB_NAME="qved-finetune-gemma3n-$(date +%Y%m%d_%H%M%S)"

# Model paths - using pre-trained Gemma-3N-E2B checkpoint
BASE_LLM_PATH="google/gemma-3n-E2B"
VISION_TOWER="OpenGVLab/VideoMamba"
IMAGE_VISION_TOWER="openai/clip-vit-base-patch16"
PROJECTOR_TYPE="etp"

# Output directory for finetuned model
OUTPUT_DIR_PATH="results/qved_finetune_gemma3n_E2B"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR_PATH"

# Training hyperparameters optimized for small dataset
EPOCHS=3
LR=2e-4
MM_PROJ_LR=1e-4
LORA_R=64
LORA_ALPHA=128
BATCH=8
GACC=8
MAXLEN=2048

echo "========================================="
echo "QVED Dataset Finetuning Configuration"
echo "========================================="
echo "Base Model: $BASE_LLM_PATH"
echo "Output Dir: $OUTPUT_DIR_PATH"
echo "Epochs: $EPOCHS"
echo "Learning Rate: $LR"
echo "Batch Size: $BATCH x $GACC accumulation steps = effective batch of $((BATCH * GACC))"
echo "========================================="

# Save hyperparameters to a config file
CONFIG_FILE="$OUTPUT_DIR_PATH/hyperparameters.json"
cat <<EOF > "$CONFIG_FILE"
{
  "base_model": "$BASE_LLM_PATH",
  "dataset": "QVED",
  "epochs": $EPOCHS,
  "learning_rate": $LR,
  "mm_projector_lr": $MM_PROJ_LR,
  "lora_r": $LORA_R,
  "lora_alpha": $LORA_ALPHA,
  "batch_size": $BATCH,
  "gradient_accumulation_steps": $GACC,
  "max_length": $MAXLEN,
  "wandb_project": "$WANDB_PROJECT",
  "wandb_entity": "$WANDB_ENTITY",
  "wandb_run_name": "$WANDB_NAME"
}
EOF
echo "Hyperparameters saved to $CONFIG_FILE"

# Stage 3: Fine-tuning on QVED dataset using Gemma-3N-E2B
deepspeed gemma3/train/train.py \
  --deepspeed scripts/zero2.json \
  --lora_enable True \
  --lora_r $LORA_R \
  --lora_alpha $LORA_ALPHA \
  --mm_projector_lr $MM_PROJ_LR \
  --model_name_or_path "$BASE_LLM_PATH" \
  --version gemma3n_v1 \
  --dataset_use QVED_TRAIN \
  --dataset_val QVED_VAL \
  --vision_tower "$VISION_TOWER" \
  --image_vision_tower "$IMAGE_VISION_TOWER" \
  --mm_projector_type "$PROJECTOR_TYPE" \
  --image_mm_projector_type "$PROJECTOR_TYPE" \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --group_by_modality_length True \
  --bf16 True \
  --tf32 True \
  --gradient_checkpointing True \
  --output_dir "$OUTPUT_DIR_PATH" \
  --num_train_epochs $EPOCHS \
  --per_device_train_batch_size $BATCH \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps $GACC \
  --eval_strategy "steps" \
  --eval_steps 70 \
  --save_strategy "steps" \
  --save_steps 70 \
  --save_total_limit 3 \
  --learning_rate $LR \
  --weight_decay 0. \
  --warmup_ratio 0.05 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --model_max_length $MAXLEN \
  --dataloader_num_workers 2 \
  --lazy_preprocess True \
  --report_to wandb \
  --run_name $WANDB_NAME \
  --num_select_k_frames_in_chunk 4 \
  --topk True

echo "========================================="
echo "Finetuning completed!"
echo "Model saved to: $OUTPUT_DIR_PATH"
echo "========================================="
