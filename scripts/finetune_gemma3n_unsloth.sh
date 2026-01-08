#!/bin/bash

# ==========================================
# Gemma-3N Fine-tuning Script with Unsloth
# ==========================================
# 
# This script fine-tunes gemma-3n-E4B-it using Unsloth FastVisionModel
# Supports both local QVED dataset and HuggingFace streaming datasets
#
# Usage:
#   # For local QVED dataset (default, prepared by dataset.py):
#   bash scripts/finetune_gemma3n_unsloth.sh
#   # OR explicitly:
#   bash scripts/finetune_gemma3n_unsloth.sh --local
#
#   # For HuggingFace dataset (streaming download, optional):
#   bash scripts/finetune_gemma3n_unsloth.sh --hf
#
# ==========================================

set -e  # Exit on error

# ===================== Configuration =====================

# Model configuration
MODEL_NAME="unsloth/gemma-3n-E4B-it"
LOAD_IN_4BIT=""  # Add "--load_in_4bit" for 4-bit quantization

# Dataset configuration (choose one)
DATASET_MODE="${1:---local}"  # Default to local mode (prepared by dataset.py)

# Local QVED dataset paths (prepared by dataset.py)
TRAIN_JSON="dataset/qved_train.json"
VAL_JSON="dataset/qved_val.json"

# HuggingFace dataset configuration
HF_DATASET="EdgeVLM-Labs/QVED-Test-Dataset"
HF_SPLIT="train"
NUM_SAMPLES=500
VIDEO_SAVE_DIR="videos"

# Training hyperparameters
OUTPUT_DIR="outputs/gemma3n_finetune_$(date +%Y%m%d_%H%M%S)"
BATCH_SIZE=1
GRAD_ACCUM=4
LEARNING_RATE=2e-4
NUM_EPOCHS=1
MAX_SEQ_LENGTH=50000
WARMUP_RATIO=0.03
MAX_GRAD_NORM=0.3
WEIGHT_DECAY=0.001

# DeepSpeed configuration (optional)
DEEPSPEED_CONFIG=""  # Set to "scripts/zero.json" to enable DeepSpeed (requires: pip install deepspeed)

# LoRA configuration
LORA_R=64
LORA_ALPHA=128
LORA_DROPOUT=0.0

# Video processing
NUM_FRAMES=8

# Wandb configuration
WANDB_PROJECT="Finetune-gemma3n"
WANDB_RUN_NAME="gemma-3n-finetune-$(date +%Y%m%d_%H%M%S)"

# HuggingFace upload (optional)
UPLOAD_TO_HF=""  # Set to "--upload_to_hf" to auto-upload after training
HF_REPO_NAME=""  # Leave empty for auto-generated name, or set custom name
HF_PRIVATE=""    # Set to "--hf_private" to make repository private

# Evaluation configuration (optional)
RUN_EVAL="--run_eval"           # Enable evaluation during training
EVAL_STEPS=50                    # Run eval every N steps
SAVE_EVAL_CSV="--save_eval_csv" # Save eval results as CSV
GENERATE_REPORT=""               # Set to "--generate_report" to generate Excel report

# HuggingFace token (optional, set via environment variable)
# export HF_TOKEN="your_token_here"

# ===================== Environment Setup =====================
export PYTHONPATH="./:$PYTHONPATH"

# Suppress warnings
export TOKENIZERS_PARALLELISM=false

echo "========================================="
echo "Gemma-3N Fine-tuning with Unsloth"
echo "========================================="
echo "Model: $MODEL_NAME"
echo "Output: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE x $GRAD_ACCUM = $((BATCH_SIZE * GRAD_ACCUM))"
echo "Learning rate: $LEARNING_RATE"
echo "Epochs: $NUM_EPOCHS"
echo "LoRA: r=$LORA_R, alpha=$LORA_ALPHA"
echo "========================================="
echo ""

# ===================== Build Command =====================

CMD="python gemma3_finetune_unsloth.py \
    --model_name $MODEL_NAME \
    $LOAD_IN_4BIT \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --num_epochs $NUM_EPOCHS \
    --max_seq_length $MAX_SEQ_LENGTH \
    --warmup_ratio $WARMUP_RATIO \
    --max_grad_norm $MAX_GRAD_NORM \
    --weight_decay $WEIGHT_DECAY \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --num_frames $NUM_FRAMES \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name $WANDB_RUN_NAME \
    --output_dir $OUTPUT_DIR \
    $RUN_EVAL \
    --eval_steps $EVAL_STEPS \
    $SAVE_EVAL_CSV \
    $GENERATE_REPORT \
    $UPLOAD_TO_HF \
    $HF_PRIVATE"

# Add DeepSpeed config if specified
if [ -n "$DEEPSPEED_CONFIG" ] && [ -f "$DEEPSPEED_CONFIG" ]; then
    CMD="$CMD --deepspeed_config $DEEPSPEED_CONFIG"
    echo "⚡ Using DeepSpeed: $DEEPSPEED_CONFIG"
fi

# Add dataset-specific arguments
if [ "$DATASET_MODE" == "--local" ]; then
    echo "📂 Using local QVED dataset (prepared by dataset.py)"
    
    # Verify dataset exists
    if [ ! -f "$TRAIN_JSON" ]; then
        echo "❌ Error: Training data not found: $TRAIN_JSON"
        echo "Please run: python dataset.py download && python dataset.py prepare"
        echo "Or if already downloaded: python dataset.py prepare"
        exit 1
    fi
    
    CMD="$CMD --train_json $TRAIN_JSON"
    if [ -f "$VAL_JSON" ]; then
        CMD="$CMD --val_json $VAL_JSON"
    fi
elif [ "$DATASET_MODE" == "--hf" ]; then
    echo "🤗 Using HuggingFace dataset: $HF_DATASET (from local cache if available)"
    CMD="$CMD \
        --hf_dataset $HF_DATASET \
        --hf_split $HF_SPLIT \
        --num_samples $NUM_SAMPLES \
        --video_save_dir $VIDEO_SAVE_DIR"
else
    echo "❌ Invalid dataset mode. Use --local or --hf"
    exit 1
fi

# Add HF token if available
if [ -n "$HF_TOKEN" ]; then
    CMD="$CMD --hf_token $HF_TOKEN"
fi

# ===================== Create Output Directory =====================
mkdir -p "$OUTPUT_DIR"

# ===================== Run Training =====================
echo ""
echo "🚀 Starting fine-tuning..."
echo ""
echo "Command:"
echo "$CMD"
echo ""

eval $CMD

# ===================== Post-training =====================
echo ""
echo "========================================="
echo "✅ Fine-tuning completed!"
echo "========================================="
echo "📁 Model saved to: $OUTPUT_DIR"
echo "📁 Merged 16-bit model: ${OUTPUT_DIR}_merged_16bit"
if [ -n "$LOAD_IN_4BIT" ]; then
    echo "📁 Merged 4-bit model: ${OUTPUT_DIR}_merged_4bit"
fi
echo ""
echo "To run inference with the fine-tuned model:"
echo "python utils/infer_qved.py \\"
echo "  --model_path ${OUTPUT_DIR}_merged_16bit \\"
echo "  --video_path sample_videos/00000340.mp4"
echo ""
echo "To upload model to HuggingFace:"
echo "python utils/hf_upload.py \\"
echo "  --model_path ${OUTPUT_DIR}_merged_16bit \\"
echo "  --repo_name my-gemma3n-finetune"
echo "========================================="
