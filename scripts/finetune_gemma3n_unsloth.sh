#!/bin/bash

# Gemma-3N E4B Fine-tuning Script with Unsloth
# This script fine-tunes unsloth/gemma-3n-E4B-it on QVED dataset

# ===================== Environment Setup =====================
export PYTHONPATH="./:$PYTHONPATH"

# WandB Configuration
export WANDB_PROJECT="gemma3n-finetune"
export WANDB_NAME="gemma3n-E4B-finetune-$(date +%Y%m%d_%H%M%S)"

# ===================== Configuration =====================
MODEL_NAME="unsloth/gemma-3n-E4B-it"
TRAIN_JSON="dataset/qved_train.json"
VAL_JSON="dataset/qved_val.json"
OUTPUT_DIR="results/gemma3n_E4B_finetune"

# Training Hyperparameters
NUM_FRAMES=8
BATCH_SIZE=2
GRADIENT_ACCUMULATION=4
LEARNING_RATE=2e-4
NUM_EPOCHS=3
MAX_SEQ_LENGTH=2048

# LoRA Configuration
LORA_R=16
LORA_ALPHA=32

echo "========================================="
echo "Gemma-3N E4B Fine-tuning (Unsloth)"
echo "========================================="
echo "Model: $MODEL_NAME"
echo "Training data: $TRAIN_JSON"
echo "Output: $OUTPUT_DIR"
echo "Epochs: $NUM_EPOCHS"
echo "Learning Rate: $LEARNING_RATE"
echo "Batch Size: $BATCH_SIZE x $GRADIENT_ACCUMULATION = $((BATCH_SIZE * GRADIENT_ACCUMULATION))"
echo "LoRA: r=$LORA_R, alpha=$LORA_ALPHA"
echo "========================================="

# Verify dataset exists
if [ ! -f "$TRAIN_JSON" ]; then
    echo "❌ Error: Training data not found: $TRAIN_JSON"
    echo "Please run: python dataset.py download && python dataset.py prepare"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training
python gemma3_finetune_unsloth.py \
    --model_name "$MODEL_NAME" \
    --train_json "$TRAIN_JSON" \
    --val_json "$VAL_JSON" \
    --output_dir "$OUTPUT_DIR" \
    --num_frames "$NUM_FRAMES" \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation "$GRADIENT_ACCUMULATION" \
    --learning_rate "$LEARNING_RATE" \
    --num_epochs "$NUM_EPOCHS" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "$WANDB_NAME"

echo ""
echo "========================================="
echo "✅ Fine-tuning completed!"
echo "LoRA adapter: $OUTPUT_DIR"
echo "Merged model: ${OUTPUT_DIR}_merged"
echo ""
echo "To run inference with the fine-tuned model:"
echo "python utils/infer_qved.py \\"
echo "  --model_path ${OUTPUT_DIR}_merged \\"
echo "  --video_path sample_videos/00000340.mp4"
echo "========================================="
