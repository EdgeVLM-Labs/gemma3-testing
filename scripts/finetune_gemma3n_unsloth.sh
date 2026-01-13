#!/bin/bash
# ============================================================================
# Gemma-3N Fine-tuning Script with Unsloth
# ============================================================================
# Production-ready fine-tuning with exact hyperparameters specified
#
# Usage:
#   bash finetune_gemma3n.sh
#
# For quick testing (100 samples):
#   bash finetune_gemma3n.sh --test
# ============================================================================

set -e  # Exit on error

# ==================== Configuration ====================

# Parse command line arguments
TEST_MODE=false
if [[ "$1" == "--test" ]]; then
    TEST_MODE=true
fi

# Model
MODEL_NAME="unsloth/gemma-3n-E2B-it"
LOAD_IN_4BIT=""  # Add "--load_in_4bit" for 4-bit mode

# Dataset paths
TRAIN_JSON="dataset/qved_train.json"
VAL_JSON="dataset/qved_val.json"
VIDEO_DIR="test_videos"

# Test mode: limit samples
if [ "$TEST_MODE" = true ]; then
    MAX_TRAIN_SAMPLES="--max_train_samples 100"
    MAX_VAL_SAMPLES="--max_val_samples 10"
    echo "⚡ TEST MODE: Training on 100 samples only"
else
    MAX_TRAIN_SAMPLES=""
    MAX_VAL_SAMPLES="--max_val_samples 20"  # Limit validation to 20 for faster eval
fi

# Output
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/gemma3n_${TIMESTAMP}"

# Hyperparameters (as specified)
BATCH_SIZE=8
GRAD_ACCUM=4
LEARNING_RATE=2e-4
PROJECTOR_LR=1e-4
NUM_EPOCHS=3
MAX_SEQ_LENGTH=4096  # Increased for video frames with multiple image tokens
WARMUP_RATIO=0.05
MAX_GRAD_NORM=0.3
WEIGHT_DECAY=0.001
DATALOADER_NUM_WORKERS=0  # Set to 0 to avoid hangs with large datasets
PER_DEVICE_EVAL_BATCH_SIZE=8

# LoRA configuration
LORA_R=64
LORA_ALPHA=128
LORA_DROPOUT=0.0

# Video processing
NUM_FRAMES=8  # Reduced from 16 to save memory
RESOLUTION=224
NUM_WORKERS=8  # Parallel video processing

# Checkpointing
SAVE_STEPS=30
EVAL_STEPS=0   # Auto-calculate (0 = auto)
LOGGING_STEPS=0  # Auto-calculate (0 = auto)

# Evaluation
RUN_EVAL="--run_eval"  # Enable evaluation

# WandB (set to your values)
USE_WANDB="--use_wandb"
WANDB_PROJECT="gemma3n-finetune"
WANDB_ENTITY="fyp-21"  # Your wandb username/team
WANDB_RUN_NAME="gemma3n-${TIMESTAMP}"

# HuggingFace (optional)
UPLOAD_TO_HF=""  # Set to "--upload_to_hf" to auto-upload
HF_REPO_NAME=""  # Leave empty for auto-name
HF_TOKEN="${HF_TOKEN:-}"  # From environment variable

# ==================== Environment Setup ====================

export PYTHONPATH="./:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false

# ==================== Pre-flight Checks ====================

echo ""
echo "=============================================="
echo "Gemma-3N Fine-tuning with Unsloth"
echo "=============================================="
echo ""

# Check if dataset exists
if [ ! -f "$TRAIN_JSON" ]; then
    echo "❌ Error: Training data not found: $TRAIN_JSON"
    echo ""
    echo "Please run:"
    echo "  python dataset.py download"
    echo "  python dataset.py prepare"
    exit 1
fi

# Check if video directory exists
if [ ! -d "$VIDEO_DIR" ]; then
    echo "⚠️  Warning: Video directory not found: $VIDEO_DIR"
    echo "   Proceeding anyway (script will search multiple locations)"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# ==================== Build Command ====================

CMD="python gemma3_finetune_unsloth.py \
    --model_name $MODEL_NAME \
    $LOAD_IN_4BIT \
    --train_json $TRAIN_JSON \
    --video_dir $VIDEO_DIR \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --projector_lr $PROJECTOR_LR \
    --num_epochs $NUM_EPOCHS \
    --max_seq_length $MAX_SEQ_LENGTH \
    --warmup_ratio $WARMUP_RATIO \
    --max_grad_norm $MAX_GRAD_NORM \
    --weight_decay $WEIGHT_DECAY \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --num_frames $NUM_FRAMES \
    --resolution $RESOLUTION \
    --num_workers $NUM_WORKERS \
    --dataloader_num_workers $DATALOADER_NUM_WORKERS \
    --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
    --save_steps $SAVE_STEPS \
    --eval_steps $EVAL_STEPS \
    --logging_steps $LOGGING_STEPS \
    $MAX_TRAIN_SAMPLES \
    $MAX_VAL_SAMPLES \
    $RUN_EVAL"

# Add validation set if exists
if [ -f "$VAL_JSON" ]; then
    CMD="$CMD --val_json $VAL_JSON"
fi

# Add WandB if enabled
if [ -n "$USE_WANDB" ]; then
    CMD="$CMD $USE_WANDB \
        --wandb_project $WANDB_PROJECT \
        --wandb_entity $WANDB_ENTITY \
        --wandb_run_name $WANDB_RUN_NAME"
fi

# Add HF token if available
if [ -n "$HF_TOKEN" ]; then
    CMD="$CMD --hf_token $HF_TOKEN"
fi

# Add HF upload if enabled
if [ -n "$UPLOAD_TO_HF" ]; then
    CMD="$CMD $UPLOAD_TO_HF"
    if [ -n "$HF_REPO_NAME" ]; then
        CMD="$CMD --hf_repo_name $HF_REPO_NAME"
    fi
fi

# ==================== Display Configuration ====================

echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Output: $OUTPUT_DIR"
echo "  Dataset: $TRAIN_JSON"
echo "  Video Dir: $VIDEO_DIR"
echo ""
echo "Hyperparameters:"
echo "  Frames: $NUM_FRAMES @ ${RESOLUTION}x${RESOLUTION}"
echo "  Batch: $BATCH_SIZE x $GRAD_ACCUM = $((BATCH_SIZE * GRAD_ACCUM))"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Projector LR: $PROJECTOR_LR"
echo "  Epochs: $NUM_EPOCHS"
echo "  LoRA: r=$LORA_R, alpha=$LORA_ALPHA"
echo "  Max Seq Length: $MAX_SEQ_LENGTH"
echo "  Warmup Ratio: $WARMUP_RATIO"
echo ""
echo "Checkpointing:"
echo "  Save every: $SAVE_STEPS steps"
echo "  Eval every: auto-calculated"
echo "  Log every: auto-calculated"
echo ""
echo "Workers:"
echo "  Video processing: $NUM_WORKERS parallel"
echo "  DataLoader: $DATALOADER_NUM_WORKERS"
echo ""
echo "=============================================="
echo ""

# ==================== Run Training ====================

echo "🚀 Starting fine-tuning..."
echo ""
echo "Command:"
echo "$CMD"
echo ""
echo "=============================================="
echo ""

# Run training
eval $CMD

# ==================== Post-training ====================

echo ""
echo "=============================================="
echo "✅ Fine-tuning Complete!"
echo "=============================================="
echo ""
echo "Outputs:"
echo "  LoRA adapter: $OUTPUT_DIR"
echo "  Merged model: ${OUTPUT_DIR}_merged_16bit"
echo ""
echo "Next steps:"
echo ""
echo "1. Test the model:"
echo "   python utils/infer_qved.py \\"
echo "     --model_path ${OUTPUT_DIR}_merged_16bit \\"
echo "     --video_path sample.mp4"
echo ""
echo "2. Upload to HuggingFace:"
echo "   python utils/hf_upload.py \\"
echo "     --model_path ${OUTPUT_DIR}_merged_16bit \\"
echo "     --repo_name my-gemma3n-finetune"
echo ""
echo "=============================================="