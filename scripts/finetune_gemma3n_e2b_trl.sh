#!/bin/bash
###############################################################################
# Gemma-3n-E2B-it Fine-tuning Script for QVED Dataset
# 
# USAGE:
#   Basic training with default settings:
#     bash scripts/finetune_gemma3n_e2b_trl.sh
#
#   Custom dataset paths:
#     TRAIN_JSON=data/custom_train.json \
#     VAL_JSON=data/custom_val.json \
#     VIDEO_PATH=data/videos \
#     bash scripts/finetune_gemma3n_e2b_trl.sh
#
#   Custom hyperparameters:
#     LEARNING_RATE=3e-4 \
#     LORA_R=128 \
#     EPOCHS=5 \
#     bash scripts/finetune_gemma3n_e2b_trl.sh
#
#   Disable wandb:
#     WANDB_MODE=disabled bash scripts/finetune_gemma3n_e2b_trl.sh
#
# REQUIREMENTS:
#   - Python environment with transformers, trl, peft, timm, opencv-python
#   - CUDA-capable GPU recommended
#   - Run: pip install -r requirements.txt
#
###############################################################################

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================================================${NC}"
echo -e "${BLUE}Gemma-3n-E2B-it Fine-tuning on QVED Dataset${NC}"
echo -e "${BLUE}========================================================================${NC}"

# ============================================================================
# Configuration - Override with environment variables
# ============================================================================

# Model configuration
MODEL_PATH="${MODEL_PATH:-google/gemma-3n-E2B-it}"

# Dataset paths
TRAIN_JSON="${TRAIN_JSON:-data/qved_train.json}"
VAL_JSON="${VAL_JSON:-data/qved_val.json}"
VIDEO_PATH="${VIDEO_PATH:-videos}"

# Output directory
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/gemma3n-e2b-qved-ft-$(date +%Y%m%d_%H%M%S)}"

# Training hyperparameters
NUM_FRAMES="${NUM_FRAMES:-8}"            # Extract 8 frames per video
EPOCHS="${EPOCHS:-3}"                     # 3 epochs
LEARNING_RATE="${LEARNING_RATE:-2e-4}"   # 2e-4 LR
BATCH_SIZE="${BATCH_SIZE:-8}"            # Batch size 8
GRAD_ACCUM="${GRAD_ACCUM:-4}"            # Gradient accumulation 4 (effective batch size 32)
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"       # Max sequence length 1024 (memory efficient for 80GB GPU)

# LoRA configuration
LORA_R="${LORA_R:-64}"                   # LoRA r=64
LORA_ALPHA="${LORA_ALPHA:-128}"          # LoRA alpha=128
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"     # LoRA dropout=0.05

# Training configuration
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"     # Warmup ratio 0.05
SAVE_STEPS="${SAVE_STEPS:-30}"           # Save every 30 steps
EVAL_STRATEGY="${EVAL_STRATEGY:-steps}"  # Evaluate by steps
DATALOADER_WORKERS="${DATALOADER_WORKERS:-2}"  # 2 workers
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"  # Eval batch size 8 (memory efficient)

# Wandb configuration
WANDB_PROJECT="${WANDB_PROJECT:-gemma3n-qved-finetuning}"
RUN_NAME="${RUN_NAME:-gemma3n-e2b-lr${LEARNING_RATE}-r${LORA_R}-epochs${EPOCHS}}"

# Other settings
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"  # Empty = train from scratch
SEED="${SEED:-42}"

# ============================================================================
# Validation
# ============================================================================

echo -e "\n${YELLOW}Validating environment...${NC}"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 not found. Please install Python 3.8+${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python3 found${NC}"

# Check if CUDA is available (optional but recommended)
if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
    echo -e "${GREEN}✓ CUDA available with ${GPU_COUNT} GPU(s)${NC}"
else
    echo -e "${YELLOW}⚠️  CUDA not available. Training will be slow on CPU.${NC}"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if training data exists
if [ ! -f "$TRAIN_JSON" ]; then
    echo -e "${RED}❌ Training JSON not found: ${TRAIN_JSON}${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Training data found: ${TRAIN_JSON}${NC}"

# Check if validation data exists (optional)
if [ -n "$VAL_JSON" ] && [ ! -f "$VAL_JSON" ]; then
    echo -e "${YELLOW}⚠️  Validation JSON not found: ${VAL_JSON}${NC}"
    echo -e "${YELLOW}   Training will proceed without validation${NC}"
    VAL_JSON=""
fi

if [ -n "$VAL_JSON" ]; then
    echo -e "${GREEN}✓ Validation data found: ${VAL_JSON}${NC}"
fi

# Check if video directory exists
if [ ! -d "$VIDEO_PATH" ]; then
    echo -e "${RED}❌ Video directory not found: ${VIDEO_PATH}${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Video directory found: ${VIDEO_PATH}${NC}"

# Check required Python packages
echo -e "\n${YELLOW}Checking Python dependencies...${NC}"
REQUIRED_PACKAGES=("torch" "transformers" "trl" "peft" "timm" "cv2" "PIL")
MISSING_PACKAGES=()

for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! python3 -c "import ${package}" 2>/dev/null; then
        MISSING_PACKAGES+=("${package}")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -ne 0 ]; then
    echo -e "${RED}❌ Missing required packages: ${MISSING_PACKAGES[*]}${NC}"
    echo -e "${YELLOW}   Install with: pip install -r requirements.txt${NC}"
    exit 1
fi
echo -e "${GREEN}✓ All required packages installed${NC}"

# ============================================================================
# Display Configuration
# ============================================================================

echo -e "\n${BLUE}========================================================================${NC}"
echo -e "${BLUE}Training Configuration${NC}"
echo -e "${BLUE}========================================================================${NC}"
echo -e "Model:                     ${MODEL_PATH}"
echo -e "Training data:             ${TRAIN_JSON}"
echo -e "Validation data:           ${VAL_JSON:-None}"
echo -e "Video directory:           ${VIDEO_PATH}"
echo -e "Output directory:          ${OUTPUT_DIR}"
echo -e ""
echo -e "${BLUE}Hyperparameters:${NC}"
echo -e "  Frames per video:        ${NUM_FRAMES}"
echo -e "  Epochs:                  ${EPOCHS}"
echo -e "  Learning rate:           ${LEARNING_RATE}"
echo -e "  Batch size:              ${BATCH_SIZE}"
echo -e "  Gradient accumulation:   ${GRAD_ACCUM}"
echo -e "  Max sequence length:     ${MAX_SEQ_LEN}"
echo -e "  LoRA r:                  ${LORA_R}"
echo -e "  LoRA alpha:              ${LORA_ALPHA}"
echo -e "  LoRA dropout:            ${LORA_DROPOUT}"
echo -e "  Warmup ratio:            ${WARMUP_RATIO}"
echo -e "  Save steps:              ${SAVE_STEPS}"
echo -e "  Eval strategy:           ${EVAL_STRATEGY}"
echo -e "  Eval batch size:         ${EVAL_BATCH_SIZE}"
echo -e "  Dataloader workers:      ${DATALOADER_WORKERS}"
echo -e ""
echo -e "${BLUE}Wandb:${NC}"
echo -e "  Project:                 ${WANDB_PROJECT}"
echo -e "  Run name:                ${RUN_NAME}"
echo -e "  Mode:                    ${WANDB_MODE:-online}"
echo -e "${BLUE}========================================================================${NC}"

# ============================================================================
# Confirmation
# ============================================================================

echo -e "\n${YELLOW}Ready to start training.${NC}"
read -p "Continue? (Y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Nn]$ ]]; then
    echo -e "${YELLOW}Training cancelled.${NC}"
    exit 0
fi

# ============================================================================
# Build command
# ============================================================================

CMD="python3 finetune_gemma3n_e2b_trl.py \
    --model_path ${MODEL_PATH} \
    --train_json ${TRAIN_JSON} \
    --data_path ${VIDEO_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --num_frames ${NUM_FRAMES} \
    --num_train_epochs ${EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${EVAL_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --max_seq_length ${MAX_SEQ_LEN} \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --warmup_ratio ${WARMUP_RATIO} \
    --save_steps ${SAVE_STEPS} \
    --eval_strategy ${EVAL_STRATEGY} \
    --dataloader_num_workers ${DATALOADER_WORKERS} \
    --gradient_checkpointing \
    --wandb_project ${WANDB_PROJECT} \
    --run_name ${RUN_NAME} \
    --seed ${SEED}"

# Add validation dataset if provided
if [ -n "$VAL_JSON" ]; then
    CMD="${CMD} --val_json ${VAL_JSON}"
fi

# Add resume checkpoint if provided
if [ -n "$RESUME_CHECKPOINT" ]; then
    CMD="${CMD} --resume_from_checkpoint ${RESUME_CHECKPOINT}"
    echo -e "${BLUE}Resuming from checkpoint: ${RESUME_CHECKPOINT}${NC}"
fi

# Disable wandb if requested
if [ "${WANDB_MODE}" = "disabled" ]; then
    CMD="${CMD} --no_wandb"
fi

# ============================================================================
# Run training
# ============================================================================

echo -e "\n${GREEN}========================================================================${NC}"
echo -e "${GREEN}Starting training...${NC}"
echo -e "${GREEN}========================================================================${NC}\n"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Save configuration
cat > "${OUTPUT_DIR}/config.txt" << EOF
Training Configuration
Generated: $(date)

Model: ${MODEL_PATH}
Training data: ${TRAIN_JSON}
Validation data: ${VAL_JSON:-None}
Video directory: ${VIDEO_PATH}
Output directory: ${OUTPUT_DIR}

Hyperparameters:
  Frames per video: ${NUM_FRAMES}
  Epochs: ${EPOCHS}
  Learning rate: ${LEARNING_RATE}
  Batch size: ${BATCH_SIZE}
  Gradient accumulation: ${GRAD_ACCUM}
  Max sequence length: ${MAX_SEQ_LEN}
  LoRA r: ${LORA_R}
  LoRA alpha: ${LORA_ALPHA}
  LoRA dropout: ${LORA_DROPOUT}
  Warmup ratio: ${WARMUP_RATIO}
  Save steps: ${SAVE_STEPS}
  Eval strategy: ${EVAL_STRATEGY}
  Eval batch size: ${EVAL_BATCH_SIZE}
  Dataloader workers: ${DATALOADER_WORKERS}

Wandb:
  Project: ${WANDB_PROJECT}
  Run name: ${RUN_NAME}
  Mode: ${WANDB_MODE:-online}

Command:
${CMD}
EOF

echo -e "${BLUE}Configuration saved to: ${OUTPUT_DIR}/config.txt${NC}\n"

# Set PyTorch memory allocation optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo -e "${GREEN}✓ PyTorch memory optimization enabled (expandable_segments:True)${NC}\n"

# Run the training command
eval ${CMD}

# ============================================================================
# Post-training
# ============================================================================

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}========================================================================${NC}"
    echo -e "${GREEN}✅ Training completed successfully!${NC}"
    echo -e "${GREEN}========================================================================${NC}"
    echo -e "${GREEN}Model saved to: ${OUTPUT_DIR}${NC}"
    echo -e "\n${YELLOW}Next steps:${NC}"
    echo -e "  1. Evaluate the model:"
    echo -e "     python utils/test_inference_transformers.py \\"
    echo -e "       --model_path ${OUTPUT_DIR} \\"
    echo -e "       --test_json data/qved_test.json \\"
    echo -e "       --data_path ${VIDEO_PATH} \\"
    echo -e "       --output results.json"
    echo -e "\n  2. Run inference on new videos:"
    echo -e "     python scripts/run_inference_transformers.sh ${OUTPUT_DIR}"
    echo -e "\n  3. Upload to HuggingFace Hub:"
    echo -e "     python utils/hf_upload.py --model_path ${OUTPUT_DIR}"
else
    echo -e "\n${RED}========================================================================${NC}"
    echo -e "${RED}❌ Training failed!${NC}"
    echo -e "${RED}========================================================================${NC}"
    echo -e "${RED}Check the logs above for error details.${NC}"
    exit 1
fi
