#!/bin/bash

# Gemma-3N-E2B Finetuning - Quick Start
# This script runs all necessary steps to start finetuning the gemma-3n-E2B model

set -e  # Exit on error

# ==========================
# Setup logging
# ==========================
mkdir -p results plots
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="results/finetune_gemma3n_E2B_${TIMESTAMP}.log"
echo "Logging all output to: $LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================="
echo "Gemma-3N-E2B Finetuning - Quick Start"
echo "========================================="

# ==========================
# Step 1: Verify setup
# ==========================
echo -e "\n[Step 1/4] Verifying setup..."
bash scripts/verify_qved_setup.sh || { echo "Setup verification failed"; exit 1; }

# ==========================
# Step 2: Confirm to proceed
# ==========================
echo -e "\n========================================="
echo -n "Setup verified! Start finetuning Gemma-3N-E2B? (y/N): "
read -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# ==========================
# Step 3: Start finetuning
# ==========================
echo -e "\n[Step 2/4] Activating environment and starting finetuning..."
bash scripts/finetune_qved.sh || { echo "Finetuning failed"; exit 1; }

# Generate training plots
echo ""
echo "Generating training plots..."
python utils/plot_training_stats.py \
  --log_file "$LOG_FILE" \
  --model_name "gemma3n_E2B_finetune"

if [ $? -eq 0 ]; then
    echo "✓ Training plots generated successfully!"
    echo "  Location: plots/gemma3n_E2B_finetune/"
else
    echo "⚠ Warning: Failed to generate plots. You can generate them later with:"
    echo "  python utils/plot_training_stats.py --log_file $LOG_FILE"
fi

echo -e "\n========================================="
echo "[Step 3/4] Finetuning complete!"
echo "========================================="
echo "Model saved to: results/gemma3n_E2B_finetune/"
echo ""

# ==========================
# Step 4: Find latest checkpoint
# ==========================
echo "Finding latest checkpoint..."
LATEST_CKPT=$(ls -d results/gemma3n_E2B_finetune/checkpoint-* 2>/dev/null | sort -V | tail -1)
if [ -n "$LATEST_CKPT" ]; then
    echo "Latest checkpoint: $LATEST_CKPT"
    MODEL_PATH="$LATEST_CKPT"
else
    echo "Note: LoRA adapters saved in results/gemma3n_E2B_finetune/"
    MODEL_PATH="results/gemma3n_E2B_finetune"
fi

# ==========================
# Step 5: Upload to HuggingFace
# ==========================
echo -e "\n========================================="
echo "[Step 4/4] Upload to HuggingFace"
echo "========================================="
echo -n "Upload finetuned model to HuggingFace? (y/N): "
read -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    HF_REPO_NAME="gemma3n_E2B_finetune_${TIMESTAMP}"
    echo "Uploading to your HuggingFace organization/repo: $HF_REPO_NAME ..."
    python utils/hf_upload.py \
        --model_path "$MODEL_PATH" \
        --repo_name "$HF_REPO_NAME" \
        --org "YOUR_HF_ORG_NAME"

    if [ $? -eq 0 ]; then
        echo "✓ Model uploaded successfully to HuggingFace!"
        echo "  URL: https://huggingface.co/YOUR_HF_ORG_NAME/${HF_REPO_NAME}"
    else
        echo "⚠ Warning: Failed to upload model to HuggingFace"
        echo "  You can upload manually later with:"
        echo "  python utils/hf_upload.py --model_path $MODEL_PATH"
    fi
else
    echo "Skipping HuggingFace upload."
    echo "You can upload later with:"
    echo "  python utils/hf_upload.py --model_path $MODEL_PATH"
fi

# ==========================
# Step 6: Instructions for inference
# ==========================
echo -e "\n========================================="
echo "All steps complete!"
echo "========================================="
echo ""
echo "To use the finetuned Gemma-3N-E2B model for inference:"
echo "  python utils/infer_qved.py \\"
echo "    --model_path $MODEL_PATH \\"
echo "    --video_path sample_videos/00000340.mp4"
echo ""
echo "Adjustable parameters in utils/infer_qved.py:"
echo "  --model_path       Path to model checkpoint (default: google/gemma-3n-E2B)"
echo "  --video_path       Path to video file (default: sample_videos/00000340.mp4)"
echo "  --prompt           Custom prompt (default: physiotherapy evaluation prompt)"
echo "  --device           Device to use (default: cuda, options: cuda/cpu)"
echo "  --max_new_tokens   Max tokens to generate (default: 512)"
echo ""
echo "To run inference via script:"
echo "Using local checkpoint:"
echo "  bash scripts/run_inference.sh --model_path $MODEL_PATH"
echo ""
echo "Using HuggingFace model:"
echo "  bash scripts/run_inference.sh --hf_repo YOUR_HF_ORG_NAME/${HF_REPO_NAME}"
echo ""
echo "========================================="
