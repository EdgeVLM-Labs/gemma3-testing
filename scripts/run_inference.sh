#!/bin/bash

# Gemma-3N-E2B Test Inference and Evaluation Report Generator
# This script runs inference on the test set and generates an evaluation report

set -e  # Exit on error

echo "========================================="
echo "Gemma-3N-E2B Test Inference & Evaluation"
echo "========================================="

# Default values
MODEL_PATH=""
HF_REPO=""
TEST_JSON="dataset/qved_test.json"
DATA_PATH="dataset"
OUTPUT_DIR=""
DEVICE="cuda"
MAX_NEW_TOKENS=256
BASE_MODEL="google/gemma-3n-E2B"
LIMIT=""
NO_BERT=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --hf_repo)
            HF_REPO="$2"
            shift 2
            ;;
        --test_json)
            TEST_JSON="$2"
            shift 2
            ;;
        --data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --max_new_tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --base_model)
            BASE_MODEL="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --no-bert)
            NO_BERT="--no-bert"
            shift
            ;;
        -h|--help)
            echo "Usage: bash scripts/run_gemma3n_inference.sh [--model_path <path> | --hf_repo <repo>] [options]"
            echo ""
            echo "Model Source (one required):"
            echo "  --model_path      Path to local finetuned model checkpoint"
            echo "  --hf_repo         HuggingFace repository ID (e.g., google/gemma-3n-E2B-finetune)"
            echo ""
            echo "Optional:"
            echo "  --test_json       Path to test set JSON (default: dataset/qved_test.json)"
            echo "  --data_path       Base path for video files (default: dataset)"
            echo "  --output_dir      Output directory for results (default: model directory or results/hf_inference)"
            echo "  --device          Device to use: cuda/cpu (default: cuda)"
            echo "  --max_new_tokens  Max tokens to generate (default: 256)"
            echo "  --base_model      Base model for adapters (default: google/gemma-3n-E2B)"
            echo "  --limit           Limit number of samples (for testing)"
            echo "  --no-bert         Skip BERT similarity (faster evaluation)"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate: either model_path or hf_repo must be provided
if [ -z "$MODEL_PATH" ] && [ -z "$HF_REPO" ]; then
    echo "❌ Error: Either --model_path or --hf_repo is required"
    echo "Use --help for usage information"
    exit 1
fi

# If HF repo is provided, use it as model path
if [ -n "$HF_REPO" ]; then
    if [[ "$HF_REPO" == *"huggingface.co/"* ]]; then
        HF_REPO=$(echo "$HF_REPO" | sed 's|.*huggingface.co/||' | sed 's|/$||')
    fi
    echo "🤗 Using HuggingFace model: $HF_REPO"
    MODEL_PATH="$HF_REPO"

    if [ -z "$OUTPUT_DIR" ]; then
        REPO_NAME=$(echo "$HF_REPO" | sed 's|/|_|g')
        OUTPUT_DIR="results/hf_inference_${REPO_NAME}"
        mkdir -p "$OUTPUT_DIR"
    fi
else
    if [ ! -e "$MODEL_PATH" ]; then
        echo "❌ Error: Model path not found: $MODEL_PATH"
        exit 1
    fi
fi

if [ ! -f "$TEST_JSON" ]; then
    echo "❌ Error: Test JSON not found: $TEST_JSON"
    exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
    if [[ "$MODEL_PATH" == *"checkpoint-"* ]]; then
        OUTPUT_DIR=$(dirname "$MODEL_PATH")
    else
        OUTPUT_DIR="$MODEL_PATH"
    fi
fi

mkdir -p "$OUTPUT_DIR" 2>/dev/null || true

PREDICTIONS_FILE="${OUTPUT_DIR}/test_predictions.json"
REPORT_FILE="${OUTPUT_DIR}/test_evaluation_report.xlsx"

echo ""
echo "Configuration:"
echo "  Model path:      $MODEL_PATH"
echo "  Test JSON:       $TEST_JSON"
echo "  Data path:       $DATA_PATH"
echo "  Output dir:      $OUTPUT_DIR"
echo "  Device:          $DEVICE"
echo "  Max new tokens:  $MAX_NEW_TOKENS"
echo "  Base model:      $BASE_MODEL"
if [ -n "$LIMIT" ]; then
    echo "  Sample limit:    $LIMIT"
fi
echo "========================================="
echo ""

# Step 1: Run inference
echo "[Step 1/2] Running inference on test set..."
echo "========================================="

LIMIT_ARG=""
if [ -n "$LIMIT" ]; then
    LIMIT_ARG="--limit $LIMIT"
fi

python utils/test_inference_gemma3n.py \
    --model_path "$MODEL_PATH" \
    --test_json "$TEST_JSON" \
    --data_path "$DATA_PATH" \
    --output "$PREDICTIONS_FILE" \
    --device "$DEVICE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --base_model "$BASE_MODEL" \
    $LIMIT_ARG

echo ""
echo "✓ Inference complete! Predictions saved to: $PREDICTIONS_FILE"
echo ""

# Step 2: Generate evaluation report
echo "[Step 2/2] Generating evaluation report..."
echo "========================================="

python utils/generate_test_report.py \
    --predictions "$PREDICTIONS_FILE" \
    --output "$REPORT_FILE" \
    $NO_BERT

echo ""
echo "✓ Evaluation report saved to: $REPORT_FILE"
echo ""
echo "========================================="
echo "✅ All steps complete!"
echo "========================================="
echo ""
echo "Output files:"
echo "  Predictions: $PREDICTIONS_FILE"
echo "  Report:      $REPORT_FILE"
echo ""
echo "========================================="
