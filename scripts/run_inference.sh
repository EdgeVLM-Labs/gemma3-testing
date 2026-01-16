#!/bin/bash

# Gemma-3N Test Inference and Evaluation Report Generator (Unsloth)
# This script runs inference on the test set and generates an evaluation report

set -e  # Exit on error

echo "========================================="
echo "Gemma-3N Test Inference & Evaluation"
echo "========================================="

# Default values
MODEL_PATH=""
HF_REPO=""
TEST_JSON="dataset/qved_test.json"
DATA_PATH="videos"
OUTPUT_DIR=""
DEVICE="cuda"
MAX_NEW_TOKENS=256
NUM_FRAMES=8
LIMIT="50"
NO_BERT="--no-bert"  # Skip BERT by default to avoid network timeouts

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
        --num_frames)
            NUM_FRAMES="$2"
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
        --use-bert)
            NO_BERT=""
            shift
            ;;
        -h|--help)
            echo "Usage: bash scripts/run_inference.sh [--model_path <path> | --hf_repo <repo>] [options]"
            echo ""
            echo "Model Source:"
            echo "  --model_path      Path to local finetuned model checkpoint (default: unsloth/gemma-3n-E2B-it)"
            echo "  --hf_repo         HuggingFace repository ID (alternative to model_path)"
            echo ""
            echo "Optional:"
            echo "  --test_json       Path to test set JSON (default: dataset/qved_test.json)"
            echo "  --data_path       Base path for video files (default: videos)"
            echo "  --output_dir      Output directory for results (default: auto-generated)"
            echo "  --device          Device to use: cuda/cpu (default: cuda)"
            echo "  --max_new_tokens  Max tokens to generate (default: 256)"
            echo "  --num_frames      Frames to extract per video (default: 8)"
            echo "  --limit           Limit number of samples (default: 50)"
            echo "  --no-bert         Skip BERT similarity (enabled by default)"
            echo "  --use-bert        Enable BERT similarity calculation"
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

# If HF repo is provided, use it as model path
if [ -n "$HF_REPO" ]; then
    if [[ "$HF_REPO" == *"huggingface.co/"* ]]; then
        HF_REPO=$(echo "$HF_REPO" | sed 's|.*huggingface.co/||' | sed 's|/$||')
    fi
    echo "🤗 Using HuggingFace model: $HF_REPO"
    MODEL_PATH="$HF_REPO"
fi

# Use default model if none specified
if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH="unsloth/gemma-3n-E2B-it"
    echo "📦 Using default model: $MODEL_PATH"
fi

if [ ! -f "$TEST_JSON" ]; then
    echo "❌ Error: Test JSON not found: $TEST_JSON"
    exit 1
fi

# Set output directory
if [ -z "$OUTPUT_DIR" ]; then
    MODEL_NAME=$(basename "$MODEL_PATH" | sed 's|/|_|g')
    OUTPUT_DIR="results/test_inference_${MODEL_NAME}"
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
echo "  Frames/video:    $NUM_FRAMES"
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

set +e  # Don't exit on error for inference step
python utils/test_inference_unsloth.py \
    --model_path "$MODEL_PATH" \
    --test_json "$TEST_JSON" \
    --data_path "$DATA_PATH" \
    --output "$PREDICTIONS_FILE" \
    --device "$DEVICE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --num_frames "$NUM_FRAMES" \
    $LIMIT_ARG

INFERENCE_EXIT_CODE=$?
set -e  # Re-enable exit on error

if [ $INFERENCE_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ Inference complete! Predictions saved to: $PREDICTIONS_FILE"
    echo ""
else
    echo ""
    echo "⚠ Inference completed with errors (exit code: $INFERENCE_EXIT_CODE)"
    echo "  Predictions may be incomplete: $PREDICTIONS_FILE"
    echo ""
fi

# Step 2: Generate evaluation report (always run if predictions file exists)
if [ -f "$PREDICTIONS_FILE" ]; then
    echo "[Step 2/2] Generating evaluation report..."
    echo "========================================="

    set +e  # Don't exit on error for report generation
    python utils/generate_test_report.py \
        --predictions "$PREDICTIONS_FILE" \
        --output "$REPORT_FILE" \
        $NO_BERT

    REPORT_EXIT_CODE=$?
    set -e  # Re-enable exit on error

    if [ $REPORT_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Evaluation report saved to: $REPORT_FILE"
        echo ""
    else
        echo ""
        echo "⚠ Report generation failed (exit code: $REPORT_EXIT_CODE)"
        echo "  You can manually generate the report with:"
        echo "    python utils/generate_test_report.py --predictions $PREDICTIONS_FILE --output $REPORT_FILE"
        echo ""
    fi
else
    echo ""
    echo "⚠ Skipping report generation - predictions file not found: $PREDICTIONS_FILE"
    echo ""
fi

echo "========================================="
echo "✅ Process complete!"
echo "========================================="
echo ""
echo "Output files:"
echo "  Predictions: $PREDICTIONS_FILE"
if [ -f "$REPORT_FILE" ]; then
    echo "  Report:      $REPORT_FILE"
else
    echo "  Report:      (not generated)"
fi
echo "  Report:      $REPORT_FILE"
echo ""
echo "========================================="
