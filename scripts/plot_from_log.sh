#!/bin/bash

# Quick script to generate training plots for google/gemma-3n-E2B
# Usage: bash scripts/plot_from_log_gemma3n.sh [log_file]

set -e

echo "========================================="
echo "Training Statistics Plotter - Gemma 3N E2B"
echo "========================================="

# Auto-detect latest log file if not provided
if [ -z "$1" ]; then
    LOG_FILE=$(find results/gemma3n_E2B_finetune/ -name "training_*.log" -type f 2>/dev/null | sort -r | head -1)

    if [ -z "$LOG_FILE" ]; then
        echo "❌ No log files found in results/gemma3n_E2B_finetune/"
        echo ""
        echo "Usage: bash scripts/plot_from_log_gemma3n.sh [log_file]"
        echo ""
        echo "Example:"
        echo "  bash scripts/plot_from_log_gemma3n.sh results/gemma3n_E2B_finetune/training_20260104_120000.log"
        exit 1
    fi

    echo "Auto-detected log file: $LOG_FILE"
else
    LOG_FILE="$1"

    if [ ! -f "$LOG_FILE" ]; then
        echo "❌ Log file not found: $LOG_FILE"
        exit 1
    fi
fi

OUTPUT_DIR="plots/gemma3n_E2B_finetune"

echo "Log file: $LOG_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "========================================="

# Activate conda environment (optional)
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate gemma3n 2>/dev/null || true
fi

# Generate plots
python utils/plot_training_stats.py \
    --log_file "$LOG_FILE" \
    --model_name "google/gemma-3n-E2B" \
    --output_dir "$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✓ Plots generated successfully!"
    echo ""
    echo "Generated files in $OUTPUT_DIR:"
    echo "  - loss.pdf"
    echo "  - gradient_norm.pdf"
    echo "  - learning_rate.pdf"
    echo "  - combined_metrics.pdf"
    echo "  - training_summary.txt"
    echo ""
    echo "PNG versions also saved for quick preview."
    echo "========================================="
else
    echo "❌ Failed to generate plots"
    exit 1
fi
