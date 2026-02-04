#!/bin/bash

# Script to generate training statistics plots for Gemma 3n

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Gemma 3n Training Stats Plotter${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Parse arguments
LOG_FILE=""
OUTPUT_DIR=""
MODEL_NAME="gemma3n-qved"

while [[ $# -gt 0 ]]; do
    case $1 in
        --log_file)
            LOG_FILE="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown argument: $1${NC}"
            echo "Usage: bash scripts/plot_training.sh [--log_file LOG] [--output_dir DIR] [--model_name NAME]"
            exit 1
            ;;
    esac
done

# Auto-detect log file if not provided
if [ -z "$LOG_FILE" ]; then
    echo -e "${YELLOW}Searching for training log files...${NC}"

    # Look in outputs directory
    if [ -d "outputs/${MODEL_NAME}" ]; then
        LOG_FILE=$(find "outputs/${MODEL_NAME}" -name "*.log" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    fi

    if [ -z "$LOG_FILE" ]; then
        echo -e "${RED}No log file found. Please specify with --log_file${NC}"
        exit 1
    fi

    echo -e "${GREEN}Found log file: $LOG_FILE${NC}"
fi

# Check if log file exists
if [ ! -f "$LOG_FILE" ]; then
    echo -e "${RED}Log file not found: $LOG_FILE${NC}"
    exit 1
fi

# Set output directory if not provided
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="plots/${MODEL_NAME}"
fi

echo ""
echo -e "${BLUE}Log file:${NC}     $LOG_FILE"
echo -e "${BLUE}Output dir:${NC}   $OUTPUT_DIR"
echo ""

# Run the plotter
echo -e "${YELLOW}Generating plots...${NC}"
python -m utils.plot_training_stats \
    --log_file "$LOG_FILE" \
    --output_dir "$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✅ Plots generated successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "View plots at: ${BLUE}$OUTPUT_DIR${NC}"
    echo ""
    echo "Generated files:"
    echo "  - training_report.pdf (full report)"
    echo "  - loss.png"
    echo "  - gradient_norm.png"
    echo "  - learning_rate.png"
    echo "  - combined_metrics.png"
    echo "  - training_summary.txt"
    echo ""
else
    echo -e "${RED}Failed to generate plots${NC}"
    exit 1
fi
