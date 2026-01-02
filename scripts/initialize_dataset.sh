#!/bin/bash

# Script to initialize QVED dataset for Gemma 3n training
# Orchestrates the complete dataset preparation pipeline

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Gemma 3n QVED Dataset Initialization${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Step 1: Ask for number of videos per exercise
echo -e "${YELLOW}Step 1: Dataset Download Configuration${NC}"
echo -n "Enter number of videos to download per exercise class [default: 5]: "
read -r VIDEO_COUNT

# Set default if empty
if [ -z "$VIDEO_COUNT" ]; then
    VIDEO_COUNT=5
fi

# Validate input
if ! [[ "$VIDEO_COUNT" =~ ^[0-9]+$ ]] || [ "$VIDEO_COUNT" -lt 1 ]; then
    echo -e "${RED}Error: Please enter a valid positive number${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Will download ${VIDEO_COUNT} videos per exercise class${NC}"
echo ""

# Step 2: Download dataset
echo -e "${YELLOW}Step 2: Downloading Dataset from HuggingFace${NC}"
echo -e "${BLUE}Running: python -m utils.load_dataset ${VIDEO_COUNT}${NC}"
python -m utils.load_dataset "$VIDEO_COUNT"

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Dataset download failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Dataset download completed${NC}"
echo ""

# Step 3: Filter ground truth
echo -e "${YELLOW}Step 3: Filtering Ground Truth Labels${NC}"
echo -e "${BLUE}Running: python -m utils.filter_ground_truth${NC}"
python -m utils.filter_ground_truth

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Ground truth filtering failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Ground truth filtering completed${NC}"
echo ""

# Step 4: Convert to Gemma format and split
echo -e "${YELLOW}Step 4: Converting Dataset to Gemma 3n Format${NC}"
echo -e "${BLUE}Running: python -m utils.convert_dataset${NC}"
python -m utils.convert_dataset

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Dataset conversion failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Dataset conversion completed${NC}"
echo ""

# Step 5: Verify setup
echo -e "${YELLOW}Step 5: Verifying Dataset Setup${NC}"
echo ""

# Check for required files
REQUIRED_FILES=(
    "dataset/gemma_train.jsonl"
    "dataset/gemma_val.jsonl"
    "dataset/gemma_test.jsonl"
    "dataset/manifest.json"
    "dataset/ground_truth.json"
)

ALL_GOOD=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        # Count samples for JSONL files
        if [[ "$file" == *.jsonl ]]; then
            num_samples=$(wc -l < "$file" 2>/dev/null || echo "?")
            echo -e "${GREEN}✓${NC} $file (${num_samples} samples)"
        else
            echo -e "${GREEN}✓${NC} $file"
        fi
    else
        echo -e "${RED}✗${NC} $file NOT found"
        ALL_GOOD=false
    fi
done

echo ""

if [ "$ALL_GOOD" = true ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✅ Dataset initialization complete!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "You can now start training with:"
    echo -e "${BLUE}  python scripts/run_finetune.py --train_jsonl dataset/gemma_train.jsonl --output_dir outputs/gemma3n-qved${NC}"
    echo ""
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}⚠️  Dataset initialization incomplete${NC}"
    echo -e "${RED}========================================${NC}"
    echo ""
    echo "Please check the errors above and retry."
    exit 1
fi
