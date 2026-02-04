#!/bin/bash

# Quick setup script for Gemma 3n QVED training
# Combines all setup steps with error handling

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Gemma 3n QVED Quick Setup${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Step 1: Install dependencies
echo -e "${YELLOW}[1/4] Installing Python dependencies...${NC}"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${RED}✗ requirements.txt not found${NC}"
    exit 1
fi
echo ""

# Step 2: Initialize dataset
echo -e "${YELLOW}[2/4] Setting up dataset...${NC}"
echo -n "Download videos per exercise class [default: 5]: "
read -r VIDEO_COUNT

if [ -z "$VIDEO_COUNT" ]; then
    VIDEO_COUNT=5
fi

echo -e "${BLUE}Downloading ${VIDEO_COUNT} videos per class...${NC}"
python -m utils.load_dataset "$VIDEO_COUNT"
python -m utils.filter_ground_truth
python -m utils.convert_dataset

echo -e "${GREEN}✓ Dataset ready${NC}"
echo ""

# Step 3: Verify setup
echo -e "${YELLOW}[3/4] Verifying setup...${NC}"
bash scripts/verify_setup.sh

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Setup verification failed${NC}"
    exit 1
fi
echo ""

# Step 4: Display training instructions
echo -e "${YELLOW}[4/4] Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✅ Gemma 3n QVED setup complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "To start training:"
echo -e "${BLUE}  python scripts/run_finetune.py \\${NC}"
echo -e "${BLUE}    --train_jsonl dataset/gemma_train.jsonl \\${NC}"
echo -e "${BLUE}    --output_dir outputs/gemma3n-qved \\${NC}"
echo -e "${BLUE}    --method qlora \\${NC}"
echo -e "${BLUE}    --epochs 3${NC}"
echo ""
echo "To run inference:"
echo -e "${BLUE}  python scripts/run_inference.py \\${NC}"
echo -e "${BLUE}    --mode video \\${NC}"
echo -e "${BLUE}    --prompt \"Evaluate this exercise\" \\${NC}"
echo -e "${BLUE}    --video_path path/to/video.mp4${NC}"
echo ""
