#!/bin/bash

# Verification script for Gemma 3n QVED setup
# Checks dataset files, video files, and Python dependencies

echo "========================================="
echo "Gemma 3n QVED Setup Verification"
echo "========================================="

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Track overall status
ALL_CHECKS_PASSED=true

# Check dataset files
echo -e "\n${YELLOW}[1] Checking dataset files...${NC}"
if [ -f "dataset/gemma_train.jsonl" ]; then
    num_samples=$(wc -l < "dataset/gemma_train.jsonl" 2>/dev/null || echo "?")
    echo -e "${GREEN}✓${NC} dataset/gemma_train.jsonl found ($num_samples samples)"
else
    echo -e "${RED}✗${NC} dataset/gemma_train.jsonl NOT found"
    ALL_CHECKS_PASSED=false
fi

if [ -f "dataset/gemma_val.jsonl" ]; then
    num_samples=$(wc -l < "dataset/gemma_val.jsonl" 2>/dev/null || echo "?")
    echo -e "${GREEN}✓${NC} dataset/gemma_val.jsonl found ($num_samples samples)"
else
    echo -e "${RED}✗${NC} dataset/gemma_val.jsonl NOT found"
    ALL_CHECKS_PASSED=false
fi

if [ -f "dataset/gemma_test.jsonl" ]; then
    num_samples=$(wc -l < "dataset/gemma_test.jsonl" 2>/dev/null || echo "?")
    echo -e "${GREEN}✓${NC} dataset/gemma_test.jsonl found ($num_samples samples)"
else
    echo -e "${RED}✗${NC} dataset/gemma_test.jsonl NOT found"
    ALL_CHECKS_PASSED=false
fi

if [ -f "dataset/manifest.json" ]; then
    echo -e "${GREEN}✓${NC} dataset/manifest.json found"
else
    echo -e "${RED}✗${NC} dataset/manifest.json NOT found"
    ALL_CHECKS_PASSED=false
fi

if [ -f "dataset/ground_truth.json" ]; then
    echo -e "${GREEN}✓${NC} dataset/ground_truth.json found"
else
    echo -e "${RED}✗${NC} dataset/ground_truth.json NOT found"
    ALL_CHECKS_PASSED=false
fi

# Check video files
echo -e "\n${YELLOW}[2] Checking video files...${NC}"
if [ -d "dataset" ]; then
    # Count video files
    VIDEO_COUNT=$(find dataset -name "*.mp4" 2>/dev/null | wc -l)
    echo -e "${GREEN}✓${NC} Found $VIDEO_COUNT video files (.mp4)"

    # List exercise directories
    EXERCISE_DIRS=$(find dataset -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    echo -e "${GREEN}✓${NC} Found $EXERCISE_DIRS exercise directories"

    if [ "$VIDEO_COUNT" -eq 0 ]; then
        echo -e "${RED}✗${NC} No video files found in dataset/"
        ALL_CHECKS_PASSED=false
    fi
else
    echo -e "${RED}✗${NC} dataset/ directory NOT found"
    ALL_CHECKS_PASSED=false
fi

# Check Python dependencies
echo -e "\n${YELLOW}[3] Checking Python dependencies...${NC}"

check_python_package() {
    package=$1
    python -c "import $package" 2>/dev/null
    if [ $? -eq 0 ]; then
        version=$(python -c "import $package; print(getattr($package, '__version__', 'unknown'))" 2>/dev/null)
        echo -e "${GREEN}✓${NC} $package ($version)"
    else
        echo -e "${RED}✗${NC} $package NOT installed"
        ALL_CHECKS_PASSED=false
    fi
}

check_python_package "torch"
check_python_package "transformers"
check_python_package "trl"
check_python_package "peft"
check_python_package "datasets"
check_python_package "PIL"
check_python_package "cv2"

# Check optional dependencies
echo -e "\n${YELLOW}[4] Checking optional dependencies...${NC}"
python -c "import decord" 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} decord (optional)"
else
    echo -e "${YELLOW}!${NC} decord not installed (optional, but recommended for faster video processing)"
fi

python -c "import bitsandbytes" 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} bitsandbytes (optional)"
else
    echo -e "${YELLOW}!${NC} bitsandbytes not installed (optional, required for 4-bit quantization)"
fi

# Check directory structure
echo -e "\n${YELLOW}[5] Checking directory structure...${NC}"

REQUIRED_DIRS=(
    "utils"
    "scripts"
    "dataset"
)

for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo -e "${GREEN}✓${NC} $dir/ directory exists"
    else
        echo -e "${RED}✗${NC} $dir/ directory NOT found"
        ALL_CHECKS_PASSED=false
    fi
done

# Check Python modules
echo -e "\n${YELLOW}[6] Checking Python modules...${NC}"

REQUIRED_MODULES=(
    "utils/config.py"
    "utils/model_utils.py"
    "utils/inference.py"
    "utils/finetune.py"
    "utils/dataset_utils.py"
    "utils/video_utils.py"
    "utils/prompt_utils.py"
    "utils/logging_utils.py"
    "scripts/run_inference.py"
    "scripts/run_finetune.py"
)

for module in "${REQUIRED_MODULES[@]}"; do
    if [ -f "$module" ]; then
        echo -e "${GREEN}✓${NC} $module"
    else
        echo -e "${RED}✗${NC} $module NOT found"
        ALL_CHECKS_PASSED=false
    fi
done

# Check GPU availability
echo -e "\n${YELLOW}[7] Checking GPU availability...${NC}"
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count() if torch.cuda.is_available() else 0); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" 2>/dev/null

# Final summary
echo ""
echo "========================================="
if [ "$ALL_CHECKS_PASSED" = true ]; then
    echo -e "${GREEN}✅ All critical checks passed!${NC}"
    echo "========================================="
    echo ""
    echo "Your Gemma 3n setup is ready for training."
    echo ""
    echo "Next steps:"
    echo "  1. Initialize dataset (if not done):"
    echo "     bash scripts/initialize_dataset.sh"
    echo ""
    echo "  2. Start training:"
    echo "     python scripts/run_finetune.py --train_jsonl dataset/gemma_train.jsonl --output_dir outputs/gemma3n-qved"
    echo ""
else
    echo -e "${RED}⚠️  Some checks failed${NC}"
    echo "========================================="
    echo ""
    echo "Please fix the issues above before proceeding."
    echo ""
    echo "To install missing dependencies:"
    echo "  pip install -r requirements.txt"
    echo ""
    echo "To initialize dataset:"
    echo "  bash scripts/initialize_dataset.sh"
    echo ""
    exit 1
fi
