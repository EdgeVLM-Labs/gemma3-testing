#!/bin/bash
# Setup Script for Gemma 3n Testing

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}🚀 Gemma 3n Setup Script${NC}"
echo "================================"
echo -e "${BLUE}Project root: ${PROJECT_ROOT}${NC}"
echo ""

# Check for Conda
if ! command -v conda &> /dev/null; then
    echo -e "${YELLOW}📦 Conda not found. Installing Miniconda...${NC}"
    cd /tmp
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    conda init bash
    source $HOME/miniconda/etc/profile.d/conda.sh
    echo -e "${GREEN}✅ Miniconda installed${NC}"
else
    echo -e "${GREEN}✅ Conda found${NC}"
    source $HOME/miniconda/etc/profile.d/conda.sh 2>/dev/null || source $(conda info --base)/etc/profile.d/conda.sh
fi

echo -e "${BLUE}📜 Accepting Conda Terms of Service...${NC}"
conda config --set allow_conda_downgrades true 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# Create Conda Environment
ENV_NAME="gemma3"

if conda env list | grep -q "^${ENV_NAME} "; then
    echo -e "${YELLOW}⚠️  Environment '${ENV_NAME}' already exists${NC}"
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n ${ENV_NAME} -y
        echo -e "${GREEN}✅ Removed existing environment${NC}"
    else
        echo -e "${YELLOW}Using existing environment${NC}"
        conda activate ${ENV_NAME}
        cd "$PROJECT_ROOT"
        pip install --upgrade pip
        if [ -f "requirements.txt" ]; then
            pip install -r requirements.txt
            echo -e "${GREEN}✅ Updated dependencies${NC}"
        fi
        exit 0
    fi
fi

echo -e "${BLUE}🔧 Creating conda environment: ${ENV_NAME}${NC}"
conda create --name=${ENV_NAME} python=3.11 -y
conda activate ${ENV_NAME}

# Install PyTorch with CUDA Support
echo -e "${BLUE}🔥 Installing PyTorch with CUDA 12.1...${NC}"
pip install --upgrade pip

if command -v nvcc &> /dev/null; then
    echo -e "${GREEN}✅ CUDA toolkit found${NC}"
    nvcc --version
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo -e "${YELLOW}⚠️  CUDA toolkit not found. Installing CPU-only PyTorch${NC}"
    pip install torch torchvision torchaudio
fi

# Install Project Dependencies
echo -e "${BLUE}📦 Installing project dependencies...${NC}"
cd "$PROJECT_ROOT"

if [ ! -f "requirements.txt" ]; then
    echo -e "${YELLOW}⚠️  requirements.txt not found in ${PROJECT_ROOT}${NC}"
    echo -e "${YELLOW}Skipping dependency installation${NC}"
else
    pip install -r requirements.txt
fi

echo -e "${YELLOW}📹 Installing optional Decord library...${NC}"
pip install decord || echo -e "${YELLOW}⚠️  Decord installation failed (optional)${NC}"

# Install FlashAttention (Optional)
read -p "Do you want to install FlashAttention for faster training? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${BLUE}⚡ Installing FlashAttention...${NC}"
    pip install flash-attn --no-build-isolation || echo -e "${YELLOW}⚠️  FlashAttention installation failed (optional)${NC}"
fi

# Install LaTeX for Plotting (Optional)
if command -v apt-get &> /dev/null; then
    read -p "Do you want to install LaTeX for publication-quality plots? (requires sudo/root) (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}📊 Installing LaTeX packages...${NC}"

        # Check if running as root or if sudo is available
        if [ "$EUID" -eq 0 ]; then
            # Running as root, use apt-get directly
            apt-get update
            apt-get install -y texlive texlive-latex-extra texlive-fonts-recommended dvipng cm-super
        elif command -v sudo &> /dev/null; then
            # sudo is available
            sudo apt-get update
            sudo apt-get install -y texlive texlive-latex-extra texlive-fonts-recommended dvipng cm-super
        else
            echo -e "${YELLOW}⚠️  Neither root privileges nor sudo available. Skipping LaTeX installation.${NC}"
            echo -e "${YELLOW}   You can install manually later with: apt-get install texlive texlive-latex-extra${NC}"
        fi
    fi
fi

# Verify Installation
echo ""
echo -e "${BLUE}🔍 Verifying Installation...${NC}"
echo "================================"

python -c "
import torch
import transformers
import peft
import trl
import bitsandbytes

print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ Transformers: {transformers.__version__}')
print(f'✅ PEFT: {peft.__version__}')
print(f'✅ TRL: {trl.__version__}')
print(f'✅ BitsAndBytes: {bitsandbytes.__version__}')
print(f'')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

# Check optional packages
echo ""
python -c "
import sys

try:
    import decord
    print(f'✅ Decord: {decord.__version__}')
except ImportError:
    print('⚠️  Decord: Not installed (optional)')

try:
    import flash_attn
    print(f'✅ Flash Attention: {flash_attn.__version__}')
except ImportError:
    print('⚠️  Flash Attention: Not installed (optional)')
"

# Authentication Setup
echo ""
echo -e "${BLUE}🔑 Authentication Setup${NC}"
echo "================================"

# HuggingFace
read -p "Do you want to set up HuggingFace authentication now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if command -v huggingface-cli &> /dev/null; then
        echo -e "${YELLOW}🤗 Logging into HuggingFace Hub...${NC}"
        huggingface-cli login
    else
        echo -e "${YELLOW}⚠️  huggingface-cli not found. Please run: pip install huggingface-hub${NC}"
    fi
fi

# WandB
read -p "Do you want to set up WandB authentication now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if command -v wandb &> /dev/null; then
        echo -e "${YELLOW}📊 Logging into WandB...${NC}"
        wandb login
    else
        echo -e "${YELLOW}⚠️  wandb not found. Please run: pip install wandb${NC}"
    fi
fi

# Dataset Setup
echo ""
read -p "Do you want to initialize the QVED dataset now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${BLUE}📥 Initializing dataset...${NC}"
    cd "$PROJECT_ROOT"
    ./scripts/initialize_dataset.sh
fi

# Make Scripts Executable
echo -e "${BLUE}🔧 Making scripts executable...${NC}"
cd "$PROJECT_ROOT"
chmod +x scripts/*.sh

# Setup Complete
echo ""
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo -e "${BLUE}📝 Next Steps:${NC}"
echo ""
echo "1. Activate the environment:"
echo -e "   ${YELLOW}conda activate ${ENV_NAME}${NC}"
echo ""
echo "2. Run inference:"
echo -e "   ${YELLOW}python scripts/run_inference.py --mode text --prompt \"Hello, Gemma!\"${NC}"
echo ""
echo "3. Initialize dataset (if not done):"
echo -e "   ${YELLOW}./scripts/initialize_dataset.sh${NC}"
echo ""
echo "4. Start fine-tuning:"
echo -e "   ${YELLOW}python scripts/run_finetune.py --train_jsonl dataset/gemma_train.jsonl --output_dir outputs/gemma3n-finetuned${NC}"
echo ""
echo "5. Verify setup:"
echo -e "   ${YELLOW}./scripts/verify_setup.sh${NC}"
echo ""
echo -e "${BLUE}📚 Documentation:${NC} See README.md for detailed usage"
echo ""

source ~/.bashrc
