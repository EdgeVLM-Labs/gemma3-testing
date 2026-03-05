#!/bin/bash
# ==========================================
# Setup Script for Gemma-3N Fine-tuning
# ==========================================

set +e

echo "Setting up Gemma-3N fine-tuning environment..."
echo ""

# ----------------------------
# System dependencies
# ----------------------------
echo "[1/6] Installing system dependencies..."
if command -v apt-get &> /dev/null; then
    apt-get update -qq 2>/dev/null || true
    apt-get install -y wget git build-essential -qq 2>/dev/null || echo "Some system packages may need manual installation"
else
    echo "apt-get not found, skipping system dependencies"
fi

# ----------------------------
# Conda bootstrap
# ----------------------------
echo ""
echo "[2/6] Setting up Conda..."
if ! command -v conda &> /dev/null; then
    echo "Installing Miniconda..."
    cd /tmp
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p /root/miniconda
    export PATH="/root/miniconda/bin:$PATH"
    eval "$(/root/miniconda/bin/conda shell.bash hook)"
    conda init bash
    echo "Miniconda installed"
else
    echo "Conda already installed"
    eval "$(conda shell.bash hook)"
fi

# Accept conda ToS
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# Use conda-forge only
conda config --remove channels defaults 2>/dev/null || true
conda config --add channels conda-forge
conda config --set channel_priority strict

# ----------------------------
# Create and activate environment
# ----------------------------
echo ""
echo "[3/6] Creating Conda environment 'gemma3n'..."

if conda env list | grep -q "^gemma3n "; then
    echo "Environment already exists"
else
    conda create \
        -n gemma3n \
        python=3.11 \
        -c conda-forge \
        --override-channels \
        -y || {
            echo "Failed to create environment"
            exit 1
        }
fi

echo "Activating environment..."
conda activate gemma3n
echo "Active env: $CONDA_DEFAULT_ENV"

# ----------------------------
# Upgrade pip
# ----------------------------
python -m pip install --upgrade pip --quiet

# ----------------------------
# Install PyTorch with CUDA 12.1
# ----------------------------
echo ""
echo "[4/6] Installing PyTorch with CUDA 12.1..."
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 --quiet

# ----------------------------
# Install requirements
# ----------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

if [ ! -f requirements.txt ]; then
    echo "requirements.txt not found in $SCRIPT_DIR"
    exit 1
fi

echo ""
echo "[5/6] Installing Python dependencies..."
# Skip torch/torchvision from requirements (already installed with CUDA index above)
grep -v -e "^torch==" -e "^torchvision==" requirements.txt > /tmp/req_no_torch.txt || true
pip install -r /tmp/req_no_torch.txt --quiet || {
    echo "Some packages failed to install, continuing..."
}
rm -f /tmp/req_no_torch.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt', quiet=True)" 2>/dev/null || true

# ----------------------------
# Verification
# ----------------------------
echo ""
echo "=========================================="
echo "[6/6] Verification"
echo "=========================================="

python - <<EOF
import torch, transformers
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
print("Transformers:", transformers.__version__)

from transformers import Gemma3nForConditionalGeneration, AutoProcessor
print("Gemma3n model: OK")
EOF

# ----------------------------
# Service Authentication
# ----------------------------
echo ""
echo "=========================================="
echo "Service Authentication"
echo "=========================================="
echo ""
echo "To use this project, you need to authenticate with:"
echo "  1. HuggingFace (for models and datasets)"
echo "  2. Weights & Biases (for training tracking)"
echo ""
read -p "Do you want to login now? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "HuggingFace Login"
    echo "Get your token from: https://huggingface.co/settings/tokens"
    echo ""
    huggingface-cli login

    echo ""
    echo "Weights & Biases Login"
    echo "Get your API key from: https://wandb.ai/authorize"
    echo ""
    wandb login

    echo ""
    echo "Authentication complete!"
else
    echo ""
    echo "You can login later by running:"
    echo "  huggingface-cli login"
    echo "  wandb login"
fi

echo ""
echo "=========================================="
echo "SETUP COMPLETE"
echo "=========================================="
echo ""
echo "Activate with:"
echo "  conda activate gemma3n"
echo ""
echo "Start fine-tuning:"
echo "  python core/finetune_gemma3n_e2b_trl.py --help"
echo ""
