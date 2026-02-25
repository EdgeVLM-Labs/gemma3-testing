#!/bin/bash
# ==========================================
# Setup Script for Gemma-3N Fine-tuning (RunPod-safe)
# ==========================================

set +e

echo "🔧 Setting up Gemma-3N fine-tuning environment..."
echo ""

# ----------------------------
# System dependencies (from RUNPOD_QUICKSTART.md)
# ----------------------------
echo "📦 Installing system dependencies..."
if command -v apt-get &> /dev/null; then
    apt-get update -qq 2>/dev/null || true
    apt-get install -y wget git build-essential -qq 2>/dev/null || echo "⚠️  Some system packages may need manual installation"
else
    echo "⚠️  apt-get not found, skipping system dependencies"
fi

# ----------------------------
# Conda bootstrap
# ----------------------------
CONDA_INSTALLED=false
if ! command -v conda &> /dev/null; then
    echo "📦 Installing Miniconda..."
    cd /tmp
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p /root/miniconda
    export PATH="/root/miniconda/bin:$PATH"
    eval "$(/root/miniconda/bin/conda shell.bash hook)"
    conda init bash
    CONDA_INSTALLED=true
    echo "✅ Miniconda installed"
else
    echo "✅ Conda already installed"
    eval "$(conda shell.bash hook)"
fi

# ----------------------------
# Accept conda Terms of Service (if required)
# ----------------------------
echo "📜 Accepting conda Terms of Service..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# ----------------------------
# FORCE conda-forge only (CRITICAL FIX)
# ----------------------------
echo "🔒 Forcing conda-forge only (avoiding Anaconda ToS)..."

conda config --remove channels defaults 2>/dev/null || true
conda config --add channels conda-forge
conda config --set channel_priority strict

echo "✅ Channel configuration:"
conda config --show channels

# ----------------------------
# Create environment
# ----------------------------
echo ""
echo "📦 Creating Conda environment 'gemma3n'..."

if conda env list | grep -q "^gemma3n "; then
    echo "✅ Environment already exists"
else
    conda create \
        -n gemma3n \
        python=3.11 \
        -c conda-forge \
        --override-channels \
        -y || {
            echo "❌ Failed to create environment"
            exit 1
        }
fi

# ----------------------------
# Activate environment
# ----------------------------
echo ""
echo "🔄 Activating environment..."
conda activate gemma3n

echo "✅ Active env: $CONDA_DEFAULT_ENV"

# ----------------------------
# Upgrade pip
# ----------------------------
echo "📦 Upgrading pip..."
python -m pip install --upgrade pip --quiet

# ----------------------------
# Install PyTorch stack first (all components together)
# ----------------------------
echo "🔥 Installing PyTorch stack with CUDA 12.1..."
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121 --quiet
echo "⚡ Installing compatible xformers..."
pip install xformers==0.0.28.post3 --quiet

# ----------------------------
# Determine script directory
# ----------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# ----------------------------
# Install requirements (excluding mamba-ssm and xformers)
# ----------------------------
if [ ! -f requirements.txt ]; then
    echo "❌ requirements.txt not found in $SCRIPT_DIR"
    echo "Current directory: $(pwd)"
    echo "Listing files:"
    ls -la
    exit 1
fi

echo "📦 Installing requirements (excluding mamba-ssm and xformers)..."
# Install everything except mamba-ssm and xformers (already installed)
grep -v -e "mamba-ssm" -e "xformers" requirements.txt > /tmp/requirements_temp.txt || true
pip install -r /tmp/requirements_temp.txt --quiet || {
    echo "⚠️  Some packages failed to install, continuing..."
}
rm -f /tmp/requirements_temp.txt

# ----------------------------
# Skip mamba-ssm installation (optional and causes build issues)
# ----------------------------
echo "⚠️  Skipping mamba-ssm (optional - causes build issues on some systems)"
echo "    If you need VideoMamba support, install manually after setup completes"

# ----------------------------
# Core dependencies
# ----------------------------
echo "📦 Installing core dependencies..."
pip install \
    opencv-python \
    wandb \
    nltk \
    rouge-score \
    --quiet

python - <<EOF
import nltk
nltk.download("punkt", quiet=True)
EOF

# ----------------------------
# Unsloth stack (ensure compatibility with PyTorch 2.5.1)
# ----------------------------
echo "🦥 Installing Unsloth stack..."
pip uninstall -y unsloth unsloth_zoo peft --quiet 2>/dev/null || true
pip install --upgrade unsloth unsloth_zoo timm --quiet
pip install --upgrade packaging ninja einops peft accelerate bitsandbytes --quiet
pip install transformers==4.56.2 --quiet
pip install --no-deps trl==0.22.2 --quiet

# ----------------------------
# Verification
# ----------------------------
echo ""
echo "=========================================="
echo "🔍 Verification"
echo "=========================================="

python - <<EOF
import torch, transformers
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
print("Transformers:", transformers.__version__)

# Check xformers
try:
    import xformers
    print("✅ xformers:", xformers.__version__)
except Exception as e:
    print("⚠️  xformers not available")

# Check mamba-ssm (optional)
try:
    import mamba_ssm
    print("✅ Mamba-SSM OK (optional)")
except Exception as e:
    print("⚠️  Mamba-SSM not available (optional - not needed for basic usage)")

# Check Unsloth
try:
    from unsloth import FastVisionModel
    print("✅ Unsloth OK")
except Exception as e:
    print("❌ Unsloth error:", e)
    print("   Run: bash fix_unsloth.sh")
EOF

# ----------------------------
# Final message
# ----------------------------
echo ""
echo "=========================================="
echo "✅ SETUP COMPLETE"
echo "=========================================="
echo ""
echo "Activate with:"
echo "  conda activate gemma3n"
echo ""

# ----------------------------
# Service Authentication
# ----------------------------
echo "=========================================="
echo "🔑 Service Authentication Required"
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
    echo "📦 Ensuring HuggingFace CLI compatibility..."
    # Install with version constraint to match transformers 4.56.2
    pip install 'huggingface_hub>=0.34.0,<1.0' --quiet

    echo ""
    echo "🤗 HuggingFace Login"
    echo "Get your token from: https://huggingface.co/settings/tokens"
    echo ""
    huggingface-cli login

    echo ""
    echo "📊 Weights & Biases Login"
    echo "Get your API key from: https://wandb.ai/authorize"
    echo ""
    wandb login

    echo ""
    echo "✅ Authentication complete!"
else
    echo ""
    echo "⚠️  You can login later by running:"
    echo "    huggingface-cli login"
    echo "    wandb login"
fi

echo ""
echo "=========================================="
echo "🚀 Ready to Start"
echo "=========================================="
echo ""
echo "Start fine-tuning:"
echo "  bash scripts/finetune_gemma3n_unsloth.sh"
echo ""
