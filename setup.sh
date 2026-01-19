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
# Install PyTorch first (required by mamba-ssm)
# ----------------------------
echo "🔥 Installing PyTorch with CUDA 12.1 (required for building mamba-ssm)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet

# ----------------------------
# Install requirements
# ----------------------------
if [ ! -f requirements.txt ]; then
    echo "❌ requirements.txt not found"
    exit 1
fi

echo "📦 Installing requirements..."
pip install -r requirements.txt --quiet || true

# ----------------------------
# Fix mamba-ssm installation (requires torch during build)
# ----------------------------
echo "🐍 Installing mamba-ssm with proper build flags..."
pip uninstall -y mamba-ssm --quiet 2>/dev/null || true
pip cache purge --quiet 2>/dev/null || true
pip install mamba-ssm --no-cache-dir --no-build-isolation --quiet || {
    echo "⚠️  mamba-ssm installation failed. Run 'bash fix_torch_int1.sh' to fix."
}

# ----------------------------
# Core dependencies
# ----------------------------
echo "📦 Installing core dependencies..."
pip install \
    opencv-python \
    matplotlib \
    wandb \
    nltk \
    rouge-score \
    sacrebleu \
    openpyxl \
    sentence-transformers \
    --quiet

python - <<EOF
import nltk
nltk.download("punkt", quiet=True)
EOF

# ----------------------------
# Unsloth stack
# ----------------------------
echo "🦥 Installing Unsloth stack..."
pip install --upgrade unsloth unsloth_zoo timm --quiet
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
print("Transformers:", transformers.__version__)
try:
    from unsloth import FastVisionModel
    print("✅ Unsloth OK")
except Exception as e:
    print("❌ Unsloth error:", e)
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
echo "Start fine-tuning:"
echo "  bash scripts/finetune_gemma3n_unsloth.sh"
echo ""
