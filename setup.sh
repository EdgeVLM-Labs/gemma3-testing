#!/bin/bash
# ==========================================
# Setup Script for Gemma-3N Fine-tuning with Unsloth
# ==========================================

set -e  # Exit on error

echo "🔧 Setting up Gemma-3N fine-tuning environment..."
echo ""

# ----------------------------
# Miniconda installation
# ----------------------------
if ! command -v conda &> /dev/null; then
    echo "📦 Installing Miniconda..."
    ORIGINAL_DIR=$(pwd)
    cd /tmp
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    
    # Initialize conda
    export PATH="$HOME/miniconda/bin:$PATH"
    eval "$($HOME/miniconda/bin/conda shell.bash hook)"
    
    conda init bash
    
    echo ""
    echo "✅ Miniconda installed successfully!"
    echo ""
    echo "⚠️  IMPORTANT: You must restart your terminal or run:"
    echo "    source ~/.bashrc"
    echo ""
    echo "Then re-run this script to continue setup:"
    echo "    bash setup.sh"
    echo ""
    cd "$ORIGINAL_DIR"
    exit 0
else
    echo "✅ Conda already installed"
    # Make sure conda is initialized in current shell
    eval "$(conda shell.bash hook)" 2>/dev/null || true
fi

# ----------------------------
# Create and activate environment
# ----------------------------
echo ""
echo "📦 Creating Conda environment 'gemma3n' with Python 3.11..."
conda create --name gemma3n python=3.11 -y

echo ""
echo "🔄 Activating environment..."
eval "$(conda shell.bash hook)"
conda activate gemma3n

# Verify activation
if [ "$CONDA_DEFAULT_ENV" != "gemma3n" ]; then
    echo "❌ Failed to activate conda environment"
    echo "Please run manually:"
    echo "  conda activate gemma3n"
    echo "  bash setup.sh"
    exit 1
fi

echo "✅ Environment 'gemma3n' activated"
echo ""

# Upgrade pip first
echo "📦 Upgrading pip..."
pip install --upgrade pip

# ----------------------------
# Check if Colab or local environment
# ----------------------------
echo "🔍 Detecting environment..."
if [ -n "$COLAB_GPU" ]; then
    echo "📍 Running in Google Colab"
    
    # Colab-specific installation
    # Install torch with correct version detection
    python -c "
import os, re, torch
v = re.match(r'[0-9]{1,}\.[0-9]{1,}', str(torch.__version__)).group(0)
xformers = 'xformers==' + ('0.0.33.post1' if v=='2.9' else '0.0.32.post2' if v=='2.8' else '0.0.29.post3')
print(f'Installing {xformers}')
os.system(f'pip install --no-deps bitsandbytes accelerate {xformers} peft trl triton cut_cross_entropy unsloth_zoo')
os.system('pip install sentencepiece protobuf \"datasets==4.3.0\" \"huggingface_hub>=0.34.0\" hf_transfer')
os.system('pip install --no-deps unsloth')
"
    
    pip install transformers==4.56.2
    pip install --no-deps trl==0.22.2
    pip install -U typing_extensions
    pip install --no-deps --upgrade timm
    
else
    echo "📍 Running in local environment"
    
    # ----------------------------
    # Base Python packages
    # ----------------------------
    echo "🧱 Installing base packages from requirements.txt..."
    pip install -r requirements.txt
    
fi

# ----------------------------
# Install additional dependencies
# ----------------------------
echo "📦 Installing additional dependencies..."
pip install opencv-python matplotlib
pip install wandb

# Install evaluation packages
echo "📊 Installing evaluation packages..."
pip install nltk rouge-score
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')" 2>/dev/null || true

# ----------------------------
# Ensure unsloth and timm are up to date
# ----------------------------
echo "🦥 Ensuring unsloth is properly installed..."
pip install --no-deps --upgrade timm  # Required for Gemma 3N
pip install --upgrade unsloth unsloth_zoo

# ----------------------------
# Environment verification
# ----------------------------
echo ""
echo "=== CUDA Check ==="
nvcc --version 2>/dev/null || echo "⚠️ nvcc not found"
nvidia-smi 2>/dev/null || echo "⚠️ nvidia-smi not found"

echo ""
echo "=== PyTorch CUDA Check ==="
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
else:
    print('⚠️ PyTorch cannot see CUDA')
"

echo ""
echo "=== Unsloth Check ==="
python -c "
try:
    from unsloth import FastVisionModel
    print('✅ Unsloth FastVisionModel available')
except ImportError as e:
    print(f'❌ Unsloth import failed: {e}')
"

# ----------------------------
# WandB & HuggingFace login
# ----------------------------
echo ""
echo "🔑 Authentication setup..."
read -p "Do you want to login to WandB? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    wandb login
fi

read -p "Do you want to login to HuggingFace? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    huggingface-cli login
fi

echo ""
echo "✅ Setup complete!"
echo "="*80
echo "🚀 Gemma-3N fine-tuning environment is ready!"
echo "="*80
echo ""
echo "📋 Current environment: $CONDA_DEFAULT_ENV"
echo ""
echo "Next steps:"
echo "1. The environment is already activated in this session"
echo ""
echo "2. For future sessions, activate with:"
echo "   conda activate gemma3n"
echo ""
echo "3. Start fine-tuning:"
echo "   bash scripts/finetune_gemma3n_unsloth.sh"
echo ""
echo "4. Or run manual training:"
echo "   python gemma3_finetune_unsloth.py --help"
echo ""
echo "="*80
