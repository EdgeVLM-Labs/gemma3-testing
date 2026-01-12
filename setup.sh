#!/bin/bash
# ==========================================
# Setup Script for Gemma-3N Fine-tuning with Unsloth
# ==========================================

# Don't exit on error initially - we'll handle errors manually
set +e

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
    
    # Accept conda TOS immediately after installation
    echo "📝 Accepting conda Terms of Service..."
    conda config --set tos_accepted yes 2>/dev/null || true
    
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
    
    # Accept conda TOS if not already accepted
    echo "📝 Accepting conda Terms of Service..."
    conda config --set tos_accepted yes 2>/dev/null || true
    
    # Accept specific channels TOS
    conda config --set anaconda_anon_usage false 2>/dev/null || true
fi

# ----------------------------
# Create and activate environment
# ----------------------------
echo ""
echo "📦 Checking for existing 'gemma3n' environment..."

# Check if environment exists
if conda env list | grep -q "^gemma3n "; then
    echo "✅ Environment 'gemma3n' already exists"
    echo "🔄 Activating existing environment..."
else
    echo "📦 Creating Conda environment 'gemma3n' with Python 3.11..."
    
    # Try creating with retry logic
    MAX_RETRIES=3
    RETRY_COUNT=0
    
    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        conda create --name gemma3n python=3.11 -y 2>&1
        
        if [ $? -eq 0 ]; then
            echo "✅ Environment created successfully"
            break
        else
            RETRY_COUNT=$((RETRY_COUNT + 1))
            if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
                echo "⚠️ Creation failed, retrying ($RETRY_COUNT/$MAX_RETRIES)..."
                sleep 2
            else
                echo "❌ Failed to create environment after $MAX_RETRIES attempts"
                echo ""
                echo "Please try manually:"
                echo "  conda config --set tos_accepted yes"
                echo "  conda create --name gemma3n python=3.11 -y"
                exit 1
            fi
        fi
    done
fi

echo ""
echo "🔄 Activating environment..."
eval "$(conda shell.bash hook)"
conda activate gemma3n

# Verify activation
if [ "$CONDA_DEFAULT_ENV" != "gemma3n" ]; then
    echo "⚠️ Environment not activated in script, but continuing..."
    echo "   (This is normal in bash scripts)"
fi

echo "✅ Environment setup ready"
echo ""

# Upgrade pip first
echo "📦 Upgrading pip..."
conda run -n gemma3n pip install --upgrade pip --quiet

# ----------------------------
# Install from requirements.txt
# ----------------------------
echo ""
echo "📦 Installing packages from requirements.txt..."
echo "   This may take several minutes..."

# Use conda run to ensure we're in the right environment
if [ -f "requirements.txt" ]; then
    conda run -n gemma3n pip install -r requirements.txt --quiet
    
    if [ $? -eq 0 ]; then
        echo "✅ Requirements installed successfully"
    else
        echo "⚠️ Some packages may have failed to install"
        echo "   Continuing with installation..."
    fi
else
    echo "❌ requirements.txt not found!"
    exit 1
fi

# ----------------------------
# Install additional critical packages
# ----------------------------
echo ""
echo "📦 Installing additional dependencies..."
conda run -n gemma3n pip install opencv-python matplotlib wandb --quiet

# Install evaluation packages
echo "📊 Installing evaluation packages..."
conda run -n gemma3n pip install nltk rouge-score sacrebleu openpyxl sentence-transformers --quiet
conda run -n gemma3n python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)" 2>/dev/null || true

# ----------------------------
# Ensure unsloth and timm are up to date
# ----------------------------
echo "🦥 Ensuring unsloth is properly installed..."
conda run -n gemma3n pip install --no-deps --upgrade timm --quiet
conda run -n gemma3n pip install --upgrade unsloth unsloth_zoo --quiet

# Ensure correct versions
echo "🔧 Installing specific package versions..."
conda run -n gemma3n pip install transformers==4.56.2 --quiet
conda run -n gemma3n pip install --no-deps trl==0.22.2 --quiet

# ----------------------------
# Environment verification
# ----------------------------
echo ""
echo "=========================================="
echo "🔍 Verifying Installation"
echo "=========================================="

echo ""
echo "=== CUDA Check ==="
nvcc --version 2>/dev/null || echo "⚠️ nvcc not found (optional)"
nvidia-smi 2>/dev/null || echo "⚠️ nvidia-smi not found"

echo ""
echo "=== PyTorch CUDA Check ==="
conda run -n gemma3n python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
else:
    print('⚠️ PyTorch cannot see CUDA')
" 2>/dev/null || echo "❌ PyTorch check failed"

echo ""
echo "=== Unsloth Check ==="
conda run -n gemma3n python -c "
try:
    from unsloth import FastVisionModel
    print('✅ Unsloth FastVisionModel available')
except ImportError as e:
    print(f'❌ Unsloth import failed: {e}')
" 2>/dev/null || echo "❌ Unsloth check failed"

echo ""
echo "=== Transformers Version ==="
conda run -n gemma3n python -c "
import transformers
print(f'Transformers: {transformers.__version__}')
expected = '4.56.2'
if transformers.__version__ == expected:
    print(f'✅ Correct version ({expected})')
else:
    print(f'⚠️ Expected {expected}, got {transformers.__version__}')
" 2>/dev/null || echo "❌ Transformers check failed"

# ----------------------------
# WandB & HuggingFace login
# ----------------------------
echo ""
echo "=========================================="
echo "🔑 Authentication Setup"
echo "=========================================="
echo ""
echo "You can login to WandB and HuggingFace now, or skip and do it later."
echo ""

read -p "Do you want to login to WandB? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    conda run -n gemma3n wandb login
fi

read -p "Do you want to login to HuggingFace? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    conda run -n gemma3n huggingface-cli login
fi

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "📋 Environment: gemma3n"
echo ""
echo "🔄 To activate the environment in a new terminal:"
echo "   conda activate gemma3n"
echo ""
echo "📦 Installed packages:"
echo "   - PyTorch with CUDA support"
echo "   - Unsloth FastVisionModel"
echo "   - Transformers 4.56.2"
echo "   - TRL 0.22.2"
echo "   - All requirements from requirements.txt"
echo ""
echo "🚀 Next steps:"
echo ""
echo "1. Activate the environment (required for each new terminal):"
echo "   conda activate gemma3n"
echo ""
echo "2. Prepare dataset:"
echo "   python dataset.py download --max-per-class 5"
echo "   python dataset.py prepare"
echo ""
echo "3. Start fine-tuning:"
echo "   bash scripts/finetune_gemma3n_unsloth.sh"
echo ""
echo "4. Or get help:"
echo "   python gemma3_finetune_unsloth.py --help"
echo ""
echo "=========================================="
echo ""
echo "⚠️  IMPORTANT: After closing this terminal, always activate:"
echo "   conda activate gemma3n"
echo ""
echo "=========================================="
