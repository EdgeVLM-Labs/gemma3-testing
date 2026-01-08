#!/bin/bash
# ==========================================
# Complete Fine-tuning Environment Setup
# ==========================================
# This script installs all dependencies needed for Gemma-3N fine-tuning

set -e  # Exit on error

echo "=========================================="
echo "🚀 Setting up Fine-tuning Environment"
echo "=========================================="
echo ""

# ----------------------------
# 1. PyTorch with CUDA 12.1 (Nightly for torch.int1 support)
# ----------------------------
echo "📦 [1/7] Installing PyTorch nightly with CUDA 12.1..."
pip uninstall -y torch torchvision torchaudio
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
echo "✅ PyTorch nightly installed"
echo ""

# ----------------------------
# 2. TorchAO (with torch.int1 support)
# ----------------------------
echo "⚡ [2/7] Installing TorchAO nightly..."
pip install --pre torchao --index-url https://download.pytorch.org/whl/nightly/cu121
echo "✅ TorchAO installed"
echo ""

# ----------------------------
# 3. Core Training Libraries
# ----------------------------
echo "📚 [3/7] Installing core training libraries..."
pip install --upgrade transformers accelerate peft bitsandbytes
pip install deepspeed ninja packaging wheel
echo "✅ Core libraries installed"
echo ""

# ----------------------------
# 4. Unsloth for Fast Training
# ----------------------------
echo "🦥 [4/7] Installing Unsloth..."
pip uninstall -y unsloth unsloth_zoo
pip install --upgrade unsloth unsloth_zoo
echo "✅ Unsloth installed"
echo ""

# ----------------------------
# 5. Mamba-SSM (VideoMamba encoder)
# ----------------------------
echo "🐍 [5/7] Installing Mamba-SSM..."
pip uninstall -y mamba-ssm
pip cache purge
pip install mamba-ssm --no-cache-dir --no-build-isolation
echo "✅ Mamba-SSM installed"
echo ""

# ----------------------------
# 6. Video Processing Libraries
# ----------------------------
echo "🎥 [6/7] Installing video processing libraries..."
pip install opencv-python opencv-contrib-python opencv-python-headless
pip install decord imageio Pillow einops einops-exts
pip install albumentations scikit-image
echo "✅ Video libraries installed"
echo ""

# ----------------------------
# 7. Additional Dependencies
# ----------------------------
echo "📦 [7/7] Installing additional dependencies..."
pip install wandb datasets huggingface_hub tqdm
pip install timm mmengine
pip install evaluate nltk sacrebleu scikit-learn
echo "✅ Additional dependencies installed"
echo ""

# ----------------------------
# Verification
# ----------------------------
echo "=========================================="
echo "🧪 Verifying Installation"
echo "=========================================="
echo ""

# Check PyTorch
python -c "import torch; print(f'✅ PyTorch {torch.__version__} (CUDA: {torch.version.cuda})')" 2>/dev/null || echo "❌ PyTorch import failed"

# Check CUDA availability
python -c "import torch; print(f'✅ CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || echo "❌ CUDA check failed"

# Check Mamba-SSM
python -c "import mamba_ssm; print('✅ Mamba-SSM imported successfully')" 2>/dev/null || echo "❌ Mamba-SSM import failed"

# Check Unsloth
python -c "from unsloth import FastModel; print('✅ Unsloth imported successfully')" 2>/dev/null || echo "❌ Unsloth import failed"

# Check Transformers
python -c "import transformers; print(f'✅ Transformers {transformers.__version__}')" 2>/dev/null || echo "❌ Transformers import failed"

# Check PEFT
python -c "import peft; print(f'✅ PEFT {peft.__version__}')" 2>/dev/null || echo "❌ PEFT import failed"

# Check Accelerate
python -c "import accelerate; print(f'✅ Accelerate {accelerate.__version__}')" 2>/dev/null || echo "❌ Accelerate import failed"

# Check DeepSpeed
python -c "import deepspeed; print(f'✅ DeepSpeed {deepspeed.__version__}')" 2>/dev/null || echo "❌ DeepSpeed import failed"

echo ""
echo "=========================================="
echo "✅ Fine-tuning Environment Setup Complete!"
echo "=========================================="
echo ""
echo "You can now run fine-tuning with:"
echo "  bash scripts/finetune_qved.sh"
echo "  bash scripts/finetune_gemma3n_unsloth.sh"
echo ""
