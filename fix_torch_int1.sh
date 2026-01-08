#!/bin/bash
# ==========================================
# Fix torch.int1 AttributeError
# ==========================================

echo "🔧 Fixing torch.int1 compatibility issue..."
echo ""

# Uninstall current versions
echo "📦 Uninstalling current PyTorch and TorchAO..."
pip uninstall -y torch torchvision torchaudio torchao

# Clear pip cache to ensure fresh downloads
echo "🧹 Clearing pip cache..."
pip cache purge

# Install PyTorch nightly (has torch.int1 support) - get absolute latest
echo "📦 Installing latest PyTorch nightly with CUDA 12.1..."
pip install --force-reinstall --no-cache-dir --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

# Install TorchAO nightly
echo "⚡ Installing TorchAO nightly..."
pip install --force-reinstall --no-cache-dir --pre torchao --index-url https://download.pytorch.org/whl/nightly/cu121

# Reinstall unsloth to ensure compatibility
echo "🦥 Reinstalling Unsloth..."
pip install --force-reinstall --no-cache-dir unsloth unsloth_zoo

# Reinstall mamba-ssm (needs to be recompiled against new PyTorch)
echo "🐍 Rebuilding Mamba-SSM for new PyTorch version..."
pip uninstall -y mamba-ssm
pip cache purge
pip install mamba-ssm --no-cache-dir --no-build-isolation

# Verify installation
echo ""
echo "🧪 Verifying installation..."
python -c "import torch; print(f'✅ PyTorch {torch.__version__}')" 2>/dev/null || echo "❌ PyTorch import failed"
python -c "import torchvision; print(f'✅ TorchVision {torchvision.__version__}')" 2>/dev/null || echo "❌ TorchVision import failed"
python -c "import torch; print(f'✅ torch.int1 available: {hasattr(torch, \"int1\")}')" 2>/dev/null || echo "❌ torch.int1 check failed"
python -c "import torchao; print(f'✅ TorchAO imported successfully')" 2>/dev/null || echo "❌ TorchAO import failed"
python -c "import mamba_ssm; print('✅ Mamba-SSM imported successfully')" 2>/dev/null || echo "❌ Mamba-SSM import failed"
python -c "from unsloth import FastModel; print('✅ Unsloth import successful')" 2>/dev/null || echo "❌ Unsloth import failed (this may take a moment)"

echo ""
echo "✅ Fix complete! Try running your fine-tuning script again."
