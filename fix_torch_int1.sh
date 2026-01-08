#!/bin/bash
# ==========================================
# Fix torch.int1 AttributeError
# ==========================================

echo "🔧 Fixing torch.int1 compatibility issue..."
echo ""

# Uninstall current versions
echo "📦 Uninstalling current PyTorch and TorchAO..."
pip uninstall -y torch torchvision torchaudio torchao

# Install PyTorch nightly (has torch.int1 support)
echo "📦 Installing PyTorch nightly with CUDA 12.1..."
pip install --upgrade --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

# Install TorchAO nightly
echo "⚡ Installing TorchAO nightly..."
pip install --upgrade --pre torchao --index-url https://download.pytorch.org/whl/nightly/cu121

# Verify installation
echo ""
echo "🧪 Verifying installation..."
python -c "import torch; print(f'✅ PyTorch {torch.__version__}')" 2>/dev/null || echo "❌ PyTorch import failed"
python -c "import torch; print(f'✅ torch.int1 available: {hasattr(torch, \"int1\")}')" 2>/dev/null || echo "❌ torch.int1 check failed"
python -c "import torchao; print(f'✅ TorchAO imported successfully')" 2>/dev/null || echo "❌ TorchAO import failed"

echo ""
echo "✅ Fix complete! Try running your fine-tuning script again."
