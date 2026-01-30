#!/bin/bash
# ==========================================
# Fix Unsloth and PEFT Compatibility Issue
# ==========================================
# Note: If you also have mamba-ssm issues, run fix_torch_int1.sh instead
# which fixes torch, unsloth, AND mamba-ssm together.

echo "🔧 Fixing unsloth and peft compatibility..."

# Uninstall problematic versions
pip uninstall -y unsloth unsloth_zoo peft

# Reinstall compatible versions
echo "📦 Installing compatible versions..."
pip install --upgrade unsloth unsloth_zoo
pip install --upgrade packaging ninja einops xformers trl peft accelerate bitsandbytes

# Verify installation
echo ""
echo "🧪 Verifying installation..."
python -c "from unsloth import FastModel; print('✅ Unsloth import successful')" 2>/dev/null && echo "✅ Unsloth working!" || echo "❌ Unsloth still has issues"

echo ""
echo "✅ Fix complete! Try running your script again."
