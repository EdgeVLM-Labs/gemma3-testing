#!/bin/bash
# ==========================================
# Setup Script for Google/Gemma-3N E2B
# ==========================================

echo "🔧 Creating workspace..."

# ----------------------------
# Miniconda installation
# ----------------------------
cd ..
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
source $HOME/miniconda/etc/profile.d/conda.sh

conda init bash
# source ~/.bashrc

echo "✅ Accepting Conda Terms of Service..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

echo "📦 Creating Conda environment..."
conda create --name=gemma3n python=3.11 -y
conda activate gemma3n

pip install --upgrade pip
pip install tqdm
# ----------------------------
# Base Python packages
# ----------------------------
echo "🧱 Installing base packages..."
pip install -r requirements.txt

# ----------------------------
# Ensure unsloth and peft compatibility
# ----------------------------
echo "🦥 Ensuring unsloth compatibility..."
pip install --upgrade unsloth>=2026.1.2 peft>=0.17.0,<=0.18.0

# ----------------------------
# Optional: Install specific versions for Float8 quantization (Unsloth LoRA fine-tuning)
# Uncomment these lines if you need Float8WeightOnlyConfig support:
# ----------------------------
# echo "⚡ Installing torch/torchao for Float8 quantization..."
# pip install --upgrade --force-reinstall \
#   "torch==2.6.0+cu121" \
#   "torchvision==0.21.0+cu121" \
#   "torchaudio==2.6.0+cu121" \
#   --index-url https://download.pytorch.org/whl/cu121
# pip install --upgrade --pre torchao==0.6.0.dev20241205 \
#   --index-url https://download.pytorch.org/whl/cu121
# pip install --no-deps --upgrade git+https://github.com/unslothai/unsloth.git
# python -c "from torchao.quantization import Float8WeightOnlyConfig; print('✅ Float8 quantization available')"


# ----------------------------
# Video augmentation: VidAug
# ----------------------------
echo "🎥 Installing VidAug..."
git clone https://github.com/okankop/vidaug
cd vidaug
python setup.py sdist && pip install dist/vidaug-0.1.tar.gz
cd ..
pip install git+https://github.com/okankop/vidaug

# ----------------------------
# FlashAttention for faster inference/training
# ----------------------------
echo "⚡ Installing FlashAttention..."
git clone https://github.com/HazyResearch/flash-attention.git
cd flash-attention
pip install ninja packaging wheel
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install flash-attn --no-build-isolation
python -c "import flash_attn; print(f'✅ Flash Attention version: {flash_attn.__version__}')"
cd ..

# ----------------------------
# Mamba-SSM for training (VideoMamba encoder)
# ----------------------------
echo "🐍 Installing Mamba-SSM..."
pip uninstall -y mamba-ssm
pip cache purge
pip install mamba-ssm --no-cache-dir --no-build-isolation
python -c "import mamba_ssm; print('✅ Mamba-SSM installed successfully')"

# ----------------------------
# Optional: Video libraries for frame handling
# ----------------------------
pip install imageio decord scikit-learn scikit-image albumentations

# ----------------------------
# LaTeX for reports (optional)
# ----------------------------
apt-get update
apt-get install -y texlive texlive-latex-extra texlive-fonts-recommended dvipng cm-super

# ----------------------------
# Environment verification
# ----------------------------
echo "=== CUDA Check ==="
nvcc --version 2>/dev/null || echo "❌ nvcc not found"
nvidia-smi 2>/dev/null || echo "❌ nvidia-smi not found"

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
    print('❌ PyTorch cannot see CUDA')
"

# ----------------------------
# WandB & HuggingFace login
# ----------------------------
pip install wandb
echo "🔑 Logging into WandB..."
wandb login

echo "🤗 Logging into HuggingFace Hub..."
hf auth login

echo "✅ Setup complete!"
echo "🚀 Gemma-3N E2B environment is ready."
source ~/.bashrc