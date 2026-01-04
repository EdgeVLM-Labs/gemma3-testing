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

echo "📦 Creating Conda environment..."
conda create --name=gemma3n python=3.11 -y
conda activate gemma3n

pip install --upgrade pip

# ----------------------------
# Base Python packages
# ----------------------------
echo "🧱 Installing base packages..."
pip install torch==2.1.2 torchvision==0.16.2 torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.55.4
pip install timm
pip install sentencepiece protobuf datasets>=3.4.1,<4.0.0 huggingface_hub>=0.34.0 hf_transfer
pip install unsloth unsloth_zoo
pip install opencv-python opencv-contrib-python Pillow
pip install gradio gradio_client requests httpx uvicorn fastapi
pip install einops einops-exts loguru tenacity numpy<2.0 wandb openai==1.54.0

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
echo "🔑 Logging into WandB..."
wandb login

echo "🤗 Logging into HuggingFace Hub..."
hf auth login

echo "✅ Setup complete!"
echo "🚀 Gemma-3N E2B environment is ready."
source ~/.bashrc
