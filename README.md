# Gemma-3N Fine-tuning for Video Understanding

Fine-tune Google's Gemma-3N model on video datasets using Unsloth for efficient LoRA training. Optimized for exercise form analysis and video understanding tasks.

---

## 🚀 Quick Start

```bash
# 1. Setup environment
git clone https://github.com/EdgeVLM-Labs/gemma3-testing.git
cd gemma3-testing
bash setup.sh
conda activate gemma3n

# 2. Login to services
wandb login
huggingface-cli login

# 3. Prepare dataset
python dataset.py download --max-per-class 5
python dataset.py prepare

# 4. Fine-tune model
bash scripts/finetune_gemma3n_unsloth.sh

# 5. Run inference
python scripts/run_inference_unsloth.py \
    --model_path outputs/gemma3n_finetune_YYYYMMDD_HHMMSS_merged_16bit \
    --video_path dataset/exercise_class_1/video_001.mp4
```

**📖 Complete Guide:** See [docs/FINETUNE_GUIDE.md](docs/FINETUNE_GUIDE.md) for detailed instructions.

---

## 📋 Prerequisites

- **OS:** Linux/macOS (Ubuntu 20.04+ recommended)
- **GPU:** NVIDIA GPU with 16GB+ VRAM (A100 recommended)
- **Storage:** 50GB+ free disk space
- **Software:** Git, CUDA 11.8+, Python 3.11

---

## 🔧 Installation

### 1. Clone Repository

```bash
git clone https://github.com/EdgeVLM-Labs/gemma3-testing.git
cd gemma3-testing
```

### 2. Run Setup Script

```bash
bash setup.sh
conda activate gemma3n
```

This installs:
- Miniconda (if not present)
- Python 3.11 environment
- PyTorch with CUDA support
- Unsloth and dependencies
- All required packages

### 3. Authenticate

```bash
# HuggingFace (required for models and datasets)
huggingface-cli login

# Weights & Biases (required for training tracking)
wandb login
```

### 4. Verify Installation

```bash
# Check CUDA
nvidia-smi

# Check PyTorch
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check Unsloth
python -c "from unsloth import FastVisionModel; print('Unsloth OK')"
```

---

## 📊 Dataset Management

The `dataset.py` script manages the QVED (Quality Video Exercise Dataset):

### Download Videos

```bash
# Download 5 videos per exercise class
python dataset.py download --max-per-class 5

# Download more videos
python dataset.py download --max-per-class 20
```

### Prepare Train/Val/Test Splits

```bash
python dataset.py prepare
```

Creates:
- `dataset/qved_train.json` (60%)
- `dataset/qved_val.json` (20%)
- `dataset/qved_test.json` (20%)

### Optional: Clean Dataset

```bash
# Filter low-quality videos based on resolution, brightness, sharpness
python dataset.py clean
```

### Copy Videos for Testing

```bash
# Copy videos to flat folder for batch inference
python dataset.py copy --num-videos 10 --output test_videos
```

### All-in-One

```bash
# Download, prepare, and clean in one command
python dataset.py all --max-per-class 10
```

---

## 🎯 Fine-tuning

### Method 1: Quick Start (Recommended)

```bash
# Uses local dataset prepared by dataset.py
bash scripts/finetune_gemma3n_unsloth.sh
```

Default configuration:
- Model: `unsloth/gemma-3n-E4B-it`
- LoRA: r=64, alpha=128
- Batch size: 1, Gradient accumulation: 4
- Learning rate: 2e-4
- Epochs: 3

### Method 2: Custom Configuration

```bash
python gemma3_finetune_unsloth.py \
    --train_json dataset/qved_train.json \
    --output_dir outputs/custom_finetune \
    --num_epochs 5 \
    --batch_size 1 \
    --gradient_accumulation 8 \
    --learning_rate 2e-4 \
    --lora_r 128 \
    --lora_alpha 256 \
    --num_frames 8
```

### Method 3: HuggingFace Streaming (Alternative)

```bash
# Download dataset during training
bash scripts/finetune_gemma3n_unsloth.sh --hf
```

### Monitor Training

View real-time metrics at https://wandb.ai in your project.

### Output Files

After training:
```
outputs/
├── gemma3n_finetune_YYYYMMDD_HHMMSS/
│   └── adapter_model.safetensors        # LoRA weights only
└── gemma3n_finetune_YYYYMMDD_HHMMSS_merged_16bit/
    └── model.safetensors                 # Full model (use this for inference)
```

---

## 🔮 Inference

### Single Video

```bash
python scripts/run_inference_unsloth.py \
    --model_path outputs/gemma3n_finetune_YYYYMMDD_HHMMSS_merged_16bit \
    --video_path test_videos/video_001.mp4 \
    --num_frames 8
```

### Batch Inference

```bash
# Prepare test videos
python dataset.py copy --num-videos 20 --output batch_test

# Run batch processing
python gemma3n_batch_inference.py \
    --model outputs/gemma3n_finetune_YYYYMMDD_HHMMSS_merged_16bit \
    --video_folder batch_test \
    --output results/batch_results.csv
```

### Custom Prompt

```bash
python scripts/run_inference_unsloth.py \
    --model_path outputs/gemma3n_finetune_YYYYMMDD_HHMMSS_merged_16bit \
    --video_path test.mp4 \
    --instruction "Analyze the exercise form and provide corrections" \
    --temperature 0.7
```

---

## ⚙️ Configuration

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch_size` | 1 | Per-device batch size |
| `--gradient_accumulation` | 4 | Gradient accumulation steps |
| `--learning_rate` | 2e-4 | Learning rate |
| `--num_epochs` | 3 | Training epochs |
| `--lora_r` | 64 | LoRA rank |
| `--lora_alpha` | 128 | LoRA alpha |
| `--num_frames` | 8 | Frames per video |
| `--max_seq_length` | 50000 | Maximum sequence length |

### Training Strategies

**Quick Test:**
```bash
python gemma3_finetune_unsloth.py --num_epochs 1 --lora_r 32
```

**Production:**
```bash
python gemma3_finetune_unsloth.py --num_epochs 5 --lora_r 128 --lora_alpha 256
```

**Memory Constrained (<24GB):**
```bash
python gemma3_finetune_unsloth.py --load_in_4bit --batch_size 1 --num_frames 4
```

---

## 🐛 Troubleshooting

### Dataset Issues

**Dataset not found:**
```bash
python dataset.py download --max-per-class 5
python dataset.py prepare
```

**Video file not found:**
```bash
# Check paths in JSON are relative to dataset/
head -n 20 dataset/qved_train.json
```

### Memory Issues

**CUDA out of memory:**
```bash
# Enable 4-bit quantization
python gemma3_finetune_unsloth.py --load_in_4bit ...

# Or reduce memory usage
python gemma3_finetune_unsloth.py \
    --batch_size 1 \
    --gradient_accumulation 8 \
    --num_frames 4
```

### Installation Issues

**Module not found:**
```bash
conda activate gemma3n
pip install -r requirements.txt --force-reinstall
```

**Unsloth import error:**
```bash
pip install --no-deps --upgrade timm
pip install --upgrade unsloth unsloth_zoo
pip install transformers==4.56.2
pip install trl==0.22.2
```

**Environment not found:**
```bash
# Re-run setup
bash setup.sh
conda activate gemma3n
```

---

## 📁 Repository Structure

```
gemma3-testing/
├── dataset.py                       # Dataset management (download, prepare, clean)
├── gemma3_finetune_unsloth.py      # Main training script
├── gemma3n_batch_inference.py      # Batch inference
├── setup.sh                         # Environment setup
├── requirements.txt                 # Dependencies
├── scripts/
│   ├── finetune_gemma3n_unsloth.sh # Training wrapper
│   └── run_inference_unsloth.py    # Inference script
├── docs/
│   ├── FINETUNE_GUIDE.md           # Complete guide
│   ├── SETUP_CHECKLIST.md          # Setup checklist
│   └── QUICKSTART.md               # Quick start
├── utils/                          # Utility scripts
├── gemma3/                         # Gemma-3N implementation
└── dataset/                        # Your prepared dataset
    ├── exercise_class_1/
    ├── qved_train.json
    ├── qved_val.json
    └── qved_test.json
```

---

## 🚀 Features

- ✅ **Efficient LoRA fine-tuning** with Unsloth FastVisionModel
- ✅ **Local dataset support** - uses your prepared data by default
- ✅ **Automatic frame extraction** from videos
- ✅ **4-bit quantization** for limited GPU memory
- ✅ **Gradient checkpointing** for memory optimization
- ✅ **WandB integration** for experiment tracking
- ✅ **Streaming support** for HuggingFace datasets
- ✅ **Dataset quality filtering** (resolution, brightness, sharpness)
- ✅ **Batch inference** for multiple videos

---

## 📚 Documentation

- **[Fine-tuning Guide](docs/FINETUNE_GUIDE.md)** - Complete training and inference guide
- [Setup Checklist](docs/SETUP_CHECKLIST.md) - Step-by-step setup with checkboxes
- [Quick Start](docs/QUICKSTART.md) - Original quick start guide
- [Gemma3 Implementation](gemma3/README.md) - Architecture details

---

## 🎓 Usage Examples

### Complete Workflow

```bash
# Setup (one time)
bash setup.sh
conda activate gemma3n
wandb login
huggingface-cli login

# Prepare data (one time)
python dataset.py download --max-per-class 10
python dataset.py prepare

# Train
bash scripts/finetune_gemma3n_unsloth.sh

# Test
python scripts/run_inference_unsloth.py \
    --model_path outputs/gemma3n_finetune_*/merged_16bit \
    --video_path dataset/exercise_class_1/video.mp4
```

### Experiment with Different Configurations

```bash
# Try different learning rates
for lr in 1e-4 2e-4 5e-4; do
    python gemma3_finetune_unsloth.py \
        --learning_rate $lr \
        --wandb_run_name "lr_${lr}" \
        --output_dir "outputs/exp_lr_${lr}"
done
```

---

## 📊 Expected Performance

| Configuration | GPU | Training Time (500 samples, 3 epochs) |
|--------------|-----|---------------------------------------|
| Default | A100 40GB | ~30-45 min |
| 4-bit | RTX 4090 | ~45-60 min |
| Large batch | A100 80GB | ~20-30 min |

---

## 🤝 Contributing

Contributions welcome! Please:
- Follow existing code style
- Update documentation for new features
- Test changes before submitting

---

## 📄 License

See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [Gemma](https://ai.google.dev/gemma) by Google
- [Unsloth](https://github.com/unslothai/unsloth) for efficient fine-tuning
- [QVED Dataset](https://huggingface.co/datasets/EdgeVLM-Labs/QVED-Test-Dataset)

---

## 📧 Support

- **Documentation:** [docs/FINETUNE_GUIDE.md](docs/FINETUNE_GUIDE.md)
- **Issues:** [GitHub Issues](https://github.com/EdgeVLM-Labs/gemma3-testing/issues)
- **Discussions:** [GitHub Discussions](https://github.com/EdgeVLM-Labs/gemma3-testing/discussions)

---

**Made with ❤️ by EdgeVLM Labs**
