# Gemma-3N-E2B Video Understanding & Fine-tuning

This repository provides tools for fine-tuning and inference with **Google Gemma-3N-E2B** models for video understanding tasks, with a focus on exercise form analysis and assistive navigation for visually impaired users.

---

## 🎯 Quick Start

**📋 New users: See [Setup Checklist](docs/SETUP_CHECKLIST.md) for a step-by-step guide with checkboxes!**

### Prerequisites
- Linux/macOS (Ubuntu 20.04+ recommended)
- NVIDIA GPU with CUDA support (16GB+ VRAM recommended)
- 50GB+ free disk space
- Root/sudo access for system packages
- Git and basic command line knowledge

### Initial Setup (First Time)

```bash
# 1. Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y wget git build-essential

# 2. Clone the repository
git clone https://github.com/EdgeVLM-Labs/gemma3-testing.git
cd gemma3-testing

# 3. Run automated setup script (THIS CREATES THE gemma3n ENVIRONMENT)
bash setup.sh

# 4. Activate the environment
conda activate gemma3n

# 5. Accept Conda Terms of Service (required for some packages)
# NOTE: If you get "EnvironmentNameNotFound", run step 3 first!
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# 6. Login to Hugging Face (required for gated models)
huggingface-cli login
# Or: hf auth login
# You'll need to:
#   - Create account at https://huggingface.co
#   - Generate token at https://huggingface.co/settings/tokens
#   - Request access to google/gemma-3n-E2B model

# 7. Download and prepare dataset
python dataset.py download          # Download 5 videos per exercise class
python dataset.py prepare           # Create train/val/test splits

# 8. Verify setup
bash scripts/verify_qved_setup.sh

# 9. (Optional) Run quick inference test
python gemma3n_batch_inference.py \
  --model unsloth/gemma-3n-E4B-it \
  --video_folder sample_videos \
  --output results/test.csv
```

### Daily Usage (After Initial Setup)

```bash
# Activate environment
conda activate gemma3n

# Fine-tune on QVED dataset
bash scripts/finetune_qved.sh

# Run inference on videos
python gemma3n_batch_inference.py \
  --model google/gemma-3n-E2B \
  --video_folder your_videos/ \
  --output results.csv
```

---

## 🚀 Features

- **Gemma-3N-E2B implementation** with dedicated folder structure
- **Fine-tuning** on QVED (Quality Video Exercise Dataset)
- **Batch video inference** for multiple videos
- **Single video inference** for quick testing
- **Dataset preparation** and quality control
- **Training visualization** and evaluation reports
- **Model uploading** to Hugging Face Hub
- **Modular architecture** with docs, eval, and training components

---

## 📁 Repository Structure

```
gemma3-testing/
├── dataset.py            # 🆕 Unified dataset management (download/prepare/clean)
├── gemma3n_batch_inference.py  # Batch inference (Unsloth FastModel)
├── setup.sh              # Environment setup script
├── requirements.txt      # Python dependencies
├── gemma3/               # 🆕 Gemma-3N-E2B implementation
│   ├── train/           # Training scripts (train.py, pretrain.py, trainer.py)
│   ├── model/           # Model architecture (arch, builder, dataloader)
│   ├── config/          # Configuration files
│   ├── docs/            # Gemma3-specific documentation
│   ├── eval/            # Evaluation scripts
│   └── README.md        # Gemma3 implementation guide
├── scripts/              # Training and inference scripts
│   ├── initialize_dataset.sh       # Dataset setup and training
│   ├── finetune_qved.sh           # QVED fine-tuning script
│   ├── run_inference.sh           # Test set inference & evaluation
│   └── verify_qved_setup.sh       # Dataset verification
├── utils/                # Utility scripts
│   ├── load_dataset.py            # [Legacy] Download QVED from HF
│   ├── qved_from_fine_labels.py   # [Legacy] Prepare train/val/test splits
│   ├── clean_dataset.py           # [Legacy] Video quality filtering
│   ├── augment_videos.py          # Video augmentation
│   ├── test_inference.py          # Batch test inference
│   ├── infer_qved.py              # Single video inference
│   ├── generate_test_report.py    # Evaluation report generator
│   ├── hf_upload.py               # Upload models to HF Hub
│   └── ...                        # Other utilities
├── mobilevideogpt/       # [Legacy] Original Mobile-VideoGPT architecture
├── docs/                 # General documentation
├── eval/                 # General evaluation scripts
└── README.md             # This file
```

### Gemma3 Folder Structure

The `gemma3/` folder contains the complete Gemma-3N-E2B implementation:

- **`train/`**: Training scripts (train.py, pretrain.py, trainer.py)
- **`model/`**: Model architecture, encoders, projectors
- **`config/`**: Dataset and model configurations
- **`docs/`**: Architecture docs, quickstart guides, troubleshooting
- **`eval/`**: Evaluation scripts and metrics
- **`README.md`**: Detailed implementation guide

See [gemma3/README.md](gemma3/README.md) and [gemma3/docs/ARCHITECTURE.md](gemma3/docs/ARCHITECTURE.md) for details.

---

## 🛠️ Detailed Installation

### Option 1: Automated Setup (Recommended)

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y wget git build-essential

# Clone repository
git clone https://github.com/EdgeVLM-Labs/gemma3-testing.git
cd gemma3-testing

# Run setup script (IMPORTANT: This creates the gemma3n environment)
bash setup.sh

# Activate environment
conda activate gemma3n
```

### Option 2: Manual Setup

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y wget git build-essential

# Clone repository
git clone https://github.com/EdgeVLM-Labs/gemma3-testing.git
cd gemma3-testing

# Create conda environment
conda create -n gemma3n python=3.11 -y
conda activate gemma3n

# Install dependencies
pip install -r requirements.txt

# Accept Conda ToS (if needed)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

### Post-Installation Steps

```bash
# 1. Login to Hugging Face
hf auth login
# Or set token:
export HF_TOKEN=your_token_here

# 2. Request access to Gemma-3N-E2B
# Visit: https://huggingface.co/google/gemma-3n-E2B
# Click "Request Access" and accept terms

# 3. Verify GPU availability
nvidia-smi

# 4. Test Python environment
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---
export HF_TOKEN=your_token_here
```

---

## 📊 Dataset Preparation

### Download QVED Dataset

```bash
# Download 5 videos per exercise class (default)
python dataset.py download

# Download more videos per class
python dataset.py download --max-per-class 10
```

### Prepare Train/Val/Test Splits

```bash
python dataset.py prepare
```

This creates:
- `dataset/qved_train.json` (60%)
- `dataset/qved_val.json` (20%)
- `dataset/qved_test.json` (20%)

### Verify Dataset

```bash
bash scripts/verify_qved_setup.sh
```

### Optional: Clean Low-Quality Videos

```bash
python dataset.py clean
```

Or run all steps at once:
```bash
python dataset.py all
```

Filters videos based on:
- Resolution (min 640x360)
- Brightness (35-190)
- Sharpness (min score 50)
- Motion detection (for dynamic exercises)

### Usage Examples

```bash
# Download dataset
python dataset.py download                    # Download 5 videos/class
python dataset.py download --max-per-class 10 # Download 10 videos/class

# Prepare splits
python dataset.py prepare                     # Create train/val/test splits

# Clean dataset
python dataset.py clean                       # Filter low-quality videos

# Run all steps
python dataset.py all                         # Download → Prepare → Clean

# Get help
python dataset.py --help                      # Show all options
```

---

## 🎯 Model Inference

### Single Video Inference (Quick Test)

Using Unsloth FastModel (recommended):

```bash
python gemma3n_batch_inference.py \
  --model unsloth/gemma-3n-E4B-it \
  --video_folder sample_videos \
  --output results/batch_results.csv
```

### Single Video with Custom Model

```bash
python utils/infer_qved.py \
  --model_path google/gemma-3n-E2B \
  --video_path sample_videos/00000340.mp4 \
  --prompt "Analyze the exercise form shown in this video"
```

### Batch Video Inference

Process all videos in a folder:

```bash
python gemma3n_batch_inference.py \
  --model unsloth/gemma-3n-E4B-it \
  --video_folder videos \
  --output results/gemma3n_results.csv \
  --num_frames 8 \
  --max_new_tokens 256
```

---

## 🔧 Fine-tuning

### Quick Start Fine-tuning

```bash
bash scripts/initialize_dataset.sh
```

This will:
1. Verify dataset setup
2. Configure training parameters
3. Start fine-tuning on QVED
4. Save checkpoints to `results/qved_finetune_gemma3n_E2B/`

### Custom Fine-tuning

Edit training parameters in `scripts/finetune_qved.sh`:

```bash
EPOCHS=3
LR=2e-4
BATCH=8
GACC=8
LORA_R=64
LORA_ALPHA=128
```

Then run:

```bash
bash scripts/finetune_qved.sh
```

---

## 📈 Evaluation

### Run Test Set Inference

```bash
bash scripts/run_inference.sh \
  --model_path results/qved_finetune_gemma3n_E2B/checkpoint-70 \
  --test_json dataset/qved_test.json \
  --output_dir results/test_results
```

This generates:
- `test_predictions.json` - Model predictions
- `test_evaluation_report.xlsx` - Detailed metrics (ROUGE-L, METEOR, BERT similarity)

### Generate Custom Report

```bash
python utils/generate_test_report.py \
  --predictions test_predictions.json \
  --output test_report.xlsx \
  --no-bert  # Skip BERT similarity (faster)
```

---

## ☁️ Model Upload

Upload fine-tuned model to Hugging Face Hub:

```bash
python utils/hf_upload.py \
  --model_path results/qved_finetune_gemma3n_E2B/checkpoint-70 \
  --repo_name gemma3n-qved-finetune \
  --org your-org-name \
  --private
```

---

## 🎨 Video Augmentation

Apply augmentations to videos:

```bash
python utils/augment_videos.py \
  --dataset_dir dataset \
  --folders 1,3,5 \
  --augmentations 1,3,5,7 \
  --run_inference
```

Available augmentations:
1. Horizontal Flip
2. Vertical Flip
3. Random Rotate
4. Random Resize
5. Gaussian Blur
6. Brightness
7. Multiply Brightness
...and more (see script for full list)

---

## 📝 Training Visualization

Plot training metrics from logs:

```bash
python utils/plot_training_stats.py \
  --log_file results/training.log \
  --output_dir plots/gemma3n
```

Generates:
- Loss curves (training & validation)
- Gradient norm
- Learning rate schedule
- Combined plots PDF

---

## 🔍 Troubleshooting

### Initial Setup Issues

**System Dependencies Missing:**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y wget git build-essential

# CentOS/RHEL
sudo yum install -y wget git gcc gcc-c++ make

# macOS (requires Homebrew)
brew install wget git
xcode-select --install
```

**Git Clone Fails:**
```bash
# If HTTPS fails, try SSH
git clone git@github.com:EdgeVLM-Labs/gemma3-testing.git

# Or download ZIP
wget https://github.com/EdgeVLM-Labs/gemma3-testing/archive/refs/heads/main.zip
unzip main.zip
```

**Setup Script Fails:**
```bash
# Make script executable
chmod +x setup.sh

# Run with bash explicitly
bash setup.sh

# Check for errors in output
bash setup.sh 2>&1 | tee setup.log
```

**Conda Environment Not Found:**
```bash
# If you get: EnvironmentNameNotFound: Could not find conda environment: gemma3n

# Option 1: Run the setup script (recommended)
bash setup.sh

# Option 2: Create environment manually
conda create -n gemma3n python=3.11 -y
conda activate gemma3n
pip install -r requirements.txt

# Then verify it exists
conda info --envs
```

**Conda Environment Not Activating:**
```bash
# Source conda
source ~/miniconda3/etc/profile.d/conda.sh

# Or add to ~/.bashrc
echo 'source ~/miniconda3/etc/profile.d/conda.sh' >> ~/.bashrc
source ~/.bashrc

# Then activate
conda activate gemma3n
```

**Python Module Not Found:**
```bash
# Reinstall requirements
conda activate gemma3n
pip install -r requirements.txt --force-reinstall

# Or install individually
pip install torch transformers unsloth peft
```

### Runtime Issues

**Out of Disk Space:**
```bash
# Clean HF cache
rm -rf ~/.cache/huggingface/hub/*

# Clean conda cache
conda clean --all

# Check disk usage
df -h
```

**Gated Model Access (403 Error):**
```bash
# 1. Request access at https://huggingface.co/google/gemma-3n-E2B
# 2. Accept terms (wait for approval email)
# 3. Login with your token
hf auth login

# Alternative: Use public mirror
--model unsloth/gemma-3n-E2B
```

**CUDA Out of Memory:**
```bash
# Reduce batch size in scripts/finetune_qved.sh
BATCH=4  # instead of 8

# Or increase gradient accumulation
GACC=16  # instead of 8

# Clear CUDA cache
python -c "import torch; torch.cuda.empty_cache()"
```

**PEFT Import Error:**
```bash
# Upgrade PEFT (needs >=0.11.0)
pip install --upgrade peft

# Verify version
python -c "import peft; print(peft.__version__)"
```

**Mamba-SSM Installation Fails:**
```bash
# Ensure PyTorch is installed first
pip install torch

# Then install mamba-ssm
pip install mamba-ssm==1.2.0

# If still fails, try with --no-cache-dir
pip install --no-cache-dir mamba-ssm==1.2.0
```

**Dataset Not Found:**
```bash
# Verify dataset exists
ls -la dataset/

# Re-download if needed
python dataset.py download
python dataset.py prepare

# Check video paths
bash scripts/verify_qved_setup.sh
```

### Need More Help?

- Check [docs/issues.md](docs/issues.md) for known issues
- See [gemma3/docs/ARCHITECTURE.md](gemma3/docs/ARCHITECTURE.md) for technical details
- Review [docs/QUICKSTART.md](docs/QUICKSTART.md) for step-by-step guide
- Open an issue on GitHub: https://github.com/EdgeVLM-Labs/gemma3-testing/issues

---

## 📚 Key Scripts Reference

| Script | Purpose |
|--------|---------|
| `gemma3n_batch_inference.py` | Batch video inference with Unsloth |
| `dataset.py` | Download, prepare, and clean QVED dataset (replaces utils/load_dataset.py, utils/qved_from_fine_labels.py, utils/clean_dataset.py) |
| `utils/load_dataset.py` | [Legacy] Download QVED dataset |
| `utils/qved_from_fine_labels.py` | [Legacy] Create train/val/test splits |
| `utils/clean_dataset.py` | [Legacy] Filter low-quality videos |
| `utils/test_inference.py` | Run inference on test set |
| `utils/generate_test_report.py` | Generate evaluation report |
| `utils/hf_upload.py` | Upload model to HF Hub |
| `scripts/initialize_dataset.sh` | Setup + fine-tuning |
| `scripts/run_inference.sh` | Test inference + evaluation |

---

## 🤝 Contributing

Contributions welcome! Please ensure:
- Code follows existing style
- Scripts include proper argument parsing
- README is updated for new features

---

## 📄 License

See [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Gemma-3N-E2B** by Google
- **Unsloth** for efficient fine-tuning
- **QVED Dataset** contributors

---

## 📧 Contact

For questions or issues, please open a GitHub issue or contact the maintainers.
