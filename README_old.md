# Gemma-3N-E2B Video Understanding & Fine-tuning

This repository provides tools for fine-tuning and inference with **Google Gemma-3N-E2B** models for video understanding tasks, with a focus on exercise form analysis and assistive navigation for visually impaired users.

---

## 🎯 Quick Start

**� Complete Guide: See [Fine-tuning Guide](docs/FINETUNE_GUIDE.md) for detailed instructions!**

### Prerequisites
- Linux/macOS (Ubuntu 20.04+ recommended)
- NVIDIA GPU with CUDA support (16GB+ VRAM recommended)
- 50GB+ free disk space
- Git and basic command line knowledge

### Fast Setup

```bash
# 1. Clone repository
git clone https://github.com/EdgeVLM-Labs/gemma3-testing.git
cd gemma3-testing

# 2. Setup environment
bash setup.sh
conda activate gemma3n

# 3. Login to services
wandb login
huggingface-cli login

# 4. Prepare dataset
python dataset.py download --max-per-class 5
python dataset.py prepare

# 5. Fine-tune model
bash scripts/finetune_gemma3n_unsloth.sh

# 6. Run inference
python scripts/run_inference_unsloth.py \
    --model_path outputs/gemma3n_finetune_YYYYMMDD_HHMMSS_merged_16bit \
    --video_path dataset/exercise_class_1/video_001.mp4
```

---

## 📚 Documentation

- **[Fine-tuning Guide](docs/FINETUNE_GUIDE.md)** - Complete guide for training and inference
- [Setup Checklist](docs/SETUP_CHECKLIST.md) - Step-by-step setup with checkboxes
- [Quick Start](docs/QUICKSTART.md) - Original quick start guide

---

## 🔧 Initial Setup (Detailed)

### Step 1: System Dependencies
```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y wget git build-essential
```

### Step 2: Clone Repository
```bash
git clone https://github.com/EdgeVLM-Labs/gemma3-testing.git
cd gemma3-testing
```

### Step 3: Environment Setup
```bash
# Run automated setup script (creates gemma3n environment)
bash setup.sh

# Activate the environment
conda activate gemma3n
```

### Step 4: Authentication
```bash
# Login to Hugging Face (required for models and datasets)
huggingface-cli login

# Login to Weights & Biases (required for training tracking)
wandb login
```
#   - Create account at https://huggingface.co
#   - Generate token at https://huggingface.co/settings/tokens
#   - Request access to google/gemma-3n-E2B model

# Login to WandB (for training monitoring)
wandb login
```

#### **Step 5: Dataset Preparation (Recommended Workflow)**
```bash
# Download videos from HuggingFace dataset
python dataset.py download --max-per-class 5

# Clean dataset (filter low-quality videos)
python dataset.py clean

# Copy cleaned videos to inference folder (keeps original filenames)
python dataset.py copy

# Create train/val/test splits from cleaned dataset
python dataset.py prepare

# Verify everything is set up correctly
bash scripts/verify_qved_setup.sh
```

**Dataset Workflow Summary:**
1. **Download** → Downloads raw videos to `dataset/`
2. **Clean** → Filters quality and saves to `cleaned_dataset/`
3. **Copy** → Copies cleaned videos to `videos/` for inference
4. **Prepare** → Creates train/val/test JSON splits in `dataset/`

#### **Step 6: Test Inference (Optional)**
```bash
# Test batch inference on 5 videos (default)
python gemma3n_batch_inference.py \
  --model unsloth/gemma-3n-E4B-it \
  --video_folder videos \
  --output results/test_inference.csv

# Or test single video
python utils/infer_qved.py \
  --model_path unsloth/gemma-3n-E4B-it \
  --video_path videos/sample.mp4 \
  --prompt "Analyze the exercise form shown in this video"
```

---

## 🔧 Troubleshooting

### Common Issues and Fixes

#### **Issue 1: `module 'torch' has no attribute 'int1'`**
**Solution:** Install PyTorch nightly with torch.int1 support
```bash
bash fix_torch_int1.sh
```

#### **Issue 2: `ImportError: cannot import name 'dequantize_module_weight'`**
**Solution:** Install compatible unsloth and peft versions
```bash
pip install --upgrade unsloth unsloth_zoo
```

#### **Issue 3: `ModuleNotFoundError: No module named 'mamba_ssm'`**
**Solution:** Reinstall mamba-ssm
```bash
pip uninstall -y mamba-ssm
pip cache purge
pip install mamba-ssm --no-cache-dir --no-build-isolation
```

#### **Issue 4: Torchvision version mismatch**
**Solution:** Run the fix script which handles all version issues
```bash
bash fix_torch_int1.sh
```

#### **Issue 5: Training dependencies not working**
**Solution:** Run complete fine-tuning environment setup
```bash
bash finetune_env.sh
```

This installs:
- PyTorch nightly with CUDA 12.1
- TorchAO with torch.int1 support
- Unsloth & Unsloth Zoo
- Mamba-SSM (recompiled)
- All training dependencies

---

## 📚 Usage Guide

### Daily Usage (After Initial Setup)

#### **Activate Environment**
```bash
conda activate gemma3n
```

#### **Fine-tuning Workflow**

**Option 1: Unsloth Fine-tuning (Recommended - Faster)**
```bash
# Prepare environment (first time only)
bash finetune_env.sh

# Run fine-tuning with Unsloth
bash scripts/finetune_gemma3n_unsloth.sh

# This will:
# - Load unsloth/gemma-3n-E4B-it
# - Train with LoRA (r=16, alpha=32)
# - Save adapter to results/gemma3n_E4B_finetune
# - Save merged model to results/gemma3n_E4B_finetune_merged
```

**Option 2: Standard Fine-tuning**
```bash
# Full training with DeepSpeed
bash scripts/finetune_qved.sh

# This uses:
# - Amshaker/Mobile-VideoGPT-0.5B (Qwen2-based)
# - LoRA (r=64, alpha=128)
# - Effective batch size: 64
```

#### **Inference Workflow**

**Batch Inference (Multiple Videos)**
```bash
# Process up to 5 videos (default)
python gemma3n_batch_inference.py \
  --model unsloth/gemma-3n-E4B-it \
  --video_folder videos \
  --output results/batch_results.csv

# Process specific number of videos
python gemma3n_batch_inference.py \
  --model unsloth/gemma-3n-E4B-it \
  --video_folder videos \
  --output results/batch_results.csv \
  --max_videos 10

# Use fine-tuned model
python gemma3n_batch_inference.py \
  --model results/gemma3n_E4B_finetune_merged \
  --video_folder videos \
  --output results/finetuned_results.csv
```

**Single Video Inference**
```bash
python utils/infer_qved.py \
  --model_path unsloth/gemma-3n-E4B-it \
  --video_path videos/sample.mp4 \
  --prompt "Analyze the exercise form shown in this video"
```

#### **Dataset Management**
```bash
# Download more videos
python dataset.py download --max-per-class 10

# Clean and prepare
python dataset.py clean
python dataset.py copy
python dataset.py prepare

# Or run all steps at once
python dataset.py all --max-per-class 10
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
├── dataset.py                    # 🆕 Unified dataset management
├── gemma3n_batch_inference.py    # Batch video inference (supports --max_videos)
├── gemma3_finetune_unsloth.py    # Unsloth fine-tuning script
├── setup.sh                      # Initial environment setup
├── finetune_env.sh               # 🆕 Complete fine-tuning dependencies
├── fix_torch_int1.sh             # 🆕 Fix PyTorch/TorchAO compatibility
├── requirements.txt              # Python dependencies
├── gemma3/                       # 🆕 Gemma-3N-E2B implementation
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

# Copy videos for batch inference
python dataset.py copy                        # Copy 5 random videos to videos/
python dataset.py copy --num-videos 10        # Copy 10 videos
python dataset.py copy --output my_videos     # Copy to custom folder

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
python utils/infer_qved.py \
  --model_path unsloth/gemma-3n-E4B-it \
  --video_path sample_videos/00000340.mp4 \
  --prompt "Analyze the exercise form shown in this video"
```

Available models:
- `unsloth/gemma-3n-E4B-it` - 4B parameter model (recommended)
- `unsloth/gemma-3n-E2B-it` - 2B parameter model (faster)
- `google/gemma-3n-E2B` - Original Google model (requires different setup)

### Batch Video Inference

Process multiple videos at once:

```bash
# Step 1: Copy videos for inference
python dataset.py copy --num-videos 5

# Step 2: Run batch inference
python gemma3n_batch_inference.py \
  --model unsloth/gemma-3n-E4B-it \
  --video_folder videos \
  --output results/batch_results.csv \
  --num_frames 8 \
  --max_new_tokens 256

# Step 3: View results
cat results/batch_results.csv
```

**Custom prompt:**
```bash
python gemma3n_batch_inference.py \
  --model unsloth/gemma-3n-E4B-it \
  --video_folder videos \
  --output results/batch_results.csv \
  --prompt "Provide detailed feedback on exercise form and safety"
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

### Unsloth LoRA Fine-tuning (Gemma-3N-E4B-It)

```bash
# Ensure dependencies expose Float8 quantization support
pip install --upgrade --force-reinstall \
  "torch==2.6.0+cu121" \
  "torchvision==0.21.0+cu121" \
  "torchaudio==2.6.0+cu121" \
  --index-url https://download.pytorch.org/whl/cu121

pip install --upgrade --pre torchao==0.6.0.dev20241205 \
  --index-url https://download.pytorch.org/whl/cu121

pip install --no-deps --upgrade git+https://github.com/unslothai/unsloth.git

# Optional: verify Float8 APIs exist
python -c "from torchao.quantization import Float8WeightOnlyConfig"

# Run SFT with LoRA adapters
bash scripts/finetune_gemma3n_unsloth.sh
```

Key files:
- [gemma3_finetune_unsloth.py](gemma3_finetune_unsloth.py) orchestrates dataset loading, LoRA configuration, and SFTTrainer.
- [scripts/finetune_gemma3n_unsloth.sh](scripts/finetune_gemma3n_unsloth.sh) wraps environment checks and launches training.

The script expects `dataset.py prepare` output and logs metrics to Weights & Biases if `WANDB_PROJECT` is set.

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

**PEFT/Unsloth Import Error:**
```bash
# Unsloth requires specific versions
pip install --upgrade "unsloth>=2026.1.2" "peft>=0.17.0,<=0.18.0"

# Or reinstall all requirements
pip install -r requirements.txt --force-reinstall

# Verify versions
python -c "import unsloth; import peft; print(f'Unsloth: {unsloth.__version__}, PEFT: {peft.__version__}')"
```

**Float8WeightOnlyConfig Missing (torchao):**
```bash
pip install --upgrade --force-reinstall \
  "torch==2.6.0+cu121" \
  "torchvision==0.21.0+cu121" \
  "torchaudio==2.6.0+cu121" \
  --index-url https://download.pytorch.org/whl/cu121

pip install --upgrade --pre torchao==0.6.0.dev20241205 \
  --index-url https://download.pytorch.org/whl/cu121

pip install --no-deps --upgrade git+https://github.com/unslothai/unsloth.git

python -c "from torchao.quantization import Float8WeightOnlyConfig"
```
These builds expose Float8 quantization APIs required by transformers and Unsloth LoRA fine-tuning.

**Mamba-SSM Installation Fails (for training):**
```bash
# Note: Only needed for training with mobilevideogpt architecture
# Inference uses unsloth which doesn't require mamba-ssm

pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip uninstall mamba-ssm

pip cache purge

pip install mamba-ssm --no-cache-dir --no-build-isolation

# Verify installation
python -c "import mamba_ssm; print('mamba-ssm installed successfully')"
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
| `gemma3n_batch_inference.py` | Batch video inference with Unsloth FastModel |
| `utils/infer_qved.py` | Single video inference with Unsloth FastModel |
| `dataset.py` | Download, prepare, clean, and copy QVED dataset (unified tool) |
| `dataset.py copy` | Copy random videos to flat folder for batch inference |
| `utils/load_dataset.py` | [Legacy] Download QVED dataset |
| `utils/qved_from_fine_labels.py` | [Legacy] Create train/val/test splits |
| `utils/clean_dataset.py` | [Legacy] Filter low-quality videos |
| `utils/test_inference.py` | Run inference on test set |
| `utils/generate_test_report.py` | Generate evaluation report |
| `utils/hf_upload.py` | Upload model to HF Hub |
| `scripts/initialize_dataset.sh` | Setup + fine-tuning |
| `scripts/finetune_gemma3n_unsloth.sh` | Unsloth LoRA fine-tuning for Gemma-3N |
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
