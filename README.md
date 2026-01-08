# Gemma-3N Fine-tuning for Video Understanding

Fine-tune Google's **Gemma-3N (unsloth/gemma-3n-E4B-it)** model on video datasets using **Unsloth FastVisionModel** for efficient LoRA training. Optimized for exercise form analysis and video understanding tasks.

**🎯 Model:** `unsloth/gemma-3n-E4B-it` | **📊 Dataset:** EdgeVLM-Labs/QVED-Test-Dataset | **⚡ Framework:** Unsloth

---

## 🚀 Quick Start

```bash
# 1. Clone and setup environment
git clone https://github.com/EdgeVLM-Labs/gemma3-testing.git
cd gemma3-testing
bash setup.sh

# If setup says "restart terminal", do this:
# - Close and reopen terminal, then:
cd gemma3-testing
bash setup.sh  # Run again

# 2. Activate environment (always needed in new terminals)
conda activate gemma3n

# 3. Login to services
wandb login
huggingface-cli login

# 4. Prepare dataset
python dataset.py download --max-per-class 5
python dataset.py prepare

# 5. Fine-tune model (uses unsloth/gemma-3n-E4B-it)
bash scripts/finetune_gemma3n_unsloth.sh

# 6. Run inference
python utils/infer_qved.py \
    --model_path outputs/gemma3n_finetune_YYYYMMDD_HHMMSS_merged_16bit \
    --video_path sample_videos/00000340.mp4 \
    --prompt "Analyze the exercise form shown in this video"

# 7. Evaluate model (limit to 50 samples for quick test)
python eval/eval_gemma3n.py \
    --model_path outputs/gemma3n_finetune_YYYYMMDD_HHMMSS_merged_16bit \
    --eval_json dataset/qved_val.json \
    --output_file results/eval_results.json \
    --max_samples 50
```

**📖 Complete Guide:** See [docs/FINETUNE_GUIDE.md](docs/FINETUNE_GUIDE.md) for detailed instructions.

---

## 📋 Prerequisites

- **OS:** Linux/macOS (Ubuntu 20.04+ recommended)
- **GPU:** NVIDIA GPU with 16GB+ VRAM (A100/H100 recommended for full training)
- **Storage:** 50GB+ free disk space
- **Software:** 
  - Git
  - CUDA 11.8+ / 12.0+
  - Python 3.11
  - Conda/Miniconda

---

## 🔧 Installation

### Step-by-Step Setup Process

#### 1. Clone Repository

```bash
git clone https://github.com/EdgeVLM-Labs/gemma3-testing.git
cd gemma3-testing
```

#### 2. Run Setup Script (First Time)

```bash
bash setup.sh
```

**What happens:**
- **If Miniconda not installed:** 
  - Script installs Miniconda
  - **STOPS and asks you to restart terminal**
  - After restart, run `bash setup.sh` again
  
- **If Miniconda already installed:**
  - Creates `gemma3n` environment with Python 3.11
  - Activates environment automatically
  - Installs all dependencies
  - Environment is ready to use!

#### 3. If Setup Asked You to Restart

After running setup.sh for the first time on a fresh system:

```bash
# Restart your terminal, then:
cd gemma3-testing
bash setup.sh  # Run again to complete installation
```

The second run will:
- ✅ Create conda environment
- ✅ Install all packages
- ✅ Verify installation
- ✅ Environment will be activated and ready!

#### 4. For Future Sessions

Every time you open a new terminal:

```bash
conda activate gemma3n
```

**What gets installed:**
- `transformers==4.56.2` (required version for Gemma-3N)
- `trl==0.22.2` (for SFTTrainer)
- `unsloth` + `unsloth_zoo` (efficient LoRA training)
- `torch>=2.1.0`, `accelerate`, `bitsandbytes`
- Vision: `opencv-python`, `timm`, `Pillow`
- Dataset: `datasets>=4.3.0`, `huggingface_hub`
- Training: `wandb` (for tracking)
- Evaluation: `nltk`, `rouge-score`, `sacrebleu`, `openpyxl`, `sentence-transformers`

### 5. Authenticate Services

After setup completes, login to required services:

```bash
# HuggingFace (required - for models and datasets)
huggingface-cli login
# Paste your token from: https://huggingface.co/settings/tokens

# Weights & Biases (required - for training tracking)
wandb login
# Paste your API key from: https://wandb.ai/authorize
```

**Note:** The setup script will prompt you for these logins at the end.

### 6. Verify Installation
wandb login
# Paste your API key from: https://wandb.ai/authorize
```

### 6. Verify Installation

Check that everything is working correctly:

```bash
# Ensure environment is activated
conda activate gemma3n

# Check GPU and CUDA
nvidia-smi
nvcc --version

# Check PyTorch CUDA support
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Check Unsloth FastVisionModel
python -c "from unsloth import FastVisionModel; print('✅ Unsloth FastVisionModel ready')"

# Check transformers version
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

Expected output:
```
PyTorch: 2.x.x+cu118
CUDA available: True
✅ Unsloth FastVisionModel ready
Transformers: 4.56.2
```

---

## 🔧 Troubleshooting Setup

### Issue: "conda: command not found" after setup

**Solution:**
```bash
# Restart terminal, then:
source ~/.bashrc
conda activate gemma3n
```

### Issue: Environment not activating

**Solution:**
```bash
# Initialize conda in current shell
eval "$(conda shell.bash hook)"
conda activate gemma3n

# Or restart terminal and try again
```

### Issue: "Setup failed during package installation"

**Solution:**
```bash
# Delete environment and start fresh
conda env remove -n gemma3n
bash setup.sh
```

### Issue: ImportError for specific packages

**Solution:**
```bash
conda activate gemma3n
pip install -r requirements.txt
```

---

## 📊 Dataset Management

The `dataset.py` script manages the **QVED (Quality Video Exercise Dataset)** from EdgeVLM-Labs/QVED-Test-Dataset:

### Download Videos

```bash
# Download 5 videos per exercise class (fast - for testing)
python dataset.py download --max-per-class 5

# Download more videos (for production training)
python dataset.py download --max-per-class 20

# Download all available videos
python dataset.py download
```

Videos are saved to `videos/` directory.

### Prepare Train/Val/Test Splits

```bash
python dataset.py prepare
```

Creates JSON files with 60/20/20 split:
- `dataset/qved_train.json` - Training set (60%)
- `dataset/qved_val.json` - Validation set (20%)
- `dataset/qved_test.json` - Test set (20%)

Each JSON entry contains:
```json
{
  "video": "videos/00000340.mp4",
  "conversations": [
    {"from": "human", "value": "Analyze this exercise..."},
    {"from": "gpt", "value": "The form shows..."}
  ]
}
```

### Optional: Clean Dataset

```bash
# Filter low-quality videos (resolution, brightness, sharpness checks)
python dataset.py clean
```

### Copy Videos for Testing

```bash
# Copy subset to flat folder for batch inference
python dataset.py copy --num-videos 10 --output test_videos
```

### All-in-One Workflow

```bash
# Download, prepare, and clean in one command
python dataset.py all --max-per-class 10
```

---

## 🎯 Fine-tuning (unsloth/gemma-3n-E4B-it)

### Method 1: Quick Start (Recommended)

The easiest way - uses local dataset prepared by `dataset.py`:

```bash
bash scripts/finetune_gemma3n_unsloth.sh
```

**Default Configuration:**
- **Model:** `unsloth/gemma-3n-E4B-it` (4B parameters, instruction-tuned)
- **Dataset:** Local QVED (`dataset/qved_train.json` + videos in `videos/`)
- **LoRA:** r=64, alpha=128, dropout=0.0, target_modules=all-linear
- **Training:** 
  - Batch size: 1 per device
  - Gradient accumulation: 4 steps
  - Effective batch size: 4
  - Learning rate: 2e-4 with cosine schedule
  - Warmup ratio: 3%
  - Max gradient norm: 0.3
  - Weight decay: 0.001
  - Epochs: 1 (adjust in script)
- **Video:** 8 frames per video
- **Max sequence length:** 50000 tokens (required for vision)
- **Optimizer:** AdamW fused
- **Tracking:** WandB (real-time metrics)

### Method 2: Custom Configuration

For advanced users who want to customize hyperparameters:

```bash
python gemma3_finetune_unsloth.py \
    --model_name unsloth/gemma-3n-E4B-it \
    --train_json dataset/qved_train.json \
    --val_json dataset/qved_val.json \
    --output_dir outputs/custom_finetune \
    --num_epochs 3 \
    --batch_size 1 \
    --gradient_accumulation 8 \
    --learning_rate 2e-4 \
    --max_seq_length 50000 \
    --lora_r 128 \
    --lora_alpha 256 \
    --num_frames 8 \
    --wandb_project "My-Gemma3N-Project" \
    --wandb_run_name "custom-run-001"
```

### Method 3: HuggingFace Streaming (Alternative)

Download dataset during training (not recommended - uses HF cache):

```bash
bash scripts/finetune_gemma3n_unsloth.sh --hf
```

### Monitor Training in Real-Time

1. Training starts with WandB logging
2. Visit https://wandb.ai/fyp-21/Finetune-gemma3n (your project)
3. View live metrics updated every step:
   - **train/loss:** Training loss per step
   - **train/grad_norm:** Gradient magnitude
   - **train/learning_rate:** Current LR (cosine schedule)
   - **train/epoch:** Training progress
   - **eval/loss:** Validation loss (every N steps if enabled)
   - **eval/runtime:** Evaluation time
   - **eval/samples_per_second:** Eval throughput

Console output shows:
```
{'loss': 1.0995, 'grad_norm': 27.17, 'learning_rate': 0.0002, 'epoch': 0.1}
{'eval_loss': 0.8543, 'eval_runtime': 12.5, 'eval_samples_per_second': 6.8}  # Every EVAL_STEPS
```

### Evaluation During Training (Enabled by Default)

Validation tracking is **enabled by default** in [finetune_gemma3n_unsloth.sh](scripts/finetune_gemma3n_unsloth.sh):

```bash
# Evaluation configuration (already enabled)
RUN_EVAL="--run_eval"              # Evaluation enabled
EVAL_STEPS=25                       # Eval every 25 steps (more frequent tracking)
SAVE_EVAL_CSV="--save_eval_csv"    # Save results as CSV after training
GENERATE_REPORT=""                  # Set to "--generate_report" for Excel report
```

**What happens during evaluation:**
- ✅ Runs automatically every 25 steps (configurable)
- ✅ Evaluates on full validation set
- ✅ Logs `eval/loss`, `eval/runtime`, `eval/samples_per_second` to WandB
- ✅ Saves best model checkpoint based on eval loss
- ✅ Shows evaluation progress in console
- ✅ Exports results to CSV after training completes

**To disable evaluation** (faster training, no validation tracking):
```bash
RUN_EVAL=""  # Remove --run_eval flag
```

**To increase evaluation frequency** (more data points in WandB):
```bash
EVAL_STEPS=10  # Eval every 10 steps (slower but more tracking)
```

Output files after training with eval:
```
outputs/gemma3n_finetune_20260108_151431/
├── eval_results.csv              # Evaluation metrics CSV
├── eval_predictions.json         # Model predictions on val set
└── eval_report.xlsx              # Excel report with similarity scores
```

### Training Output Files

After training completes:

```
outputs/
├── gemma3n_finetune_20260108_151431/
│   ├── adapter_config.json           # LoRA configuration
│   ├── adapter_model.safetensors     # LoRA weights (small ~600MB)
│   └── checkpoint-*/                  # Intermediate checkpoints
│
└── gemma3n_finetune_20260108_151431_merged_16bit/
    ├── config.json
    ├── model.safetensors              # Full merged model (~16GB)
    ├── tokenizer.json
    └── processor_config.json          # Use this for inference!
```

**Which to use:**
- **For inference:** Use `*_merged_16bit/` directory
- **For further training:** Use base directory with adapter weights

---

## � Upload to HuggingFace

### Automatic Upload After Training

Enable automatic upload by setting flags in [finetune_gemma3n_unsloth.sh](scripts/finetune_gemma3n_unsloth.sh):

```bash
# Edit the script
UPLOAD_TO_HF="--upload_to_hf"     # Enable upload
HF_REPO_NAME=""                    # Auto-generate name
HF_PRIVATE="--hf_private"          # Make private (optional)

# Run training (will auto-upload at end)
bash scripts/finetune_gemma3n_unsloth.sh
```

### Manual Upload After Training

Upload a saved model manually:

```bash
# Upload merged 16-bit model (recommended)
python utils/hf_upload.py \
    --model_path outputs/gemma3n_finetune_20260108_162806_merged_16bit \
    --repo_name my-gemma3n-finetune

# Upload with custom settings
python utils/hf_upload.py \
    --model_path outputs/gemma3n_finetune_20260108_162806_merged_16bit \
    --repo_name my-custom-model \
    --private

# Upload LoRA adapters only
python utils/hf_upload.py \
    --model_path outputs/gemma3n_finetune_20260108_162806
```

### Use Uploaded Model

Once uploaded, anyone can use your model:

```python
from unsloth import FastVisionModel

# Load from HuggingFace
model, processor = FastVisionModel.from_pretrained(
    "your-username/my-gemma3n-finetune"
)
```

---

## �🔮 Inference

### Single Video Analysis

Use the dedicated inference script for analyzing individual videos:

```bash
# With fine-tuned model
python utils/infer_qved.py \
    --model_path outputs/gemma3n_finetune_20260108_151431_merged_16bit \
    --video_path sample_videos/00000340.mp4\
    --prompt "Analyze the exercise form shown in this video" \
    --num_frames 8 \
    --max_new_tokens 512 \
    --show_stream

# With base model (no fine-tuning)
python utils/infer_qved.py \
    --model_path unsloth/gemma-3n-E4B-it \
    --video_path sample_videos/00000340.mp4 \
    --prompt "What is shown in this video?"
```

**Output example:**
```
================================================================================
QVED Inference - Gemma-3N Video Analysis
================================================================================

📦 Loading model: outputs/gemma3n_finetune_20260108_151431_merged_16bit...
✅ Model loaded successfully!

🎥 Processing video: sample_videos/00000340.mp4
✅ Extracted 8 frames from video
💬 Prompt: Analyze the exercise form shown in this video

================================================================================
🤖 Gemma-3N Output:
The individual is performing a barbell back squat with good depth and control. 
The bar path is relatively straight, and the lifter maintains proper back angle 
throughout the movement. Minor improvement could be made in keeping the knees 
tracking over the toes consistently.
================================================================================
```

### Batch Inference (Multiple Videos)

For processing multiple videos:

```bash
python scripts/run_inference_unsloth.py \
    --model_path outputs/gemma3n_finetune_merged_16bit \
    --video_dir test_videos/ \
    --output_file results/batch_predictions.json
```

---

## 📊 Evaluation

Evaluate model performance on validation/test sets with automatic metrics:

### Evaluate Fine-tuned Model

```bash
# Full evaluation (all samples)
python eval/eval_gemma3n.py \
    --model_path outputs/gemma3n_finetune_20260108_151431_merged_16bit \
    --eval_json dataset/qved_val.json \
    --output_file results/eval_finetuned.json \
    --num_frames 8 \
    --max_new_tokens 256

# Quick test (50 samples)
python eval/eval_gemma3n.py \
    --model_path outputs/gemma3n_finetune_20260108_151431_merged_16bit \
    --eval_json dataset/qved_val.json \
    --output_file results/eval_finetuned.json \
    --max_samples 50
```

### Evaluate Base Model (for comparison)

```bash
# Full evaluation
python eval/eval_gemma3n.py \
    --model_path unsloth/gemma-3n-E4B-it \
    --eval_json dataset/qved_val.json \
    --output_file results/eval_base.json \
    --num_frames 8

# Quick test (50 samples)
python eval/eval_gemma3n.py \
    --model_path unsloth/gemma-3n-E4B-it \
    --eval_json dataset/qved_val.json \
    --output_file results/eval_base.json \
    --max_samples 50
```

### Quick Test (First 10 samples)

```bash
python eval/eval_gemma3n.py \
    --model_path outputs/gemma3n_finetune_merged_16bit \
    --eval_json dataset/qved_val.json \
    --output_file results/eval_quick.json \
    --max_samples 10
```

### Evaluation Metrics

The script calculates:
- **BLEU:** Measures n-gram overlap with reference
- **ROUGE-1:** Unigram overlap
- **ROUGE-2:** Bigram overlap  
- **ROUGE-L:** Longest common subsequence

**Example output:**
```
================================================================================
📊 Evaluation Results
================================================================================
Samples evaluated: 84
Samples skipped: 0

Average Metrics:
  BLEU:    0.4523
  ROUGE-1: 0.6234
  ROUGE-2: 0.4156
  ROUGE-L: 0.5789
================================================================================

💾 Results saved to: results/eval_finetuned.json

================================================================================
📝 Sample Predictions (first 3)
================================================================================

--- Sample 1 ---
Prompt: Please evaluate the exercise form shown...
Ground Truth: The squat form shows good depth with neutral spine.
Prediction: The individual demonstrates excellent squat form with proper depth.
Metrics: BLEU=0.521, ROUGE-L=0.634
```

### Compare Base vs Fine-tuned

```bash
# Run both evaluations (limit to 50 samples for quick comparison)
python eval/eval_gemma3n.py --model_path unsloth/gemma-3n-E4B-it --eval_json dataset/qved_val.json --output_file results/base.json --max_samples 50
python eval/eval_gemma3n.py --model_path outputs/gemma3n_finetune_merged_16bit --eval_json dataset/qved_val.json --output_file results/finetuned.json --max_samples 50

# Compare results manually or use plotting tools
python utils/plot_training_stats.py results/
```

### Test Set Inference & Evaluation Report

Run full test set inference and generate a detailed Excel evaluation report with BERT/ROUGE/METEOR metrics:

```bash
# Test with your fine-tuned model (default: 50 samples)
bash scripts/run_inference.sh \
    --model_path outputs/gemma3n_finetune_20260108_162806_merged_16bit

# Test with base model for comparison
bash scripts/run_inference.sh \
    --model_path unsloth/gemma-3n-E4B-it

# Full test set (all samples)
bash scripts/run_inference.sh \
    --model_path outputs/gemma3n_finetune_20260108_162806_merged_16bit \
    --limit ""

# Fast evaluation without BERT (uses ROUGE/METEOR only)
bash scripts/run_inference.sh \
    --model_path outputs/gemma3n_finetune_20260108_162806_merged_16bit \
    --no-bert
```

**What happens:**
1. Runs inference on test set ([qved_test.json](dataset/qved_test.json))
2. Generates predictions JSON with ground truth comparison
3. Creates Excel report with:
   - BERT Similarity scores (semantic similarity)
   - ROUGE-L scores (n-gram overlap)
   - METEOR scores (word-level matching)
   - Exercise identification accuracy
   - Color-coded results (green/yellow/red)
   - Charts and summary statistics

**Output files:**
```
results/test_inference_<model_name>/
├── test_predictions.json          # Raw predictions with metrics
└── test_evaluation_report.xlsx    # Excel report with charts
```

**Manual test report generation:**

If you already have predictions JSON, generate report separately:

```bash
# Generate report with BERT similarity
python utils/generate_test_report.py \
    --predictions results/test_inference_model/test_predictions.json \
    --output test_report.xlsx

# Without BERT (faster)
python utils/generate_test_report.py \
    --predictions test_predictions.json \
    --no-bert
```

---

## ⚙️ Configuration Reference

### Model Options

| Model | Parameters | Context | Use Case |
|-------|-----------|---------|----------|
| `unsloth/gemma-3n-E4B-it` | 4B | 50K tokens | **Recommended** - Best balance |
| `unsloth/gemma-3n-E2B-it` | 2B | 50K tokens | Faster, lower memory |

### LoRA Hyperparameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `lora_r` | 64 | 8-128 | Rank (higher = more capacity) |
| `lora_alpha` | 128 | 16-256 | Scaling factor (typically 2×r) |
| `lora_dropout` | 0.0 | 0.0-0.1 | Dropout (0 for no dropout) |

### Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 1 | Per-device batch size |
| `gradient_accumulation` | 4 | Effective batch = batch_size × this |
| `learning_rate` | 2e-4 | Initial learning rate |
| `num_epochs` | 1 | Training epochs (adjust based on dataset size) |
| `max_seq_length` | 50000 | Max tokens (required for 8 frames) |
| `warmup_ratio` | 0.03 | 3% of steps for warmup |
| `max_grad_norm` | 0.3 | Gradient clipping |
| `weight_decay` | 0.001 | L2 regularization |

### Video Processing

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_frames` | 8 | Frames extracted per video |
| Frame extraction | `numpy.linspace` | Even spacing across video |

### Memory Requirements

| Configuration | VRAM | Training Speed |
|--------------|------|----------------|
| Batch=1, Grad=4, R=64 | ~24GB | Baseline |
| Batch=1, Grad=8, R=64 | ~24GB | 2× slower |
| Batch=1, Grad=4, R=128 | ~28GB | 1.5× slower |
| 4-bit quantization | ~16GB | Slower inference |

---

## 🛠️ Troubleshooting

### Common Issues

**1. `torch._dynamo.config.recompile_limit does not exist`**
- **Fixed in latest version** - This line has been removed from all scripts

**2. `ValueError: Mismatch in image token count`**
- **Cause:** `max_seq_length` too small for video frames
- **Solution:** Set `MAX_SEQ_LENGTH=50000` in bash script (already fixed)

**3. `PackageNotFoundError: deepspeed`**
- **Cause:** DeepSpeed not installed
- **Solution:** Already disabled by default. To use: `pip install deepspeed` and edit script

**4. `Video file not found`**
- **Cause:** Videos not in correct directory
- **Solution:** Ensure videos are in `videos/` or `workspace/gemma3-testing/videos/`

**5. CUDA Out of Memory**
- Reduce `batch_size` or `gradient_accumulation`
- Reduce `lora_r` (e.g., from 64 to 32)
- Enable 4-bit quantization: `LOAD_IN_4BIT="--load_in_4bit"`

**6. WandB Not Logging**
- Run `wandb login` and paste your API key
- Check internet connection
- Set `report_to="none"` in SFTConfig to disable

**7. Warning: "Gemma3nForConditionalGeneration does not accept `num_items_in_batch`"**
- **This is expected and not an error!**
- It's an informational message from Unsloth about gradient accumulation
- Training will work correctly, just slightly less accurate gradient accumulation
- You can safely ignore this warning
- The warning has been suppressed in the latest version

---

## 📁 Repository Structure

```
gemma3-testing/
├── gemma3_finetune_unsloth.py      # Main fine-tuning script
├── dataset.py                       # Dataset download/preparation
├── requirements.txt                 # Python dependencies
├── setup.sh                         # Environment setup
├── README.md                        # This file
│
├── scripts/
│   ├── finetune_gemma3n_unsloth.sh # Training wrapper script
│   ├── run_inference_unsloth.py    # Batch inference
│   └── zero*.json                  # DeepSpeed configs (optional)
│
├── eval/
│   └── eval_gemma3n.py             # Evaluation script
│
├── utils/
│   ├── infer_qved.py               # Single video inference
│   ├── plot_training_stats.py      # Visualization tools
│   └── ...                          # Other utilities
│
├── docs/
│   └── FINETUNE_GUIDE.md           # Comprehensive guide
│
├── dataset/                         # Created by dataset.py
│   ├── qved_train.json
│   ├── qved_val.json
│   └── qved_test.json
│
├── videos/                          # Downloaded videos
│   ├── 00000340.mp4
│   └── ...
│
├── outputs/                         # Training outputs
│   └── gemma3n_finetune_*/
│
└── results/                         # Evaluation results
    └── eval_*.json
```

---

## ✨ Key Features

- ✅ **Unsloth FastVisionModel:** 2× faster training with LoRA
- ✅ **Efficient Video Processing:** numpy.linspace frame extraction
- ✅ **Local Dataset Support:** No re-downloading during training
- ✅ **WandB Integration:** Real-time training metrics
- ✅ **Automatic Model Merging:** Ready-to-use merged models
- ✅ **Comprehensive Evaluation:** BLEU, ROUGE metrics
- ✅ **Production Ready:** Clean scripts, error handling
- ✅ **Well Documented:** Complete guides and examples

---

## 📚 Documentation

- **[FINETUNE_GUIDE.md](docs/FINETUNE_GUIDE.md)** - Complete fine-tuning guide
- **Model:** [unsloth/gemma-3n-E4B-it](https://huggingface.co/unsloth/gemma-3n-E4B-it)
- **Dataset:** [EdgeVLM-Labs/QVED-Test-Dataset](https://huggingface.co/datasets/EdgeVLM-Labs/QVED-Test-Dataset)
- **Unsloth:** [https://github.com/unslothai/unsloth](https://github.com/unslothai/unsloth)

---

## 🎓 Usage Examples

### Complete Workflow

```bash
# 1. Setup (one-time)
bash setup.sh && conda activate gemma3n
wandb login && huggingface-cli login

# 2. Prepare dataset
python dataset.py all --max-per-class 10

# 3. Fine-tune
bash scripts/finetune_gemma3n_unsloth.sh

# 4. Evaluate (50 samples)
python eval/eval_gemma3n.py \
    --model_path outputs/gemma3n_finetune_*_merged_16bit \
    --eval_json dataset/qved_val.json \
    --max_samples 50

# 5. Test inference
python utils/infer_qved.py \
    --model_path outputs/gemma3n_finetune_*_merged_16bit \
    --video_path videos/00000340.mp4
```

### Custom Dataset

```bash
# Prepare your own dataset JSON:
# [{"video": "path/to/video.mp4", "conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}]

python gemma3_finetune_unsloth.py \
    --train_json my_dataset/train.json \
    --val_json my_dataset/val.json \
    --output_dir outputs/my_model \
    --num_epochs 5
```

---

## 🚀 Performance

**Training Speed (A100 80GB):**
- Base model loading: ~30 seconds
- Training: ~15 samples/minute with batch=1, grad_accum=4
- 1 epoch on 420 samples: ~30 minutes
- Full 3 epochs: ~1.5 hours

**Memory Usage:**
- LoRA training: ~24GB VRAM
- Inference: ~16GB VRAM (merged model)
- Inference: ~8GB VRAM (4-bit quantized)

**Model Sizes:**
- LoRA adapter: ~600MB
- Merged 16-bit: ~16GB
- Merged 4-bit: ~4GB (optional)

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

See [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Unsloth AI:** For FastVisionModel and efficient training
- **Google:** For Gemma-3N models
- **EdgeVLM Labs:** For QVED dataset
- **HuggingFace:** For transformers and datasets

---

## 📞 Support

- **Issues:** [GitHub Issues](https://github.com/EdgeVLM-Labs/gemma3-testing/issues)
- **Discussions:** [GitHub Discussions](https://github.com/EdgeVLM-Labs/gemma3-testing/discussions)
- **Email:** Contact repository maintainers

---

**🎯 Ready to get started?** Run `bash setup.sh` and follow the [Quick Start](#-quick-start) guide!
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
