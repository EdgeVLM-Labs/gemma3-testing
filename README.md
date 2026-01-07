# Gemma-3N-E2B Video Understanding & Fine-tuning

This repository provides tools for fine-tuning and inference with **Google Gemma-3N-E2B** models for video understanding tasks, with a focus on exercise form analysis and assistive navigation for visually impaired users.

## 🚀 Features

- **Fine-tuning** on QVED (Quality Video Exercise Dataset)
- **Batch video inference** for multiple videos
- **Single video inference** for quick testing
- **Dataset preparation** and quality control
- **Training visualization** and evaluation reports
- **Model uploading** to Hugging Face Hub

---

## 📁 Repository Structure

```
gemma3-testing/
├── dataset.py            # 🆕 Unified dataset management (download/prepare/clean)
├── gemma3n_batch_inference.py  # Batch inference (Unsloth FastModel)
├── setup.sh              # Environment setup script
├── requirements.txt      # Python dependencies
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
├── mobilevideogpt/       # Model architecture & utilities
├── eval/                 # Evaluation scripts
└── README.md             # This file
```

---

## 🛠️ Installation

### 1. Environment Setup

```bash
# Run the complete setup (installs Miniconda, creates env, installs dependencies)
bash setup.sh
```

Or manually:

```bash
# Create conda environment
conda create -n gemma3n python=3.11 -y
conda activate gemma3n

# Install dependencies
pip install -r requirements.txt
```

### 2. Accept Conda Terms of Service

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

### 3. Login to Hugging Face

```bash
hf auth login
# Or set token:
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

### Out of Disk Space

```bash
# Clean HF cache
rm -rf ~/.cache/huggingface/hub/*

# Check disk usage
df -h
```

### Gated Model Access (403 Error)

```bash
# 1. Request access at https://huggingface.co/google/gemma-3n-E2B
# 2. Accept terms
# 3. Login
hf auth login

# Or use public mirror
--model unsloth/gemma-3n-E2B
```

### PEFT Import Error

```bash
# Upgrade PEFT
pip install --upgrade peft
```

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
