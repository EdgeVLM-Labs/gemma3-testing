# Gemma-3N Fine-tuning for Video Understanding

Fine-tune Google's **Gemma-3N (google/gemma-3n-E2B-it)** model on video datasets using **Hugging Face Transformers + TRL**. Optimized for exercise form analysis and physiotherapy video understanding tasks.

**Model:** `google/gemma-3n-E2B-it` | **Dataset:** QVED | **Framework:** Transformers + TRL

---

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/EdgeVLM-Labs/gemma3-testing.git
cd gemma3-testing
bash setup.sh

# 2. Activate environment
conda activate gemma3n

# 3. Initialize dataset
bash scripts/initialize_dataset.sh

# 4. Fine-tune
bash scripts/finetune_gemma3n_e2b_trl.sh

# 5. Run inference on test set
bash scripts/run_inference_transformers.sh \
  --test_json dataset/qved_test.json \
  --data_path dataset
```

---

## Project Structure

```
gemma3-testing/
├── setup.sh                          # Environment setup
├── requirements.txt                  # Python dependencies
├── core/
│   ├── finetune_gemma3n_e2b_trl.py   # Main fine-tuning script
│   └── inference.py                  # Main inference script
├── scripts/
│   ├── finetune_gemma3n_e2b_trl.sh   # Fine-tuning shell wrapper
│   ├── run_inference_transformers.sh  # Inference + evaluation
│   ├── initialize_dataset.sh         # Dataset download & prep
│   ├── verify_qved_setup.sh          # Setup verification
│   ├── quickstart_finetune.sh        # Quick-start workflow
│   └── zero*.json                    # DeepSpeed configs
├── utils/
│   ├── hf_upload.py                  # Upload model to HuggingFace
│   ├── load_dataset.py               # Download QVED dataset
│   ├── filter_ground_truth.py        # Filter labels to downloaded videos
│   ├── generate_test_report.py       # Evaluation report generation
│   └── plot_training_stats.py        # Training visualization
├── unsloth/                          # Unsloth-based alternative pipeline
├── archive/                          # Legacy/deprecated code
├── docs/                             # Documentation
├── dataset/                          # Video dataset (after download)
└── sample_videos/                    # Example videos for testing
```

---

## Setup

### Prerequisites
- Linux (Ubuntu 20.04+)
- NVIDIA GPU with CUDA support (A40 48GB recommended)
- 50GB+ free disk space

### Install
```bash
bash setup.sh
```

This will:
1. Install system dependencies
2. Create conda environment `gemma3n` (Python 3.11)
3. Install PyTorch 2.5.1 with CUDA 12.1
4. Install all Python dependencies (transformers, trl, peft, timm, bitsandbytes, etc.)
5. Prompt for HuggingFace and WandB authentication

---

## Fine-tuning

### Using the shell wrapper (recommended)
```bash
bash scripts/finetune_gemma3n_e2b_trl.sh
```

### Using Python directly
```bash
python core/finetune_gemma3n_e2b_trl.py \
  --train_json dataset/qved_train.json \
  --val_json dataset/qved_val.json \
  --data_path dataset \
  --output_dir outputs/gemma3n_finetune \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 32 \
  --learning_rate 2e-4 \
  --lora_r 64 \
  --num_frames 8
```

### Hyperparameters (A40 48GB)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--train_json` | `dataset/qved_train.json` | Path to training dataset JSON |
| `--val_json` | `dataset/qved_val.json` | Path to validation dataset JSON |
| `--data_path` | `dataset` | Base path for video files |
| `--output_dir` | auto-generated | Where to save checkpoints |
| `--num_train_epochs` | 3 | Number of training epochs |
| `--per_device_train_batch_size` | 1 | Per-device batch size |
| `--gradient_accumulation_steps` | 32 | Gradient accumulation (effective batch = 32) |
| `--learning_rate` | 2e-4 | Learning rate |
| `--lora_r` | 64 | LoRA rank |
| `--lora_alpha` | 128 | LoRA alpha |
| `--num_frames` | 8 | Frames to extract per video |
| `--max_seq_length` | 1024 | Maximum sequence length |
| `--warmup_ratio` | 0.05 | Warmup ratio |
| `--dataloader_num_workers` | 2 | Dataloader workers |

### Override defaults via environment variables
```bash
BATCH_SIZE=2 GRAD_ACCUM=16 NUM_FRAMES=16 bash scripts/finetune_gemma3n_e2b_trl.sh
```

---

## Inference

### Batch inference with evaluation
```bash
bash scripts/run_inference_transformers.sh \
  --test_json dataset/qved_test.json \
  --data_path dataset
```

### Upload model to HuggingFace
```bash
python utils/hf_upload.py \
  --model_path outputs/gemma3n_finetune/checkpoint-70 \
  --repo_name my-gemma3n-model \
  --org your-org
```

---

## Dataset

The project uses the **QVED** (Qualitative Exercise Video Dataset) with 10 exercise categories:
- alternating_forward_lunges, floor_touches, high_knees, jumping_jacks, mountain_climbers
- plank_taps, pushups, shoulder_gators, squats, toe_touch

### Download and prepare
```bash
bash scripts/initialize_dataset.sh
```

---

## Tech Stack

- **Model:** Google Gemma-3N-E2B-it (~2B effective parameters)
- **Training:** Hugging Face Transformers 4.56.2 + TRL 0.22.2
- **Fine-tuning:** LoRA via PEFT (r=64, alpha=128)
- **Optimizer:** Paged AdamW 8-bit (via bitsandbytes)
- **Vision encoder:** timm (MobileNetV5 via TimmWrapper)
- **Video processing:** OpenCV + Pillow
- **Monitoring:** Weights & Biases
- **Evaluation:** ROUGE, BLEU, NLTK

---

## Troubleshooting

**Gated Model 403:**
1. Visit https://huggingface.co/google/gemma-3n-E2B
2. Request access and accept terms
3. Run `huggingface-cli login`

**CUDA Out of Memory (A40 48GB):**
- Use `batch_size=1` with `grad_accum=32` (default)
- Use `num_frames=8` (default)
- Use `max_seq_length=1024` (default)
- If still OOM, try reducing `lora_r` from 64 to 32

**Dependency issues:**
Re-run `bash setup.sh`

---

## Documentation

- [Quick Start Guide](docs/QUICKSTART.md)
- [RunPod Setup](docs/RUNPOD_QUICKSTART.md)
- [Setup Checklist](docs/SETUP_CHECKLIST.md)
- [Known Issues](docs/issues.md)
