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
python core/finetune_gemma3n_e2b_trl.py \
  --train_json dataset/qved_train.json \
  --val_json dataset/qved_val.json \
  --video_path dataset \
  --output_dir outputs/gemma3n_finetune

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
│   ├── plot_training_stats.py        # Training visualization
│   ├── test_inference_transformers.py # Transformers-based batch inference
│   └── ...
├── unsloth/                          # Unsloth-based scripts (alternative)
├── archive/                          # Legacy/deprecated code
├── docs/                             # Documentation
├── dataset/                          # Video dataset (after download)
└── sample_videos/                    # Example videos for testing
```

---

## Setup

### Prerequisites
- Linux (Ubuntu 20.04+)
- NVIDIA GPU with CUDA support (16GB+ VRAM recommended)
- 50GB+ free disk space

### Install
```bash
bash setup.sh
```

This will:
1. Install system dependencies
2. Create conda environment `gemma3n` (Python 3.11)
3. Install PyTorch 2.5.1 with CUDA 12.1
4. Install all Python dependencies
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
  --video_path dataset \
  --output_dir outputs/gemma3n_finetune \
  --num_epochs 3 \
  --batch_size 4 \
  --learning_rate 2e-4 \
  --lora_r 64
```

### Key parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--train_json` | required | Path to training dataset JSON |
| `--val_json` | optional | Path to validation dataset JSON |
| `--video_path` | required | Base path for video files |
| `--output_dir` | required | Where to save checkpoints |
| `--num_epochs` | 3 | Number of training epochs |
| `--batch_size` | 8 | Per-device batch size |
| `--learning_rate` | 2e-4 | Learning rate |
| `--lora_r` | 64 | LoRA rank |
| `--num_frames` | 16 | Frames to extract per video |

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

- **Model:** Google Gemma-3N-E2B-it
- **Training:** Hugging Face Transformers 4.56.2 + TRL 0.22.2
- **Fine-tuning:** LoRA via PEFT
- **Video processing:** OpenCV + Pillow
- **Monitoring:** Weights & Biases
- **Evaluation:** ROUGE, BLEU, NLTK

---

## Troubleshooting

**Gated Model 403:**
1. Visit https://huggingface.co/google/gemma-3n-E2B
2. Request access and accept terms
3. Run `huggingface-cli login`

**CUDA Out of Memory:**
Reduce batch size or number of frames in the training script.

**Dependency issues:**
Re-run `bash setup.sh`

---

## Documentation

- [Quick Start Guide](docs/QUICKSTART.md)
- [RunPod Setup](docs/RUNPOD_QUICKSTART.md)
- [Setup Checklist](docs/SETUP_CHECKLIST.md)
- [Known Issues](docs/issues.md)
