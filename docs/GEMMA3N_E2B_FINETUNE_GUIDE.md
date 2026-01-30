# Gemma-3n-E2B-it Fine-tuning Guide for QVED

## Overview

This guide covers fine-tuning the `google/gemma-3n-E2B-it` model on the QVED physiotherapy exercise dataset using TRL (Transformers Reinforcement Learning).

## Files Created

1. **`finetune_gemma3n_e2b_trl.py`** - Main Python training script
2. **`scripts/finetune_gemma3n_e2b_trl.sh`** - Bash wrapper script with validation

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `torch>=2.5.1`
- `transformers>=4.49.0` (for Gemma-3 vision support)
- `trl==0.22.2`
- `peft`
- `timm`
- `opencv-python`
- `Pillow`
- `wandb` (optional, for experiment tracking)

### 2. Prepare Dataset

Your dataset should be in JSON format:

```json
[
  {
    "video": "path/to/video.mp4",
    "conversations": [
      {"from": "human", "value": "What exercise is being performed?"},
      {"from": "gpt", "value": "Squat - Good form with proper knee alignment"}
    ]
  }
]
```

### 3. Run Training

#### Option A: Using the bash script (recommended)

```bash
bash scripts/finetune_gemma3n_e2b_trl.sh
```

#### Option B: Using Python directly

```bash
python finetune_gemma3n_e2b_trl.py \
    --train_json data/qved_train.json \
    --val_json data/qved_val.json \
    --data_path videos/ \
    --output_dir ./outputs/gemma3n-e2b-qved-ft
```

## Hyperparameters

The default hyperparameters match your specifications:

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `num_frames` | 8 | Frames extracted per video |
| `num_train_epochs` | 3 | Training epochs |
| `learning_rate` | 2e-4 | Learning rate |
| `per_device_train_batch_size` | 8 | Batch size per device |
| `gradient_accumulation_steps` | 4 | Gradient accumulation |
| `max_seq_length` | 2048 | Maximum sequence length |
| `lora_r` | 64 | LoRA attention dimension |
| `lora_alpha` | 128 | LoRA alpha parameter |
| `lora_dropout` | 0.05 | LoRA dropout |
| `warmup_ratio` | 0.05 | Warmup ratio |
| `save_steps` | 30 | Save checkpoint every N steps |
| `eval_strategy` | "steps" | Evaluation strategy |
| `dataloader_num_workers` | 2 | Dataloader workers |
| `gradient_checkpointing` | True | Enable gradient checkpointing |

## Customization

### Environment Variables (for bash script)

```bash
# Dataset paths
TRAIN_JSON=data/custom_train.json \
VAL_JSON=data/custom_val.json \
VIDEO_PATH=data/videos \
\
# Hyperparameters
LEARNING_RATE=3e-4 \
LORA_R=128 \
EPOCHS=5 \
BATCH_SIZE=4 \
\
# Wandb
WANDB_PROJECT="my-project" \
RUN_NAME="custom-run" \
\
bash scripts/finetune_gemma3n_e2b_trl.sh
```

### Command Line Arguments (for Python script)

```bash
python finetune_gemma3n_e2b_trl.py \
    --train_json data/qved_train.json \
    --val_json data/qved_val.json \
    --data_path videos/ \
    --output_dir ./outputs/custom-run \
    --num_train_epochs 5 \
    --learning_rate 3e-4 \
    --lora_r 128 \
    --per_device_train_batch_size 4 \
    --wandb_project "my-project" \
    --run_name "custom-run"
```

## Video Frame Extraction

The script automatically:
1. Extracts 8 evenly-spaced frames from each video
2. Resizes frames to 224x224 (matching Gemma-3n encoder input)
3. Feeds all frames to the model simultaneously

This matches the inference pipeline in `utils/test_inference_transformers.py`.

## Wandb Integration

### Setup Wandb

```bash
# Login to wandb (first time only)
wandb login

# Or set API key
export WANDB_API_KEY=your_api_key
```

### Enable Tracking

Wandb is enabled by default. To disable:

```bash
# Using bash script
WANDB_MODE=disabled bash scripts/finetune_gemma3n_e2b_trl.sh

# Using Python script
python finetune_gemma3n_e2b_trl.py --no_wandb ...
```

### Configure Project and Run Name

```bash
python finetune_gemma3n_e2b_trl.py \
    --wandb_project "gemma3n-qved-experiments" \
    --run_name "gemma3n-e2b-lr2e4-r64-epoch3" \
    ...
```

## Training Features

### Gradient Checkpointing
Enabled by default to reduce memory usage. Critical for training with limited VRAM.

### Mixed Precision Training
- Uses `bfloat16` if supported by GPU
- Falls back to `float16` for older GPUs
- Automatic detection and configuration

### LoRA (Low-Rank Adaptation)
- Efficient fine-tuning with minimal parameters
- Targets all linear layers (`target_modules="all-linear"`)
- Saves storage and speeds up training

### Checkpoint Management
- Saves checkpoint every 30 steps (configurable)
- Keeps only last 3 checkpoints to save disk space
- Auto-saves best model based on validation loss

## Resume Training

To resume from a checkpoint:

```bash
python finetune_gemma3n_e2b_trl.py \
    --resume_from_checkpoint ./outputs/gemma3n-e2b-qved-ft/checkpoint-30 \
    ...
```

## Evaluation

After training, evaluate on test set:

```bash
python utils/test_inference_transformers.py \
    --model_path ./outputs/gemma3n-e2b-qved-ft \
    --test_json data/qved_test.json \
    --data_path videos/ \
    --output results/predictions.json
```

## Troubleshooting

### CUDA Out of Memory

1. Reduce batch size:
   ```bash
   --per_device_train_batch_size 4
   ```

2. Increase gradient accumulation:
   ```bash
   --gradient_accumulation_steps 8
   ```

3. Reduce sequence length:
   ```bash
   --max_seq_length 1024
   ```

4. Reduce number of frames:
   ```bash
   --num_frames 4
   ```

### Video Loading Errors

- Ensure video files exist at specified paths
- Check video codec compatibility with OpenCV
- Verify video files are not corrupted

### Wandb Connection Issues

```bash
# Disable wandb
export WANDB_MODE=disabled

# Or use offline mode
export WANDB_MODE=offline
```

## Output Structure

After training completes:

```
outputs/gemma3n-e2b-qved-ft/
├── config.json                 # Model configuration
├── model.safetensors          # Model weights
├── adapter_config.json        # LoRA adapter config
├── adapter_model.safetensors  # LoRA adapter weights
├── training_args.json         # Training arguments used
├── config.txt                 # Human-readable config
├── checkpoint-30/             # Checkpoint at step 30
├── checkpoint-60/             # Checkpoint at step 60
└── ...
```

## Performance Expectations

### Training Time (approximate)
- **100 samples**: ~10-15 minutes on A100
- **1000 samples**: ~1.5-2 hours on A100
- **10000 samples**: ~15-20 hours on A100

### Memory Requirements
- **Minimum**: 24GB VRAM (batch_size=1, grad_accum=16)
- **Recommended**: 40GB+ VRAM (batch_size=8, grad_accum=4)
- **Optimal**: 80GB VRAM (batch_size=16, grad_accum=2)

## Next Steps

1. **Fine-tune the model** on your QVED dataset
2. **Evaluate** on test set using `test_inference_transformers.py`
3. **Generate report** using `utils/generate_test_report.py`
4. **Upload to HuggingFace Hub** using `utils/hf_upload.py`
5. **Deploy** for inference in production

## Additional Resources

- [Gemma-3 Documentation](https://huggingface.co/google/gemma-3n-E2B-it)
- [TRL Documentation](https://huggingface.co/docs/trl/)
- [PEFT Documentation](https://huggingface.co/docs/peft/)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
