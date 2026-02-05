# Gemma-3N Fine-tuning Guide

Complete guide for fine-tuning Gemma-3N model using Unsloth with your prepared QVED dataset.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Setup](#setup)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Inference](#inference)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

```bash
# 1. Setup environment (one time)
bash setup.sh
conda activate gemma3n

# 2. Initialize dataset (one time - interactive)
bash scripts/initialize_dataset.sh

# 3. Fine-tune model
bash scripts/finetune_gemma3n_unsloth.sh

# 4. Run inference
python scripts/run_inference_unsloth.py \
    --model_path outputs/gemma3n_finetune_YYYYMMDD_HHMMSS_merged_16bit \
    --video_path dataset/exercise_class_1/video_001.mp4
```

---

## 🔧 Setup

### 1. Install Dependencies

```bash
# Make setup script executable
chmod +x setup.sh

# Run setup (installs conda, packages, etc.)
bash setup.sh

# Activate environment
conda activate gemma3n
```

### 2. Login to Services

```bash
# WandB for experiment tracking (required)
wandb login

# HuggingFace (required for model download)
huggingface-cli login
```

### System Requirements

**Minimum:**
- GPU: NVIDIA GPU with 16GB VRAM (e.g., RTX 4090)
- RAM: 32GB
- CUDA: 11.8+
- Python: 3.11

**Recommended:**
- GPU: NVIDIA A100 (40GB) or H100
- RAM: 64GB
- CUDA: 12.1

---

## 📊 Dataset Preparation

### Using initialize_dataset.sh (Recommended)

The interactive initialization script orchestrates the complete dataset preparation pipeline:

```bash
bash scripts/initialize_dataset.sh
```

This will:
1. Download videos from HuggingFace Hub (prompts for count per class)
2. Optionally filter ground truth labels
3. Optionally augment videos
4. Generate train/val/test splits

**For manual control**, use individual scripts:

```bash
# Download videos
python utils/load_dataset.py 5  # 5 videos per class

# Filter ground truth
python utils/filter_ground_truth.py

# Augment videos (optional)
python utils/augment_videos.py

# Create train/val/test splits
python utils/qved_from_fine_labels.py
```

This creates:
- `dataset/qved_train.json` - Training data
- `dataset/qved_val.json` - Validation data
- `dataset/qved_test.json` - Test data
- `dataset/exercise_class_*/video_*.mp4` - Video files

### Dataset Format

The JSON files follow this structure:

```json
[
  {
    "video": "exercise_class_1/video_001.mp4",
    "conversations": [
      {
        "from": "human",
        "value": "Please evaluate the exercise form shown..."
      },
      {
        "from": "gpt",
        "value": "The form shows correct posture..."
      }
    ],
    "split": "train"
  }
]
```

---

## 🎯 Training

### Option 1: Quick Start (Bash Script)

Uses your prepared local dataset by default:

```bash
# Default: uses dataset/qved_train.json
bash scripts/finetune_gemma3n_unsloth.sh

# Or explicitly specify local mode
bash scripts/finetune_gemma3n_unsloth.sh --local
```

### Option 2: Direct Python Script

For more control over parameters:

```bash
python gemma3_finetune_unsloth.py \
    --train_json dataset/qved_train.json \
    --output_dir outputs/my_finetune \
    --num_epochs 3 \
    --batch_size 1 \
    --gradient_accumulation 4 \
    --learning_rate 2e-4 \
    --lora_r 64 \
    --lora_alpha 128 \
    --num_frames 8 \
    --wandb_project my-gemma3n-project
```

### Option 3: HuggingFace Streaming (Alternative)

If you want to download directly from HuggingFace during training:

```bash
bash scripts/finetune_gemma3n_unsloth.sh --hf

# Or with Python
python gemma3_finetune_unsloth.py \
    --hf_dataset EdgeVLM-Labs/QEVD-fine-grained-feedback-cleaned \
    --num_samples 500 \
    --batch_size 1 \
    --gradient_accumulation 4
```

### Monitoring Training

Training metrics are logged to Weights & Biases:

1. Open https://wandb.ai
2. Navigate to your project (default: `Finetune-gemma3n`)
3. View real-time metrics:
   - Training loss
   - Learning rate schedule
   - GPU memory usage
   - Training speed (samples/sec)

### Expected Training Time

Based on typical configurations:

| Configuration | GPU | Time (500 samples, 3 epochs) |
|--------------|-----|------------------------------|
| Default (batch=1, acc=4) | A100 40GB | ~30-45 min |
| Memory Optimized (4-bit) | RTX 4090 | ~45-60 min |
| Large batch (batch=2, acc=2) | A100 80GB | ~20-30 min |

---

## 🔮 Inference

### Single Video Inference

```bash
python scripts/run_inference_unsloth.py \
    --model_path outputs/gemma3n_finetune_YYYYMMDD_HHMMSS_merged_16bit \
    --video_path test_videos/video_001.mp4 \
    --num_frames 8
```

### Custom Instructions

```bash
python scripts/run_inference_unsloth.py \
    --model_path outputs/gemma3n_finetune_YYYYMMDD_HHMMSS_merged_16bit \
    --video_path test_videos/video_001.mp4 \
    --instruction "Evaluate this exercise and provide feedback" \
    --temperature 0.7 \
    --top_p 0.9
```

### Batch Inference

Process multiple videos:

```bash
# Run batch inference on sample videos
python gemma3n_batch_inference.py \
    --model_path outputs/gemma3n_finetune_YYYYMMDD_HHMMSS_merged_16bit \
    --video_folder sample_videos \
    --output results/batch_results.csv
```

### Programmatic Inference

```python
from unsloth import FastVisionModel
import cv2, numpy as np
from PIL import Image

# Load model
model, processor = FastVisionModel.from_pretrained(
    "outputs/gemma3n_finetune_YYYYMMDD_HHMMSS_merged_16bit",
    use_gradient_checkpointing="unsloth",
)
FastVisionModel.for_inference(model)

# Extract frames
def downsample_video(video_path, num_frames=8):
    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    for i in indices:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, image = vidcap.read()
        if success:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(image))
    vidcap.release()
    return frames

# Prepare input
frames = downsample_video("test.mp4")
instruction = "Evaluate this exercise form."
messages = [{"role": "user", "content": [{"type": "text", "text": instruction}]}]
for img in frames:
    messages[0]["content"].append({"type": "image", "image": img})

# Generate
input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(images=frames, text=input_text, add_special_tokens=False, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=128, temperature=1.0)
response = processor.decode(output[0], skip_special_tokens=True)
print(response)
```

---

## ⚙️ Configuration

### Key Hyperparameters

| Parameter | Default | Description | Recommended Range |
|-----------|---------|-------------|-------------------|
| `--batch_size` | 1 | Per-device batch size | 1-2 |
| `--gradient_accumulation` | 4 | Gradient accumulation steps | 2-8 |
| `--learning_rate` | 2e-4 | AdamW learning rate | 1e-4 to 5e-4 |
| `--num_epochs` | 3 | Training epochs | 2-5 |
| `--lora_r` | 64 | LoRA rank | 32-128 |
| `--lora_alpha` | 128 | LoRA alpha (usually 2×r) | 64-256 |
| `--num_frames` | 8 | Frames per video | 4-16 |
| `--max_seq_length` | 50000 | Max context length | 10000-50000 |
| `--warmup_ratio` | 0.03 | Warmup ratio (3% of steps) | 0.01-0.1 |
| `--max_grad_norm` | 0.3 | Gradient clipping | 0.1-1.0 |
| `--weight_decay` | 0.001 | L2 regularization | 0.0001-0.01 |

### Training Strategies

**Quick Test (Fast iteration)**
```bash
python gemma3_finetune_unsloth.py \
    --num_epochs 1 \
    --batch_size 2 \
    --gradient_accumulation 2 \
    --lora_r 32
```

**Production (Best quality)**
```bash
python gemma3_finetune_unsloth.py \
    --num_epochs 5 \
    --batch_size 1 \
    --gradient_accumulation 8 \
    --lora_r 128 \
    --lora_alpha 256
```

**Memory Constrained (< 24GB VRAM)**
```bash
python gemma3_finetune_unsloth.py \
    --load_in_4bit \
    --batch_size 1 \
    --gradient_accumulation 8 \
    --num_frames 4
```

### Output Files

After training, you'll find:

```
outputs/
├── gemma3n_finetune_YYYYMMDD_HHMMSS/
│   ├── adapter_config.json          # LoRA configuration
│   ├── adapter_model.safetensors    # LoRA weights (lightweight)
│   └── ...
│
└── gemma3n_finetune_YYYYMMDD_HHMMSS_merged_16bit/
    ├── config.json                  # Full model config
    ├── model.safetensors            # Complete merged weights
    └── ...                          # Ready for inference
```

**Use the `_merged_16bit` folder for inference** - it contains the full model with LoRA weights merged in.

---

## 🐛 Troubleshooting

### Dataset Issues

**Error: "Dataset not found"**
```bash
# Solution: Initialize the dataset first
bash scripts/initialize_dataset.sh

# Or manually prepare
python utils/load_dataset.py 5
python utils/qved_from_fine_labels.py

# Verify files exist
ls -la dataset/qved_train.json
```

**Error: "Video file not found"**
```bash
# Check video paths in JSON (should be relative)
head -n 20 dataset/qved_train.json

# Verify videos exist
ls -la dataset/exercise_class_*/
```

### Memory Issues

**Error: "CUDA out of memory"**
```bash
# Solution 1: Enable 4-bit quantization
python gemma3_finetune_unsloth.py --load_in_4bit ...

# Solution 2: Reduce batch size
python gemma3_finetune_unsloth.py --batch_size 1 --gradient_accumulation 8 ...

# Solution 3: Reduce frames
python gemma3_finetune_unsloth.py --num_frames 4 ...

# Solution 4: Combination
python gemma3_finetune_unsloth.py \
    --load_in_4bit \
    --batch_size 1 \
    --gradient_accumulation 16 \
    --num_frames 4
```

### Performance Issues

**Training is slow**
```bash
# Check GPU utilization
watch -n 1 nvidia-smi

# Verify CUDA is available
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check if gradient checkpointing is enabled (it should be)
# Look for: use_gradient_checkpointing="unsloth" in the script
```

**Low GPU utilization**
```bash
# Increase batch size if you have memory
python gemma3_finetune_unsloth.py --batch_size 2 ...

# Reduce gradient accumulation to match
python gemma3_finetune_unsloth.py --batch_size 2 --gradient_accumulation 2 ...
```

### Installation Issues

**Error: "No module named 'torch'"**
```bash
# Ensure you activated the environment
conda activate gemma3n

# Reinstall if needed
bash setup.sh
```

**Error: "No module named 'unsloth'"**
```bash
pip install --no-deps --upgrade timm
pip install --upgrade unsloth unsloth_zoo
```

**Error: "ImportError: FastVisionModel"**
```bash
# Update to latest Unsloth
pip install --upgrade unsloth unsloth_zoo
pip install transformers==4.56.2
pip install trl==0.22.2
```

### Model Loading Issues

**Error: "Model not found"**
```bash
# Make sure you're using the merged model path
ls -la outputs/gemma3n_finetune_*/

# Use the _merged_16bit directory
python scripts/run_inference_unsloth.py \
    --model_path outputs/gemma3n_finetune_YYYYMMDD_HHMMSS_merged_16bit \
    ...
```

---

## 📚 Additional Resources

### File Structure
```
gemma3-testing/
├── dataset/                         # Your prepared dataset
│   ├── exercise_class_1/           # Videos by class
│   ├── qved_train.json             # Training data
│   ├── qved_val.json               # Validation data
│   └── qved_test.json              # Test data
├── gemma3_finetune_unsloth.py      # Main training script
├── setup.sh                         # Environment setup
├── requirements.txt                 # Dependencies
├── scripts/
│   ├── initialize_dataset.sh       # Dataset preparation
│   ├── finetune_gemma3n_unsloth.sh # Training wrapper
│   └── run_inference_unsloth.py    # Inference script
├── utils/
│   ├── load_dataset.py             # Download videos
│   ├── qved_from_fine_labels.py    # Create splits
│   └── augment_videos.py           # Data augmentation
└── docs/
    └── FINETUNE_GUIDE.md           # This file
```

### Key Features

✅ **Efficient LoRA fine-tuning** with Unsloth FastVisionModel  
✅ **Local dataset support** - uses your prepared data  
✅ **Automatic frame extraction** from videos  
✅ **4-bit quantization** for limited GPU memory  
✅ **Gradient checkpointing** for memory optimization  
✅ **WandB integration** for experiment tracking  
✅ **Streaming support** for HuggingFace datasets (optional)  

### Related Documentation

- [QVED Dataset Repository](https://huggingface.co/datasets/EdgeVLM-Labs/QEVD-fine-grained-feedback-cleaned)
- [Unsloth Documentation](https://docs.unsloth.ai/)
- [Gemma Model Card](https://huggingface.co/google/gemma-3n-E2B-it)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

---

## 🎯 Tips & Best Practices

### For Best Results

1. **Start small** - Test with 1-2 epochs first
2. **Monitor training** - Watch loss curve in WandB
3. **Validate regularly** - Use validation set to check for overfitting
4. **Experiment** - Try different learning rates and LoRA ranks
5. **Augment data** - Use augmentation to increase dataset size

### For Production Deployment

1. **Use merged model** - The `_merged_16bit` folder is ready for deployment
2. **Test thoroughly** - Validate on held-out test set
3. **Monitor performance** - Track inference speed and accuracy
4. **Version control** - Save different checkpoints for comparison
5. **Document changes** - Keep notes on hyperparameters used

### Common Workflows

**Experiment Iteration:**
```bash
# Try different configurations quickly
for lr in 1e-4 2e-4 5e-4; do
    python gemma3_finetune_unsloth.py \
        --learning_rate $lr \
        --wandb_run_name "lr_${lr}" \
        --output_dir "outputs/exp_lr_${lr}"
done
```

**Resume Training:**
```bash
# If training was interrupted, start from checkpoint
python gemma3_finetune_unsloth.py \
    --train_json dataset/qved_train.json \
    --output_dir outputs/gemma3n_finetune_previous \
    --resume_from_checkpoint outputs/gemma3n_finetune_previous/checkpoint-100
```

---

**Happy fine-tuning! 🚀**

For issues or questions, check the troubleshooting section above or open an issue on GitHub.
