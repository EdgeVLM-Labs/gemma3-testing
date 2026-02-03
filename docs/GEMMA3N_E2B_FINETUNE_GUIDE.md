# Gemma-3n-E2B Fine-tuning Guide for QVED

Complete guide for fine-tuning Gemma-3n-E2B-it model on the QVED physiotherapy exercise dataset using native Transformers and TRL.

## 📋 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Inference](#inference)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Performance](#performance)

---

## 🎯 Overview

This guide covers fine-tuning the `google/gemma-3n-E2B-it` vision-language model for physiotherapy exercise feedback using:
- **Base Model**: Google's Gemma-3n-E2B-it (2B parameters)
- **Training Method**: LoRA (Low-Rank Adaptation) with TRL
- **Framework**: Native PyTorch Transformers
- **Dataset**: QVED physiotherapy exercise videos

### Key Features

✅ **Efficient LoRA fine-tuning** - Train only 0.5% of parameters  
✅ **Video frame extraction** - Automatic 8-frame sampling  
✅ **Gradient checkpointing** - Fits on 24GB VRAM  
✅ **Mixed precision training** - BF16/FP16 automatic selection  
✅ **Weights & Biases integration** - Real-time monitoring  
✅ **Checkpoint management** - Auto-save best models  

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare QVED dataset
python dataset.py download --max-per-class 5
python dataset.py prepare

# 3. Start fine-tuning
bash scripts/finetune_gemma3n_e2b_trl.sh

# 4. Run inference on test videos
bash scripts/run_inference_transformers.sh \
  --hf_repo outputs/gemma3n-e2b-qved-ft \
  --test_json dataset/qved_test.json \
  --data_path dataset \
  --limit 10
```

---

## 📦 Installation

### System Requirements

**Minimum:**
- GPU: NVIDIA GPU with 24GB VRAM (e.g., RTX 3090, RTX 4090)
- RAM: 32GB
- CUDA: 11.8+
- Python: 3.10+

**Recommended:**
- GPU: NVIDIA A100 (40GB) or H100
- RAM: 64GB+
- CUDA: 12.1+

### Install Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install transformers and training libraries
pip install transformers>=4.49.0
pip install trl==0.22.2
pip install peft
pip install accelerate
pip install deepspeed

# Install video processing libraries
pip install opencv-python
pip install Pillow
pip install timm

# Optional: Weights & Biases for experiment tracking
pip install wandb
wandb login
```

Or install all at once:

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset Preparation

### Dataset Structure

```
dataset/
├── qved_train.json              # Training annotations
├── qved_val.json                # Validation annotations  
├── qved_test.json               # Test annotations
├── exercise_class_1/            # Exercise videos
│   ├── video_001.mp4
│   └── ...
└── exercise_class_2/
    └── ...
```

### Dataset Format

JSON files follow this structure:

```json
[
  {
    "video": "exercise_class_1/video_001.mp4",
    "conversations": [
      {
        "from": "human",
        "value": "Analyze this exercise video and provide feedback."
      },
      {
        "from": "gpt",
        "value": "Squat - Good form with proper knee alignment and depth."
      }
    ]
  }
]
```

### Prepare Your Dataset

```bash
# Download QVED dataset from HuggingFace
python dataset.py download --max-per-class 5

# Create train/val/test splits
python dataset.py prepare

# Verify dataset integrity
python dataset.py verify
```

---

## 🎯 Training

### Training Files

1. **`finetune_gemma3n_e2b_trl.py`** - Main Python training script
2. **`scripts/finetune_gemma3n_e2b_trl.sh`** - Bash wrapper with validation

### Option 1: Using Bash Script (Recommended)

```bash
bash scripts/finetune_gemma3n_e2b_trl.sh
```

This script:
- Validates dataset files exist
- Sets optimal hyperparameters
- Enables Weights & Biases tracking
- Handles gradient checkpointing

### Option 2: Using Python Script Directly

```bash
python finetune_gemma3n_e2b_trl.py \
    --train_json dataset/qved_train.json \
    --val_json dataset/qved_val.json \
    --data_path dataset/ \
    --output_dir ./outputs/gemma3n-e2b-qved-ft \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --learning_rate 2e-4
```

### Custom Configuration

#### Using Environment Variables

```bash
# Dataset paths
TRAIN_JSON=dataset/custom_train.json \
VAL_JSON=dataset/custom_val.json \
VIDEO_PATH=dataset/videos \
\
# Hyperparameters
LEARNING_RATE=3e-4 \
LORA_R=128 \
EPOCHS=5 \
BATCH_SIZE=4 \
\
# Experiment tracking
WANDB_PROJECT="gemma3n-experiments" \
RUN_NAME="experiment-1" \
\
bash scripts/finetune_gemma3n_e2b_trl.sh
```

#### Using Command Line Arguments

```bash
python finetune_gemma3n_e2b_trl.py \
    --train_json dataset/qved_train.json \
    --val_json dataset/qved_val.json \
    --data_path dataset/ \
    --output_dir ./outputs/custom-run \
    --num_train_epochs 5 \
    --learning_rate 3e-4 \
    --lora_r 128 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --wandb_project "my-project" \
    --run_name "custom-run"
```

### Monitoring Training

Training metrics are automatically logged to Weights & Biases:

1. Visit https://wandb.ai
2. Navigate to your project (default: `gemma3n-e2b-qved-finetune`)
3. View real-time metrics:
   - Training/validation loss
   - Learning rate schedule
   - GPU memory usage
   - Training throughput (samples/sec)

### Expected Training Time

| Configuration | GPU | Samples | Time |
|--------------|-----|---------|------|
| Default (batch=8, acc=4) | A100 40GB | 100 | ~15 min |
| Default (batch=8, acc=4) | A100 40GB | 1000 | ~2 hours |
| Memory Optimized (batch=4, acc=8) | RTX 4090 24GB | 100 | ~20 min |
| Large Batch (batch=16, acc=2) | A100 80GB | 1000 | ~1.5 hours |

---

## 🔮 Inference

### Single Video Inference

```bash
python utils/test_inference_transformers.py \
    --model_path outputs/gemma3n-e2b-qved-ft \
    --test_json dataset/qved_test.json \
    --data_path dataset/ \
    --output predictions.json \
    --limit 1
```

### Batch Inference with Evaluation

```bash
# Run inference on all test videos
bash scripts/run_inference_transformers.sh \
  --hf_repo outputs/gemma3n-e2b-qved-ft \
  --test_json dataset/qved_test.json \
  --data_path dataset
```

This automatically:
1. Runs inference on all test videos
2. Generates predictions JSON
3. Creates Excel evaluation report with:
   - BERT similarity scores
   - METEOR scores
   - ROUGE-L scores
   - Exercise identification accuracy

### Using Your Fine-tuned Model

```python
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
import torch
from PIL import Image
import cv2

# Load model
model = Gemma3nForConditionalGeneration.from_pretrained(
    "outputs/gemma3n-e2b-qved-ft",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("outputs/gemma3n-e2b-qved-ft")

# Extract frames from video
def extract_frames(video_path, num_frames=8):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // num_frames)
    
    frames = []
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames

# Prepare input
frames = extract_frames("test_video.mp4")
messages = [{
    "role": "user",
    "content": [{"type": "text", "text": "Analyze this exercise."}] + 
               [{"type": "image", "image": img} for img in frames]
}]

# Generate response
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, 
    tokenize=True, return_tensors="pt"
).to(model.device)

with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=256)
    
response = processor.decode(outputs[0], skip_special_tokens=True)
print(response)
```

---

## ⚙️ Configuration

#### Hyperparameters

Default configuration optimized for QVED dataset:

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

### Training Strategies

**Quick Test (Fast iteration)**
```bash
python finetune_gemma3n_e2b_trl.py \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --lora_r 32
```

**Production (Best quality)**
```bash
python finetune_gemma3n_e2b_trl.py \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --lora_r 128 \
    --lora_alpha 256
```

**Memory Constrained (< 24GB VRAM)**
```bash
python finetune_gemma3n_e2b_trl.py \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --num_frames 4 \
    --max_seq_length 1024
```

### Output Structure

After training completes:

```
outputs/gemma3n-e2b-qved-ft/
├── config.json                    # Model configuration
├── model.safetensors             # Full model weights (LoRA merged)
├── adapter_config.json           # LoRA adapter config
├── adapter_model.safetensors     # LoRA adapter weights
├── preprocessor_config.json      # Processor config
├── training_args.json            # Training arguments
├── trainer_state.json            # Training state
├── checkpoint-30/                # Checkpoint at step 30
├── checkpoint-60/                # Checkpoint at step 60
└── ...
```

---

## 🐛 Troubleshooting

### CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:

1. **Reduce batch size**:
   ```bash
   --per_device_train_batch_size 4
   ```

2. **Increase gradient accumulation**:
   ```bash
   --gradient_accumulation_steps 8
   ```

3. **Reduce sequence length**:
   ```bash
   --max_seq_length 1024
   ```

4. **Reduce number of frames**:
   ```bash
   --num_frames 4
   ```

5. **Use smaller LoRA rank**:
   ```bash
   --lora_r 32 --lora_alpha 64
   ```

### Video Loading Errors

**Error**: `Could not open video file`

**Solutions**:
- Verify video files exist: `ls -la dataset/exercise_*/`
- Check video paths in JSON match actual files
- Ensure videos are not corrupted: `ffprobe video.mp4`
- Install required codecs: `pip install opencv-python-headless`

### Dataset Not Found

**Error**: `FileNotFoundError: dataset/qved_train.json`

**Solutions**:
```bash
# Verify files exist
ls -la dataset/qved*.json

# Regenerate dataset
python dataset.py prepare

# Check JSON format
python -c "import json; json.load(open('dataset/qved_train.json'))"
```

### Slow Training Speed

**Issue**: Training is taking too long

**Solutions**:
1. **Check GPU utilization**:
   ```bash
   watch -n 1 nvidia-smi
   ```

2. **Increase dataloader workers**:
   ```bash
   --dataloader_num_workers 4
   ```

3. **Use larger batch size** (if memory allows):
   ```bash
   --per_device_train_batch_size 16
   ```

4. **Enable gradient checkpointing** (if not already):
   ```bash
   --gradient_checkpointing
   ```

### Wandb Connection Issues

**Error**: `wandb: ERROR Unable to connect`

**Solutions**:
```bash
# Disable wandb
export WANDB_MODE=disabled

# Or use offline mode
export WANDB_MODE=offline

# Or disable in script
python finetune_gemma3n_e2b_trl.py --no_wandb ...
```

### Model Download Issues

**Error**: `Cannot download google/gemma-3n-E2B-it`

**Solutions**:
```bash
# Login to HuggingFace
huggingface-cli login

# Pre-download model
huggingface-cli download google/gemma-3n-E2B-it

# Or set token
export HF_TOKEN=your_token_here
```

---

## 📈 Performance

### Training Metrics

Expected metrics for QVED dataset (100-200 samples):

| Metric | Initial | After 1 Epoch | After 3 Epochs |
|--------|---------|---------------|----------------|
| Training Loss | 2.5-3.0 | 1.0-1.5 | 0.5-1.0 |
| Validation Loss | 2.5-3.0 | 1.2-1.7 | 0.8-1.3 |
| Learning Rate | 2e-4 | → | 0 (with decay) |

### Evaluation Metrics

After fine-tuning on QVED:
### Evaluation Metrics

After fine-tuning on QVED:

| Metric | Expected Range | Description |
|--------|---------------|-------------|
| BERT Similarity | 0.65-0.85 | Semantic similarity to ground truth |
| METEOR Score | 0.45-0.65 | Machine translation quality |
| ROUGE-L Score | 0.40-0.60 | Longest common subsequence |
| Exercise Accuracy | 85-95% | Correct exercise identification |

### Memory Requirements
- **Minimum**: 24GB VRAM (batch_size=2, grad_accum=16)
- **Recommended**: 40GB+ VRAM (batch_size=8, grad_accum=4)
- **Optimal**: 80GB VRAM (batch_size=16, grad_accum=2)

---

## 📚 Additional Resources

### Project Structure

```
gemma3-testing/
├── dataset/                           # QVED dataset
│   ├── exercise_class_*/             # Video files
│   ├── qved_train.json               # Training annotations
│   ├── qved_val.json                 # Validation annotations
│   └── qved_test.json                # Test annotations
├── finetune_gemma3n_e2b_trl.py       # Training script
├── dataset.py                         # Dataset preparation
├── requirements.txt                   # Python dependencies
├── scripts/
│   ├── finetune_gemma3n_e2b_trl.sh   # Training wrapper
│   └── run_inference_transformers.sh # Inference + evaluation
├── utils/
│   ├── test_inference_transformers.py # Inference script
│   └── generate_test_report.py       # Evaluation report
└── docs/
    └── GEMMA3N_E2B_FINETUNE_GUIDE.md # This guide
```

### Related Documentation

- [Gemma-3 Model Card](https://huggingface.co/google/gemma-3n-E2B-it)
- [TRL Documentation](https://huggingface.co/docs/trl/)
- [PEFT/LoRA Documentation](https://huggingface.co/docs/peft/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [QVED Dataset](https://huggingface.co/datasets/EdgeVLM-Labs/QVED-Test-Dataset)

### Training Tips

1. **Start with small learning rate** - 2e-4 is a good default
2. **Monitor validation loss** - Stop if it starts increasing
3. **Use gradient checkpointing** - Essential for limited memory
4. **Save checkpoints frequently** - Every 30-50 steps
5. **Log to Wandb** - Track experiments systematically

### Deployment

After successful training:

1. **Test thoroughly** on held-out test set
2. **Generate evaluation report** with metrics
3. **Upload to HuggingFace Hub**:
   ```bash
   python utils/hf_upload.py \
     --model_path outputs/gemma3n-e2b-qved-ft \
     --repo_name your-username/gemma3n-qved-ft
   ```
4. **Deploy for inference** in production environment

---

## 🎓 Next Steps

1. ✅ **Fine-tune** on your QVED dataset
2. ✅ **Evaluate** using test set
3. ✅ **Generate report** with metrics
4. ✅ **Iterate** on hyperparameters if needed
5. ✅ **Deploy** your fine-tuned model

---

**Questions or issues?** Open an issue on GitHub or check the troubleshooting section above.

**Happy fine-tuning! 🚀**
