# Gemma 3n Inference & Fine-tuning

A clean, production-ready Python project for running inference and fine-tuning on **Gemma 3n** multimodal models using Hugging Face Transformers, TRL, and PEFT.

## Features

- ✅ **Multimodal inference**: Text-only, image+text, video+text (frame-based)
- ✅ **Fine-tuning**: LoRA and QLoRA (PEFT) with TRL's SFTTrainer
- ✅ **Flexible video handling**: Sample frames uniformly or at specified FPS
- ✅ **Single-GPU friendly**: Optimized for consumer hardware with 4-bit quantization
- ✅ **CPU fallback**: Run inference on CPU when CUDA is unavailable
- ✅ **Type hints & docstrings**: Clean, maintainable code

## Table of Contents

- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
  - [Quick Setup](#quick-setup)
  - [Manual Setup](#manual-setup)
- [Quick Start](#quick-start)
  - [Text-Only Inference](#text-only-inference)
  - [Image + Text Inference](#image--text-inference)
  - [Video + Text Inference](#video--text-inference)
  - [Fine-Tuning](#fine-tuning)
- [Training Monitoring](#training-monitoring)
- [Model Sharing](#model-sharing)
- [Project Structure](#project-structure)
- [Dataset Format](#dataset-format)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (recommended for GPU acceleration)
- Git

### Install Dependencies

```bash
# Clone or navigate to the project
cd gemma3-testing

# Install requirements
pip install -r requirements.txt
```

### Optional Dependencies

For faster video decoding:

```bash
pip install decord
```

For audio support (future use):

```bash
pip install soundfile
```

### Hugging Face Token

Gemma models may be gated. You'll need to:

1. Accept the license on the [Gemma model page](https://huggingface.co/google/gemma-3n-E2B)
2. Log in to Hugging Face:
   ```bash
   huggingface-cli login
   ```

---

## Dataset Preparation

This project includes utilities for working with the **QVED** (Qualified Exercise Dataset) from EdgeVLM-Labs.

### Quick Setup

Run the complete dataset preparation pipeline:

```bash
# One-command setup (install + dataset + verification)
./scripts/quick_setup.sh

# Or manually initialize dataset only
./scripts/initialize_dataset.sh
```

This will:

1. Download videos from the QVED dataset
2. Filter ground truth annotations
3. Convert to Gemma JSONL format (train/val/test splits)
4. Verify the setup

### Manual Setup

If you prefer manual control:

```bash
# 1. Download videos (will prompt for count per class)
python -m utils.load_dataset

# 2. Filter annotations to match downloaded videos
python -m utils.filter_ground_truth

# 3. Convert to Gemma format with 60/20/20 split
python -m utils.convert_dataset

# 4. Verify everything is set up correctly
./scripts/verify_setup.sh
```

**Dataset Structure:**

```
dataset/
├── videos/
│   ├── exercise1/
│   │   ├── video1.mp4
│   │   └── ...
│   └── exercise2/
│       └── ...
├── manifest.json              # Downloaded video metadata
├── fine_grained_labels.json   # Original QVED annotations
├── ground_truth.json          # Filtered annotations
├── gemma_train.jsonl          # Training set (60%)
├── gemma_val.jsonl            # Validation set (20%)
└── gemma_test.jsonl           # Test set (20%)
```

---

## Quick Start

### Text-Only Inference

```bash
python scripts/run_inference.py \
  --mode text \
  --model_id google/gemma-3n-E2B \
  --prompt "Explain the theory of relativity in simple terms." \
  --max_new_tokens 256
```

### Image + Text Inference

```bash
python scripts/run_inference.py \
  --mode image \
  --model_id google/gemma-3n-E2B \
  --prompt "Describe what you see in this image." \
  --image_path path/to/image.jpg \
  --max_new_tokens 200
```

### Video + Text Inference

Sample 8 frames uniformly:

```bash
python scripts/run_inference.py \
  --mode video \
  --model_id google/gemma-3n-E2B \
  --prompt "What is happening in this video?" \
  --video_path path/to/video.mp4 \
  --num_frames 8 \
  --max_new_tokens 300
```

Sample at 1 FPS:

```bash
python scripts/run_inference.py \
  --mode video \
  --prompt "Describe the key events." \
  --video_path path/to/video.mp4 \
  --fps 1.0 \
  --max_new_tokens 300
```

### Fine-Tuning

Prepare a JSONL file (see [Dataset Format](#dataset-format)), then:

```bash
python scripts/run_finetune.py \
  --model_id google/gemma-3n-E2B \
  --train_jsonl data/train.jsonl \
  --output_dir outputs/gemma3n-finetuned \
  --method qlora \
  --quant 4bit \
  --epochs 3 \
  --batch_size 2 \
  --grad_accum 4 \
  --lr 2e-4 \
  --num_frames 8
```

**Loading the fine-tuned model:**

```python
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained("google/gemma-3n-E2B", trust_remote_code=True)
model = PeftModel.from_pretrained(model, "outputs/gemma3n-finetuned")
processor = AutoProcessor.from_pretrained("google/gemma-3n-E2B", trust_remote_code=True)
```

---

## Training Monitoring

Monitor your training progress with visualization tools:

```bash
# Plot training statistics (auto-detects latest log)
./scripts/plot_training.sh

# Or specify a specific log file
./scripts/plot_training.sh --log_file outputs/training.log --model_name "Gemma-3n-QVED"
```

This generates:

- **training_report.pdf**: Combined plots (loss, gradient norm, learning rate)
- Individual PNG plots for each metric
- **training_summary.txt**: Statistical summary

**Direct Python usage:**

```python
from utils.plot_training_stats import plot_combined

plot_combined(
    log_file="outputs/training.log",
    output_dir="plots",
    model_name="Gemma-3n-QVED"
)
```

---

## Model Sharing

Upload your fine-tuned model to Hugging Face Hub:

```bash
python -m utils.hf_upload \
  --model_path outputs/gemma3n-finetuned \
  --repo_name username/gemma3n-qved \
  --base_model google/gemma-3n-E2B \
  --dataset EdgeVLM-Labs/QVED-Test-Dataset \
  --private
```

This automatically:

- Creates a model card with training details
- Uploads adapter weights or full model
- Documents hyperparameters and usage examples

**From Python:**

```python
from utils.hf_upload import upload_model_to_hf

upload_model_to_hf(
    model_path="outputs/gemma3n-finetuned",
    repo_name="username/gemma3n-qved",
    base_model="google/gemma-3n-E2B",
    dataset_name="EdgeVLM-Labs/QVED-Test-Dataset",
    private=False
)
```

---

## Project Structure

```
.
├── requirements.txt          # Dependencies
├── README.md                 # This file
├── utils/
│   ├── __init__.py
│   ├── config.py             # AppConfig and dtype parser
│   ├── logging_utils.py      # Logging setup
│   ├── model_utils.py        # Model/processor loading
│   ├── prompt_utils.py       # Chat message builders
│   ├── video_utils.py        # Video frame sampling
│   ├── dataset_utils.py      # JSONL dataset loader
│   ├── inference.py          # Inference functions
│   ├── finetune.py           # Fine-tuning with TRL + PEFT
│   ├── load_dataset.py       # QVED video downloader
│   ├── filter_ground_truth.py # Annotation filtering
│   ├── convert_dataset.py    # QVED to Gemma converter
│   ├── plot_training_stats.py # Training visualization
│   └── hf_upload.py          # HuggingFace model upload
└── scripts/
    ├── run_inference.py      # Inference CLI
    ├── run_finetune.py       # Fine-tuning CLI
    ├── initialize_dataset.sh # Complete dataset setup
    ├── verify_setup.sh       # Setup verification
    ├── quick_setup.sh        # One-command install & setup
    └── plot_training.sh      # Training plots wrapper
```

---

## Dataset Format

Create a **JSONL** file where each line is a JSON object:

### Text-only sample:

```json
{
  "prompt": "What is AI?",
  "response": "AI stands for Artificial Intelligence..."
}
```

### Image sample:

```json
{
  "image_path": "images/cat.jpg",
  "prompt": "What animal is this?",
  "response": "This is a cat."
}
```

### Video sample:

```json
{
  "video_path": "videos/running.mp4",
  "prompt": "Describe the action.",
  "response": "A person is running in a park."
}
```

**Notes:**

- Paths can be absolute or relative to the JSONL file location.
- The dataset will automatically load images/videos during training.
- Missing keys are handled gracefully (e.g., text-only samples won't load images).

---

## Advanced Usage

### Quantization

Run inference with 4-bit quantization to save memory:

```bash
python scripts/run_inference.py \
  --mode text \
  --prompt "Hello!" \
  --quant 4bit
```

### Custom Sampling

Sample 16 frames at 2 FPS:

```bash
python scripts/run_inference.py \
  --mode video \
  --video_path video.mp4 \
  --num_frames 16 \
  --fps 2.0 \
  --prompt "Analyze the video"
```

### CPU Inference

```bash
python scripts/run_inference.py \
  --mode text \
  --prompt "Test" \
  --device cpu \
  --dtype fp32
```

### Fine-Tuning with LoRA (no quantization)

```bash
python scripts/run_finetune.py \
  --train_jsonl data/train.jsonl \
  --output_dir outputs/lora \
  --method lora \
  --quant none \
  --dtype bf16 \
  --epochs 2
```

### Verbose Logging

```bash
python scripts/run_inference.py --mode text --prompt "Test" -vv
```

---

## Troubleshooting

### Issue: `bitsandbytes` not found

**Solution:**

```bash
pip install bitsandbytes
```

If you're on Windows or have issues, see [bitsandbytes installation guide](https://github.com/bitsandbytes-foundation/bitsandbytes#installation).

### Issue: `decord` not installed

**Solution:**

```bash
pip install decord
```

Alternatively, use `--video_method opencv` (the default).

### Issue: CUDA out of memory

**Solutions:**

1. Use 4-bit quantization: `--quant 4bit`
2. Reduce batch size: `--batch_size 1 --grad_accum 16`
3. Reduce `--max_seq_len` (default 2048)
4. Sample fewer frames: `--num_frames 4`

### Issue: Model is gated or requires authentication

**Solution:**

1. Accept the license: [https://huggingface.co/google/gemma-3n-E2B](https://huggingface.co/google/gemma-3n-E2B)
2. Log in:
   ```bash
   huggingface-cli login
   ```

### Issue: `trust_remote_code` error

**Solution:**
Gemma 3n uses custom modeling code. The scripts automatically set `trust_remote_code=True`. If you modify the code, ensure this flag is set.

### Issue: Video frames not loading

**Causes:**

- File path incorrect
- Video codec not supported by OpenCV

**Solutions:**

1. Verify file exists: `ls path/to/video.mp4`
2. Try decord: `--video_method decord`
3. Convert video to a common format (e.g., H.264 MP4)

### Issue: Training loss is NaN

**Solutions:**

- Lower learning rate: `--lr 1e-4`
- Check dataset quality (ensure responses aren't empty)
- Ensure images/videos load correctly (check logs)

---

## References

### Official Documentation

- **Gemma 3n Overview**: [https://ai.google.dev/gemma/docs/gemma-3n](https://ai.google.dev/gemma/docs/gemma-3n)
- **Video Understanding**: [https://ai.google.dev/gemma/docs/capabilities/vision/video-understanding](https://ai.google.dev/gemma/docs/capabilities/vision/video-understanding)
- **Audio (Optional)**: [https://ai.google.dev/gemma/docs/capabilities/audio](https://ai.google.dev/gemma/docs/capabilities/audio)

### Model Cards

- **Gemma 3n-E2B**: [https://huggingface.co/google/gemma-3n-E2B](https://huggingface.co/google/gemma-3n-E2B)
- **Gemma 3n-E4B**: [https://huggingface.co/google/gemma-3n-E4B](https://huggingface.co/google/gemma-3n-E4B)

### Fine-Tuning Guides

- **Vision QLoRA**: [https://ai.google.dev/gemma/docs/core/huggingface_vision_finetune_qlora](https://ai.google.dev/gemma/docs/core/huggingface_vision_finetune_qlora)
- **Text QLoRA**: [https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora](https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora)

### Libraries

- **Transformers**: [https://huggingface.co/docs/transformers/](https://huggingface.co/docs/transformers/)
- **Chat Templates**: [https://huggingface.co/docs/transformers/en/chat_templating](https://huggingface.co/docs/transformers/en/chat_templating)
- **TRL**: [https://huggingface.co/docs/trl/](https://huggingface.co/docs/trl/)
- **PEFT**: [https://huggingface.co/docs/peft/](https://huggingface.co/docs/peft/)

### Ecosystem

- **Gemma 3n Announcement**: [https://huggingface.co/blog/gemma3n](https://huggingface.co/blog/gemma3n)

---

## License

This project is provided as-is for educational and research purposes. Gemma models are subject to their own licenses—please review them on Hugging Face before use.

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear descriptions

---

## Contact

For issues or questions, please open an issue on GitHub.

---

**Happy fine-tuning! 🚀**
