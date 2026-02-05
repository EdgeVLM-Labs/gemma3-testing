# Quick Start Guide

## 🎯 Get Started in 5 Minutes

### 1. Setup Environment (First Time Only)

```bash
# Clone repository
git clone https://github.com/EdgeVLM-Labs/gemma3-testing.git
cd gemma3-testing

# Run setup script
bash setup.sh

# Activate environment
conda activate gemma3n

# Login to Hugging Face
hf auth login
```

**If you encounter dependency issues (mamba-ssm, unsloth, etc.):**
```bash
bash finetune_env.sh  # Complete fine-tuning environment setup
```

### 2. Initialize Dataset

```bash
# Interactive dataset initialization
bash scripts/initialize_dataset.sh

# Verify setup
bash scripts/verify_qved_setup.sh
```

### 3. Run Inference (No Training Needed)

**Single Video:**
```bash
python utils/infer_qved.py \
  --video_path sample_videos/00000340.mp4 \
  --prompt "Analyze this exercise form"
```

**Batch Videos:**
```bash
python gemma3n_batch_inference.py \
  --video_folder sample_videos \
  --output results/batch_results.csv
```

### 4. Fine-tune Model (Optional)

```bash
# Start fine-tuning on QVED
bash scripts/initialize_dataset.sh
```

This will:
- Configure training parameters
- Start training for 3 epochs
- Save checkpoints to `results/qved_finetune_gemma3n_E2B/`
- Track progress with WandB

### 5. Evaluate Model

```bash
# Run inference on test set with evaluation report
bash scripts/run_inference.sh \
  --model_path outputs/gemma3n_finetune_20260108_162806_merged_16bit

# Test with base model for comparison
bash scripts/run_inference.sh \
  --model_path unsloth/gemma-3n-E4B-it

# Outputs:
# - results/test_inference_<model>/test_predictions.json
# - results/test_inference_<model>/test_evaluation_report.xlsx (with BERT/ROUGE/METEOR metrics)
```

---

## 🔥 Common Tasks

### Test Pre-trained Model
```bash
python utils/infer_qved.py \
  --model_path google/gemma-3n-E2B-it \
  --video_path sample_videos/00000340.mp4
```

### Process Multiple Videos
```bash
python gemma3n_batch_inference.py \
  --model google/gemma-3n-E2B-it \
  --video_folder your_videos/ \
  --output results/results.csv \
  --show_stream  # Watch generation in real-time
```

### Fine-tune with Custom Data
1. Place videos in `dataset/` with folder structure
2. Create `dataset/fine_grained_labels.json`
3. Run `python dataset.py prepare`
4. Run `bash scripts/initialize_dataset.sh`

### Upload Model to HuggingFace
```bash
python utils/hf_upload.py \
  --model_path results/qved_finetune_gemma3n_E2B/checkpoint-70 \
  --repo_name my-gemma3n-model \
  --org your-org
```

---

## ⚠️ Troubleshooting

**Disk Space Error:**
```bash
rm -rf ~/.cache/huggingface/hub/*
df -h  # Check available space
```

**Gated Model 403:**
```bash
# 1. Visit https://huggingface.co/google/gemma-3n-E2B
# 2. Request access + accept terms
# 3. hf auth login
# OR use: --model unsloth/gemma-3n-E2B
```

**PEFT Import Error:**
```bash
pip install --upgrade peft
```

**CUDA Out of Memory:**
```bash
# Reduce batch size or use 4-bit quantization
# Edit scripts/finetune_qved.sh:
BATCH=4  # instead of 8
```

---

## 📖 Next Steps

- Read full [README.md](../README.md)
- Check [docs/issues.md](issues.md) for known issues
- See [docs/finetuning_updates.md](finetuning_updates.md) for advanced training

---

## 🎓 Learning Resources

- [Gemma-3N Documentation](https://huggingface.co/google/gemma-3n-E2B)
- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [QVED Dataset](https://huggingface.co/datasets/EdgeVLM-Labs/QEVD-fine-grained-feedback-cleaned)
