# Quick Start Guide

## Get Started

### 1. Setup Environment (First Time Only)

```bash
git clone https://github.com/EdgeVLM-Labs/gemma3-testing.git
cd gemma3-testing

# Run setup script
bash setup.sh

# Activate environment
conda activate gemma3n

# Login to Hugging Face
huggingface-cli login
```

**If you encounter dependency issues, re-run the setup:**
```bash
bash setup.sh
```

### 2. Initialize Dataset

```bash
# Interactive dataset initialization
bash scripts/initialize_dataset.sh

# Verify setup
bash scripts/verify_qved_setup.sh
```

### 3. Fine-tune Model

```bash
# Using shell wrapper
bash scripts/finetune_gemma3n_e2b_trl.sh

# Or directly
python core/finetune_gemma3n_e2b_trl.py \
  --train_json dataset/qved_train.json \
  --val_json dataset/qved_val.json \
  --video_path dataset \
  --output_dir outputs/gemma3n_finetune
```

### 4. Run Inference

```bash
bash scripts/run_inference_transformers.sh \
  --test_json dataset/qved_test.json \
  --data_path dataset
```

### 5. Upload Model to HuggingFace

```bash
python utils/hf_upload.py \
  --model_path outputs/gemma3n_finetune/checkpoint-70 \
  --repo_name my-gemma3n-model \
  --org your-org
```

---

## Troubleshooting

**Gated Model 403:**
```bash
# 1. Visit https://huggingface.co/google/gemma-3n-E2B
# 2. Request access + accept terms
# 3. huggingface-cli login
```

**PEFT Import Error:**
```bash
pip install --upgrade peft
```

**CUDA Out of Memory:**
Reduce batch size in `scripts/finetune_gemma3n_e2b_trl.sh`

---

## Resources

- [Full README](../README.md)
- [Known Issues](issues.md)
- [QVED Dataset](https://huggingface.co/datasets/EdgeVLM-Labs/QEVD-fine-grained-feedback-cleaned)
