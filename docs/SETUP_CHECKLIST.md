# Setup Checklist

## Pre-Setup Requirements

- [ ] Linux/macOS (Ubuntu 20.04+ recommended)
- [ ] NVIDIA GPU with CUDA support (16GB+ VRAM recommended)
- [ ] 50GB+ free disk space
- [ ] Internet connection
- [ ] HuggingFace account (https://huggingface.co)

## Setup Steps

### 1. Install System Dependencies
```bash
sudo apt-get update
sudo apt-get install -y wget git build-essential
```

### 2. Clone Repository
```bash
git clone https://github.com/EdgeVLM-Labs/gemma3-testing.git
cd gemma3-testing
```

### 3. Run Setup Script
```bash
bash setup.sh
```
- [ ] Conda environment `gemma3n` created
- [ ] PyTorch installed with CUDA
- [ ] All dependencies installed

### 4. Activate Environment
```bash
conda activate gemma3n
```
- [ ] Python 3.11 available

### 5. HuggingFace Authentication
```bash
huggingface-cli login
```
- [ ] Token entered and verified
- [ ] Access requested for `google/gemma-3n-E2B` model

### 6. Verify GPU
```bash
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```
- [ ] CUDA available shows `True`

### 7. Initialize Dataset
```bash
bash scripts/initialize_dataset.sh
```
- [ ] Videos downloaded
- [ ] Train/val/test splits created

### 8. Verify Setup
```bash
bash scripts/verify_qved_setup.sh
```
- [ ] All checks passed

## Ready to Start

### Fine-tune:
```bash
bash scripts/finetune_gemma3n_e2b_trl.sh
```

### Run inference:
```bash
bash scripts/run_inference_transformers.sh \
  --test_json dataset/qved_test.json \
  --data_path dataset
```

---

## Troubleshooting

### "No module named 'torch'"
```bash
conda activate gemma3n
pip install -r requirements.txt
```

### "403 Forbidden" for google/gemma-3n-E2B
1. Visit https://huggingface.co/google/gemma-3n-E2B
2. Click "Request Access" and accept terms
3. Run `huggingface-cli login` again

### "CUDA out of memory"
Reduce batch size in the training script.
