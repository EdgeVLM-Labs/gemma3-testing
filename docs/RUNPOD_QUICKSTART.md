# RunPod Quick Start Guide

## Initial Setup (One-time)

```bash
# 1. Update and install dependencies
apt-get update
apt-get install -y wget git build-essential

# 2. Clone the repository
git clone https://github.com/EdgeVLM-Labs/gemma3-testing.git
cd gemma3-testing

# 3. Run automated setup (installs conda, creates environment, installs packages)
bash setup.sh

# 4. After setup completes, reload shell and activate environment
source ~/.bashrc
conda activate gemma3n
```

## Manual Environment Setup (Alternative)

If the automated setup fails:

```bash
# 1. Accept conda Terms of Service
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# 2. Create environment
conda create -n gemma3n python=3.11 -y

# 3. Activate environment
conda activate gemma3n

# 4. Install PyTorch with CUDA support
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# 5. Install project dependencies
pip install -r requirements.txt

# 6. If issues arise, re-run setup
bash setup.sh
```

## Running Inference

```bash
# 1. Make sure you're in the gemma3n environment
conda activate gemma3n

# 2. Run inference with transformers
bash scripts/run_inference_transformers.sh \
  --test_json dataset/qved_test.json \
  --data_path dataset
```

## Running Fine-tuning

```bash
bash scripts/finetune_gemma3n_e2b_trl.sh
```

## Common Issues

### Issue: "conda: command not found"
```bash
bash setup.sh
source ~/.bashrc
```

### Issue: "Could not open video file"
Check your video path and make sure `--data_path` points to the correct directory.

## HuggingFace Authentication

```bash
huggingface-cli login
# Enter your HuggingFace token when prompted
```
