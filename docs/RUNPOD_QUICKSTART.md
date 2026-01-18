# RunPod Quick Start Guide

## Initial Setup (One-time)

```bash
# 1. Update and install dependencies
apt-get update
apt-get install -y wget git build-essential

# 2. Clone the repository
git clone https://github.com/EdgeVLM-Labs/gemma3-testing.git
cd gemma3-testing
git checkout gemma-1

# 3. Run automated setup (installs conda, creates environment, installs packages)
bash setup.sh

# 4. After setup completes, reload shell and activate environment
source ~/.bashrc
conda activate gemma3n
```

## Manual Environment Setup (Alternative)

If the automated setup fails, follow these steps:

```bash
# 1. Accept conda Terms of Service (required for newer conda versions)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# 2. Create environment
conda create -n gemma3n python=3.11 -y

# 3. Activate environment
conda activate gemma3n

# 4. Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. Install project dependencies
pip install -r requirements.txt

# 6. Fix potential issues
bash fix_unsloth.sh
bash fix_torch_int1.sh
```

## Running Inference on Sample Videos

```bash
# 1. Make sure you're in the gemma3n environment
conda activate gemma3n

# 2. Create test dataset from sample_videos
python create_test_json.py --video_dir sample_videos

# 3. Run inference with your fine-tuned model
bash scripts/run_inference.sh \
  --hf_repo EdgeVLM-Labs/gemma-3n-E2B-qved-1000 \
  --test_json dataset/qved_test.json \
  --data_path sample_videos
```

## Running Inference on Custom Videos

```bash
# 1. Place your videos in a folder (e.g., test_videos/)
mkdir -p test_videos
# Copy your .mp4 files to test_videos/

# 2. Create test dataset JSON
python create_test_json.py --video_dir test_videos

# 3. Run inference
bash scripts/run_inference.sh \
  --hf_repo EdgeVLM-Labs/gemma-3n-E2B-qved-1000 \
  --test_json dataset/qved_test.json \
  --data_path test_videos \
  --limit 10
```

## Output Files

Inference results are saved to:
- **Predictions JSON**: `results/test_inference_<model_name>/test_predictions.json`
- **Evaluation Report**: `results/test_inference_<model_name>/test_evaluation_report.xlsx`

## Common Issues

### Issue: "conda: command not found"
**Solution**: Install conda first:
```bash
bash setup.sh
source ~/.bashrc
```

### Issue: "CondaToSNonInteractiveError: Terms of Service have not been accepted"
**Solution**: Accept the ToS:
```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

### Issue: "RuntimeError: Unsloth: The tokenizer is weirdly not loaded?"
**Solution**: This is fixed in the latest version. Make sure you have the updated code:
```bash
git pull origin gemma-1
```

### Issue: "Could not open video file"
**Solution**: Check your video path and make sure `--data_path` points to the correct directory:
```bash
# If videos are in /workspace/gemma3-testing/test_videos/
bash scripts/run_inference.sh \
  --hf_repo EdgeVLM-Labs/gemma-3n-E2B-qved-1000 \
  --data_path /workspace/gemma3-testing/test_videos
```

## Environment Management

```bash
# Activate environment
conda activate gemma3n

# Deactivate environment
conda deactivate

# List all environments
conda env list

# Delete environment (if needed)
conda env remove -n gemma3n
```

## HuggingFace Authentication

If your model is private or you need to download from HuggingFace:

```bash
# Install huggingface-cli
pip install huggingface_hub

# Login
huggingface-cli login
# Enter your HuggingFace token when prompted
```
