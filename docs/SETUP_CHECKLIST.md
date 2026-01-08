# 🚀 Gemma3-Testing Setup Checklist

Use this checklist to ensure your environment is properly configured.

## ☑️ Pre-Setup Requirements

- [ ] Linux/macOS operating system (Ubuntu 20.04+ recommended)
- [ ] NVIDIA GPU with CUDA support (16GB+ VRAM recommended)
- [ ] 50GB+ free disk space available
- [ ] Root/sudo access for system packages
- [ ] Git installed or will be installed in next step
- [ ] Internet connection for downloads
- [ ] HuggingFace account created (https://huggingface.co)

## ☑️ Initial Setup Steps

### 1. Install System Dependencies (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install -y wget git build-essential
```
- [ ] System packages updated
- [ ] wget installed
- [ ] git installed
- [ ] build-essential installed (includes gcc, make, etc.)

### 2. Clone Repository
```bash
git clone https://github.com/EdgeVLM-Labs/gemma3-testing.git
cd gemma3-testing
```
- [ ] Repository cloned successfully
- [ ] Changed directory to `gemma3-testing`

### 2. Clone Repository
```bash
git clone https://github.com/EdgeVLM-Labs/gemma3-testing.git
cd gemma3-testing
```
- [ ] Repository cloned successfully
- [ ] Changed directory to `gemma3-testing`

### 3. Run Setup Script
```bash
bash setup.sh
```
- [ ] Setup script completed without errors
- [ ] Miniconda installed (if not already present)
- [ ] Conda environment `gemma3n` created
- [ ] Dependencies installed

### 4. Activate Environment
```bash
conda activate gemma3n
```
- [ ] Environment activated (should see `(gemma3n)` in terminal)
- [ ] Python available: `python --version` shows Python 3.11

### 5. Accept Conda Terms
```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```
- [ ] Conda ToS accepted for main channel
- [ ] Conda ToS accepted for r channel

### 6. HuggingFace Authentication
```bash
huggingface-cli login
# Or: hf auth login
```
- [ ] HuggingFace token generated (https://huggingface.co/settings/tokens)
- [ ] Token entered and verified
- [ ] Access requested for `google/gemma-3n-E2B` model
- [ ] Model terms accepted and approved

### 7. Verify GPU
```bash
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```
- [ ] `nvidia-smi` shows GPU information
- [ ] CUDA available shows `True`
- [ ] PyTorch can access GPU

### 8. Download Dataset
```bash
python dataset.py download
```
- [ ] Dataset download started
- [ ] Videos downloaded to `dataset/` folder
- [ ] No rate limit errors or download failures

### 9. Prepare Dataset
```bash
python dataset.py prepare
```
- [ ] Train/val/test splits created
- [ ] `dataset/qved_train.json` exists
- [ ] `dataset/qved_val.json` exists
- [ ] `dataset/qved_test.json` exists

### 10. Verify Setup
```bash
bash scripts/verify_qved_setup.sh
```
- [ ] All dataset files found
- [ ] Video folders exist with correct counts
- [ ] Video paths are valid
- [ ] Required scripts exist
- [ ] Conda environment detected
- [ ] GPU availability confirmed

### 11. Test Inference (Optional)
```bash
python gemma3n_batch_inference.py \
  --model unsloth/gemma-3n-E4B-it \
  --video_folder sample_videos \
  --output results/test.csv \
  --limit 1
```
- [ ] Model loaded successfully
- [ ] Inference completed
- [ ] Results saved to CSV

## ☑️ Troubleshooting Checklist

If you encounter issues, verify:

- [ ] All commands run from `gemma3-testing` directory
- [ ] Conda environment is activated (`conda activate gemma3n`)
- [ ] Python version is 3.11 (`python --version`)
- [ ] Required packages installed (`pip list | grep -E "torch|transformers|unsloth"`)
- [ ] Sufficient disk space (`df -h`)
- [ ] GPU accessible (`nvidia-smi`)
- [ ] HuggingFace token is valid (try `huggingface-cli whoami`)
- [ ] Model access approved (check email from HuggingFace)

## ☑️ Common Issues & Solutions

### ❌ "Command not found: conda"
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gemma3n
```

### ❌ "No module named 'torch'"
```bash
conda activate gemma3n
pip install -r requirements.txt
```

### ❌ "403 Forbidden" for google/gemma-3n-E2B
1. Visit https://huggingface.co/google/gemma-3n-E2B
2. Click "Request Access"
3. Accept terms
4. Wait for approval email
5. Run `hf auth login` again

### ❌ "CUDA out of memory"
- Use smaller model: `unsloth/gemma-3n-E2B` (smaller variant)
- Reduce batch size in scripts
- Close other GPU-using processes
- Restart kernel/terminal

### ❌ "Dataset files not found"
```bash
# Re-download dataset
python dataset.py download
python dataset.py prepare

# Verify
ls -la dataset/
```

## ✅ Ready to Start!

Once all checkboxes are complete, you're ready to:

### Fine-tune the model:
```bash
bash scripts/finetune_qved.sh
```

### Run inference:
```bash
# Single video inference
python utils/infer_qved.py \
  --model_path unsloth/gemma-3n-E4B-it \
  --video_path sample_videos/00000340.mp4

# Batch inference on test set with evaluation report
bash scripts/run_inference.sh \
  --model_path outputs/gemma3n_finetune_merged_16bit
```

---

## 📚 Additional Resources

- **Full Documentation**: [README.md](../README.md)
- **Quick Start Guide**: [docs/QUICKSTART.md](QUICKSTART.md)
- **Architecture Details**: [gemma3/docs/ARCHITECTURE.md](../gemma3/docs/ARCHITECTURE.md)
- **Known Issues**: [docs/issues.md](issues.md)
- **Training Updates**: [docs/finetuning_updates.md](finetuning_updates.md)

---

## 🆘 Need Help?

- Check troubleshooting section in main README
- Review error messages carefully
- Verify all prerequisites are met
- Open an issue: https://github.com/EdgeVLM-Labs/gemma3-testing/issues
- Include: OS, GPU, Python version, full error message

Happy training! 🚀
