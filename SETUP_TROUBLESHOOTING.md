# Setup Troubleshooting Guide

Quick fixes for common setup issues with Gemma-3N fine-tuning environment.

---

## Issue: "Terms of Service have not been accepted"

**Error message:**
```
CondaToSNonInteractiveError: Terms of Service have not been accepted
```

**Fix:**
The updated `setup.sh` now handles this automatically. If you still see this error:

```bash
# Accept conda TOS manually
conda config --set tos_accepted yes

# Then re-run setup
bash setup.sh
```

---

## Issue: "Environment gemma3n not found"

**Error message:**
```
EnvironmentNameNotFound: Could not find conda environment: gemma3n
```

**Fix:**

```bash
# 1. Make sure conda is initialized
source ~/.bashrc

# 2. Accept conda TOS
conda config --set tos_accepted yes

# 3. Run setup again
bash setup.sh
```

The setup script will detect if the environment already exists and skip creation.

---

## Issue: Setup script fails during package installation

**Symptoms:**
- Script exits with pip errors
- Some packages fail to install

**Fix:**

```bash
# 1. Activate the environment first
conda activate gemma3n

# 2. Install requirements manually
pip install -r requirements.txt

# 3. Install critical packages
pip install transformers==4.56.2
pip install --no-deps trl==0.22.2
pip install --no-deps --upgrade timm
pip install --upgrade unsloth unsloth_zoo
pip install opencv-python matplotlib wandb
pip install nltk rouge-score sacrebleu openpyxl sentence-transformers
```

---

## Issue: "conda activate gemma3n" doesn't work

**Fix:**

```bash
# Option 1: Initialize conda in current shell
eval "$(conda shell.bash hook)"
conda activate gemma3n

# Option 2: Restart terminal and try again
source ~/.bashrc
conda activate gemma3n

# Option 3: Use conda run instead
conda run -n gemma3n python your_script.py
```

---

## Issue: CUDA not available in PyTorch

**Symptoms:**
```python
torch.cuda.is_available()  # Returns False
```

**Check:**

```bash
# Check if NVIDIA driver is working
nvidia-smi

# Check CUDA installation
nvcc --version

# Check PyTorch installation
conda activate gemma3n
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

**Fix:**

If CUDA is not available, reinstall PyTorch with CUDA:

```bash
conda activate gemma3n
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Issue: Unsloth import fails

**Error message:**
```
ImportError: cannot import name 'FastVisionModel'
```

**Fix:**

```bash
conda activate gemma3n

# Update timm first (required dependency)
pip install --no-deps --upgrade timm

# Reinstall unsloth
pip uninstall unsloth unsloth_zoo -y
pip install --upgrade unsloth unsloth_zoo

# Verify
python -c "from unsloth import FastVisionModel; print('✅ Success')"
```

---

## Issue: Transformers version mismatch

**Symptoms:**
- Model loading fails
- Compatibility errors

**Fix:**

```bash
conda activate gemma3n

# Force correct version
pip install transformers==4.56.2 --force-reinstall

# Verify
python -c "import transformers; print(transformers.__version__)"
```

---

## Issue: Out of memory during setup

**Symptoms:**
- Setup hangs
- Installation fails with memory errors

**Fix:**

```bash
# Use --no-cache-dir to reduce memory usage
conda activate gemma3n
pip install -r requirements.txt --no-cache-dir
```

---

## Quick Recovery: Start Fresh

If nothing works, completely reset:

```bash
# 1. Remove existing environment
conda deactivate
conda env remove -n gemma3n -y

# 2. Clear conda cache
conda clean --all -y

# 3. Accept TOS
conda config --set tos_accepted yes

# 4. Run setup again
bash setup.sh
```

---

## Verify Installation

After setup completes, verify everything works:

```bash
# Activate environment
conda activate gemma3n

# Test all components
python -c "
import torch
from unsloth import FastVisionModel
import transformers

print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('Transformers:', transformers.__version__)
print('FastVisionModel imported successfully')
print('✅ All checks passed!')
"
```

---

## Still Having Issues?

1. **Check the logs** - The setup script shows detailed error messages
2. **Run with verbose mode**:
   ```bash
   bash -x setup.sh 2>&1 | tee setup.log
   ```
3. **Check system requirements**:
   - NVIDIA GPU with CUDA support
   - 16GB+ GPU VRAM
   - 50GB+ free disk space
   - Python 3.11
   - CUDA 11.8 or 12.0+

4. **Manual installation** - If setup.sh continues to fail, see [MANUAL_INSTALL.md](docs/MANUAL_INSTALL.md) for step-by-step manual installation

---

## Contact Support

If you've tried all troubleshooting steps and still encounter issues:

- **GitHub Issues**: [Create an issue](https://github.com/EdgeVLM-Labs/gemma3-testing/issues)
- **Include**:
  - Full error message
  - Output of `conda list -n gemma3n`
  - Output of `nvidia-smi`
  - Your operating system and GPU model
