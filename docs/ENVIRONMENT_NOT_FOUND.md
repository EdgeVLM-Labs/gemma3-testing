# Quick Fix: Environment Not Found

If you see this error:
```
EnvironmentNameNotFound: Could not find conda environment: gemma3n
```

## Solution

You need to **create the environment first** before you can activate it.

### Option 1: Run the Setup Script (Recommended)

```bash
cd /workspace/gemma3-testing
bash setup.sh
```

This will:
- Install/update Miniconda if needed
- Create the `gemma3n` conda environment with Python 3.11
- Install all required dependencies from requirements.txt

Then activate:
```bash
conda activate gemma3n
```

### Option 2: Manual Environment Creation

If setup.sh doesn't work, create manually:

```bash
# Create environment with Python 3.11
conda create -n gemma3n python=3.11 -y

# Activate it
conda activate gemma3n

# Install dependencies
pip install -r requirements.txt
```

### Verify It Worked

```bash
# Check environment exists
conda info --envs

# Should see output like:
# gemma3n                  /root/miniconda3/envs/gemma3n
# base                  *  /root/miniconda3

# Verify Python version
python --version
# Should show: Python 3.11.x
```

## Correct Setup Order

For future reference, the correct order is:

1. Install system dependencies
2. Clone repository
3. **Run `bash setup.sh`** ← Creates the environment
4. Accept Conda ToS (optional, for some packages)
5. Activate environment with `conda activate gemma3n`
6. Continue with dataset download, etc.

## Your Next Steps

Now that you understand the issue, run:

```bash
cd /workspace/gemma3-testing

# Create the environment
bash setup.sh

# Wait for it to complete, then:
conda activate gemma3n

# Verify
python --version
pip list | grep torch

# Continue with setup
python dataset.py download
python dataset.py prepare
```
