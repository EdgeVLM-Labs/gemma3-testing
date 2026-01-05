#!/bin/bash

echo "========================================="
echo "Gemma-3N-E2B Finetuning Setup Verification"
echo "========================================="

# 1️⃣ Check dataset split files
echo -e "\n[1] Checking dataset files..."

declare -A dataset_files=(
    ["qved_train.json"]="Training set"
    ["qved_val.json"]="Validation set"
    ["qved_test.json"]="Test set"
    ["manifest.json"]="Manifest file"
    ["ground_truth.json"]="Ground truth labels"
)

for file in "${!dataset_files[@]}"; do
    if [ -f "dataset/$file" ]; then
        if [[ "$file" == *.json ]]; then
            num_samples=$(python -c "import json; print(len(json.load(open('dataset/$file'))))" 2>/dev/null || echo "?")
            echo "✓ dataset/$file found ($num_samples samples)"
        else
            echo "✓ dataset/$file found"
        fi
    else
        echo "✗ dataset/$file NOT found"
    fi
done

# 2️⃣ Check video folders
echo -e "\n[2] Checking video files..."
exercise_dirs=(
    "alternating_single_leg_glutes_bridge"
    "cat-cow_pose"
    "elbow_plank"
    "glute_hamstring_walkout"
    "glutes_bridge"
    "heel_lift"
    "high_plank"
    "lunges_leg_out_in_front"
    "opposite_arm_and_leg_lifts_on_knees"
    "pushups"
    "side_plank"
    "squats"
    "toe_touch"
    "tricep_stretch"
)

for dir in "${exercise_dirs[@]}"; do
    if [ -d "dataset/$dir" ]; then
        num_videos=$(ls -1 dataset/$dir/*.mp4 2>/dev/null | wc -l)
        echo "✓ dataset/$dir/ found ($num_videos videos)"
    else
        echo "✗ dataset/$dir/ NOT found"
    fi
done

# 3️⃣ Verify video paths in split JSON files
echo -e "\n[3] Verifying video paths in dataset splits..."

for split in qved_train.json qved_val.json qved_test.json; do
    if [ -f "dataset/$split" ]; then
        result=$(python -c "
import json, os
with open('dataset/$split') as f:
    data = json.load(f)
    total = len(data)
    missing = sum(1 for item in data if not os.path.exists(os.path.join('dataset', item.get('video',''))))
    print(f'{total},{missing}')
" 2>/dev/null)
        total_count=$(echo $result | cut -d',' -f1)
        missing_count=$(echo $result | cut -d',' -f2)
        if [ "$missing_count" -eq 0 ]; then
            echo "✓ $split: All $total_count video paths are valid"
        else
            echo "✗ $split: $missing_count out of $total_count videos are missing"
        fi
    fi
done

# 4️⃣ Check required scripts
echo -e "\n[4] Checking required scripts..."
required_scripts=("scripts/finetune_gemma.sh" "scripts/zero3.json")
for script in "${required_scripts[@]}"; do
    if [ -f "$script" ]; then
        echo "✓ $script found"
    else
        echo "✗ $script NOT found"
    fi
done

# 5️⃣ Check Python / Conda environment
echo -e "\n[5] Checking environment..."
if command -v conda &>/dev/null; then
    if conda env list | grep -q "gemma3n_env"; then
        echo "✓ Conda environment 'gemma3n_env' exists"
    else
        echo "✗ Conda environment 'gemma3n_env' NOT found"
    fi
elif command -v python &>/dev/null; then
    python_path=$(which python)
    echo "⚠ Conda not found, Python available at: $python_path"
else
    echo "✗ Neither conda nor python found"
fi

# 6️⃣ Check GPU availability
echo -e "\n[6] Checking GPU availability..."
if command -v nvidia-smi &>/dev/null; then
    gpu_count=$(nvidia-smi --list-gpus | wc -l)
    echo "✓ $gpu_count GPU(s) detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "✗ nvidia-smi not found - GPU may not be available"
fi

echo -e "\n========================================="
echo "Setup verification complete!"
echo "========================================="
echo -e "\nTo start finetuning Gemma-3N-E2B, run:"
echo "  conda activate gemma3n_env"
echo "  bash scripts/finetune_gemma.sh"
echo "========================================="
