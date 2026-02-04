#!/usr/bin/env python3
"""
Prepare QVED dataset for Google GEMMA-3N-E2B / Mobile-VideoGPT inference.

This script:
  1. Loads a manifest mapping video filenames to paths.
  2. Loads fine-grained labels (ground_truth.json).
  3. Converts the dataset into Mobile-VideoGPT conversation format.
  4. Splits data into train, val, and test sets.
  5. Saves JSON files ready for training or evaluation.

Usage:
  python prepare_qved_dataset.py
"""

import json
import os
import random
from pathlib import Path

# ----------------- CONFIG ----------------- #
BASE_DIR = Path("dataset")  # Root dataset folder
MANIFEST_JSON = BASE_DIR / "manifest.json"          # Mapping of filenames -> full paths
FINE_LABELS_JSON = BASE_DIR / "ground_truth.json"  # Fine-grained labels

OUTPUT_TRAIN_JSON = BASE_DIR / "qved_train.json"
OUTPUT_VAL_JSON = BASE_DIR / "qved_val.json"
OUTPUT_TEST_JSON = BASE_DIR / "qved_test.json"

# User prompt for GEMMA-3N-E2B
USER_PROMPT_TEMPLATE = "Please evaluate the exercise form shown. What mistakes, if any, are present, and what corrections would you recommend?"

# Dataset split ratios
TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20
RANDOM_SEED = 42  # For reproducibility

# ------------------------------------------ #

def load_manifest(manifest_path):
    """Load manifest JSON mapping filenames to paths."""
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    filename_to_path = {}
    filename_to_exercise = {}

    for full_path, exercise in manifest.items():
        filename = os.path.basename(full_path)
        filename_to_path[filename] = full_path

        if isinstance(exercise, str):
            filename_to_exercise[filename] = exercise.replace('_', ' ')
        elif isinstance(exercise, dict):
            exercise_name = full_path.split('/')[0] if '/' in full_path else 'unknown'
            filename_to_exercise[filename] = exercise_name.replace('_', ' ')
        else:
            filename_to_exercise[filename] = str(exercise).replace('_', ' ')

    return filename_to_path, filename_to_exercise

def load_fine_labels(labels_path):
    """Load fine-grained labels JSON."""
    with open(labels_path, 'r') as f:
        return json.load(f)

def prepare_output_data(fine_labels, filename_to_path, filename_to_exercise):
    """Convert raw labels into Mobile-VideoGPT conversation format."""
    output_data = []

    for record in fine_labels:
        video_path = record.get('video_path', '')
        filename = os.path.basename(video_path)

        if filename not in filename_to_path:
            print(f"⚠️ Warning: {filename} not found in manifest, skipping")
            continue

        full_video_path = filename_to_path[filename]
        exercise = filename_to_exercise[filename]

        # Make path relative to dataset root
        if full_video_path.startswith('dataset/'):
            relative_video_path = full_video_path[len('dataset/'):]
        else:
            relative_video_path = full_video_path

        # Determine assistant answer
        if 'labels_descriptive' in record and record['labels_descriptive']:
            assistant_answer = record['labels_descriptive']
        elif 'labels' in record and record['labels']:
            if isinstance(record['labels'], list):
                assistant_answer = record['labels'][0]
            else:
                assistant_answer = record['labels']
        else:
            assistant_answer = "No feedback available."

        if isinstance(assistant_answer, list):
            assistant_answer = '\n'.join(str(item) for item in assistant_answer)
        else:
            assistant_answer = str(assistant_answer)

        user_prompt = USER_PROMPT_TEMPLATE

        output_data.append({
            "video": relative_video_path,
            "conversations": [
                {"from": "human", "value": user_prompt},
                {"from": "gpt", "value": assistant_answer}
            ],
            "split": "train"  # Temporary, will be updated during split
        })

    return output_data

def split_dataset(output_data):
    """Shuffle and split dataset into train, val, test."""
    random.seed(RANDOM_SEED)
    random.shuffle(output_data)

    total_count = len(output_data)
    train_end = int(total_count * TRAIN_RATIO)
    val_end = train_end + int(total_count * VAL_RATIO)

    train_data = output_data[:train_end]
    val_data = output_data[train_end:val_end]
    test_data = output_data[val_end:]

    for item in train_data:
        item["split"] = "train"
    for item in val_data:
        item["split"] = "val"
    for item in test_data:
        item["split"] = "test"

    return train_data, val_data, test_data

def save_json(data, output_path):
    """Save data to JSON file."""
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

def main():
    print("📂 Loading manifest...")
    filename_to_path, filename_to_exercise = load_manifest(MANIFEST_JSON)

    print("📝 Loading fine-grained labels...")
    fine_labels = load_fine_labels(FINE_LABELS_JSON)

    print("⚙️ Preparing output dataset...")
    output_data = prepare_output_data(fine_labels, filename_to_path, filename_to_exercise)

    print("🔀 Splitting dataset...")
    train_data, val_data, test_data = split_dataset(output_data)

    save_json(train_data, OUTPUT_TRAIN_JSON)
    save_json(val_data, OUTPUT_VAL_JSON)
    save_json(test_data, OUTPUT_TEST_JSON)

    print(f"\n{'='*60}")
    print("✅ Dataset Split Summary")
    print(f"Total videos: {len(output_data)}")
    print(f"Exercise classes: {len(set(filename_to_exercise.values()))}")
    print(f"Train: {len(train_data)} samples ({len(train_data)/len(output_data)*100:.1f}%)")
    print(f"Val:   {len(val_data)} samples ({len(val_data)/len(output_data)*100:.1f}%)")
    print(f"Test:  {len(test_data)} samples ({len(test_data)/len(output_data)*100:.1f}%)")
    print(f"Output files saved in {BASE_DIR}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
