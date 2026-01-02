"""
Dataset Conversion Utility for Gemma 3n Training

Converts QVED fine-grained labels to Gemma 3n training format.
Creates train/val/test splits with proper conversation structure.

Format:
{
    "video_path": "exercise_name/video.mp4",
    "prompt": "User instruction...",
    "response": "Assistant answer..."
}

Usage:
    python -m utils.convert_dataset
"""

import json
import os
import random
import logging
from pathlib import Path
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path("dataset")
FINE_LABELS_JSON = BASE_DIR / "ground_truth.json"
MANIFEST_JSON = BASE_DIR / "manifest.json"
OUTPUT_TRAIN_JSON = BASE_DIR / "gemma_train.jsonl"
OUTPUT_VAL_JSON = BASE_DIR / "gemma_val.jsonl"
OUTPUT_TEST_JSON = BASE_DIR / "gemma_test.jsonl"

# Prompt template for exercise assessment
USER_PROMPT_TEMPLATE = "Please evaluate the exercise form shown. What mistakes, if any, are present, and what corrections would you recommend?"

# Dataset split ratios
TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20
RANDOM_SEED = 42


def convert_to_gemma_format(record: dict, full_video_path: str, exercise: str) -> dict:
    """
    Convert a single record to Gemma 3n training format.

    Args:
        record: Original fine-grained label record
        full_video_path: Full path to video file
        exercise: Exercise class name

    Returns:
        Dictionary in Gemma format with video_path, prompt, and response
    """
    # Remove 'dataset/' prefix if present to make path relative to data_path
    if full_video_path.startswith('dataset/'):
        relative_video_path = full_video_path[len('dataset/'):]
    else:
        relative_video_path = full_video_path

    # Get assistant answer from most descriptive label
    if 'labels_descriptive' in record and record['labels_descriptive']:
        assistant_answer = record['labels_descriptive']
    elif 'labels' in record and record['labels']:
        assistant_answer = record['labels'][0] if isinstance(record['labels'], list) else record['labels']
    else:
        assistant_answer = "No feedback available."

    # Ensure assistant answer is a single string
    if isinstance(assistant_answer, list):
        assistant_answer = '\n'.join(str(item) for item in assistant_answer)
    else:
        assistant_answer = str(assistant_answer)

    return {
        "video_path": relative_video_path,
        "prompt": USER_PROMPT_TEMPLATE,
        "response": assistant_answer
    }


def main():
    """
    Convert QVED dataset to Gemma 3n format and split into train/val/test.
    """
    # Validate required files exist
    if not FINE_LABELS_JSON.exists() or not MANIFEST_JSON.exists():
        logger.error("⚠️  Required files missing. Please run load_dataset.py and filter_ground_truth.py first.")
        return

    # Load manifest to map video filenames to full paths
    logger.info("📂 Loading manifest...")
    with open(MANIFEST_JSON, 'r') as f:
        manifest = json.load(f)

    # Create reverse lookup: filename -> full_path and exercise
    filename_to_path = {}
    filename_to_exercise = {}
    for full_path, exercise in manifest.items():
        filename = os.path.basename(full_path)
        filename_to_path[filename] = full_path

        # Handle both string and dict values in manifest
        if isinstance(exercise, str):
            filename_to_exercise[filename] = exercise.replace('_', ' ')
        elif isinstance(exercise, dict):
            # For dict entries, extract exercise from path
            exercise_name = full_path.split('/')[0] if '/' in full_path else 'unknown'
            filename_to_exercise[filename] = exercise_name.replace('_', ' ')
        else:
            filename_to_exercise[filename] = str(exercise).replace('_', ' ')

    # Load fine-grained labels
    logger.info("🧠 Loading fine-grained labels...")
    with open(FINE_LABELS_JSON, 'r') as f:
        fine_labels = json.load(f)
    logger.info(f"📋 Found {len(fine_labels)} labeled entries")

    # Convert to Gemma format
    logger.info("🔄 Converting to Gemma format...")
    output_data = []

    for record in fine_labels:
        video_path = record.get('video_path', '')
        filename = os.path.basename(video_path)

        # Look up full path in manifest
        if filename not in filename_to_path:
            logger.warning(f"⚠️  {filename} not found in manifest, skipping")
            continue

        full_video_path = filename_to_path[filename]
        exercise = filename_to_exercise[filename]

        # Convert to Gemma format
        gemma_record = convert_to_gemma_format(record, full_video_path, exercise)
        output_data.append(gemma_record)

    logger.info(f"✅ Converted {len(output_data)} records")

    # Shuffle data for random split
    random.seed(RANDOM_SEED)
    random.shuffle(output_data)

    # Calculate split indices
    total_count = len(output_data)
    train_end = int(total_count * TRAIN_RATIO)
    val_end = train_end + int(total_count * VAL_RATIO)

    # Split the data
    train_data = output_data[:train_end]
    val_data = output_data[train_end:val_end]
    test_data = output_data[val_end:]

    # Write output JSONL files (one JSON object per line)
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("💾 Saving split datasets...")

    with open(OUTPUT_TRAIN_JSON, 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')

    with open(OUTPUT_VAL_JSON, 'w') as f:
        for item in val_data:
            f.write(json.dumps(item) + '\n')

    with open(OUTPUT_TEST_JSON, 'w') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Dataset Split Summary")
    logger.info(f"{'='*60}")
    logger.info(f"Total videos: {total_count}")
    logger.info(f"Exercise classes: {len(set(filename_to_exercise.values()))}")
    logger.info(f"\nSplit Distribution:")
    logger.info(f"  Train: {len(train_data)} samples ({len(train_data)/total_count*100:.1f}%)")
    logger.info(f"  Val:   {len(val_data)} samples ({len(val_data)/total_count*100:.1f}%)")
    logger.info(f"  Test:  {len(test_data)} samples ({len(test_data)/total_count*100:.1f}%)")
    logger.info(f"\nOutput files:")
    logger.info(f"  Train: {OUTPUT_TRAIN_JSON}")
    logger.info(f"  Val:   {OUTPUT_VAL_JSON}")
    logger.info(f"  Test:  {OUTPUT_TEST_JSON}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
