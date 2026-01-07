#!/usr/bin/env python3
"""
QVED Dataset Management for Gemma-3N-E2B

This script provides a unified interface for:
  1. Downloading videos from QVED dataset
  2. Creating train/val/test splits
  3. Cleaning dataset (quality filtering)
  4. Generating reports

Usage:
  python dataset.py download [--max-per-class N]
  python dataset.py prepare
  python dataset.py clean
  python dataset.py all
"""

import os
import sys
import random
import json
import shutil
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from huggingface_hub import list_repo_files, hf_hub_download
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ============================================
# Configuration
# ============================================
REPO_ID = "EdgeVLM-Labs/QVED-Test-Dataset"
LOCAL_DIR = Path("dataset")
MAX_PER_CLASS = 5
FILE_EXT = ".mp4"
GROUND_TRUTH_FILE = "fine_grained_labels.json"
RANDOM_SEED = 42

# Output files
MANIFEST_JSON = LOCAL_DIR / "manifest.json"
FINE_LABELS_JSON = LOCAL_DIR / GROUND_TRUTH_FILE  # Use fine_grained_labels.json
OUTPUT_TRAIN_JSON = LOCAL_DIR / "qved_train.json"
OUTPUT_VAL_JSON = LOCAL_DIR / "qved_val.json"
OUTPUT_TEST_JSON = LOCAL_DIR / "qved_test.json"

# Dataset split ratios
TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20

# Cleaning thresholds
MIN_FRAME_WIDTH = 640
MIN_FRAME_HEIGHT = 360
MIN_SHARPNESS_SCORE = 50
MIN_BRIGHTNESS = 35
MAX_BRIGHTNESS = 190
NUM_SAMPLED_FRAMES = 20
FRAME_STRIDE = 15
MOTION_DIFF_THRESHOLD = 18
MOTION_MIN_PIXEL_CHANGE_RATIO = 0.01
MOTION_MIN_ACTIVE_FRAME_PCT = 0.3

CLEANED_DATASET_PATH = Path("cleaned_dataset")
MOTION_FLAGS_FILE = Path("utils/exercise_motion_overview.json")

# User prompt template
USER_PROMPT_TEMPLATE = "Please evaluate the exercise form shown. What mistakes, if any, are present, and what corrections would you recommend?"


# ============================================
# Step 1: Download Dataset
# ============================================
def collect_videos(repo_id: str) -> Tuple[Dict[str, List[str]], List[str]]:
    """Collect all video files grouped by class (subfolder)."""
    print(f"📂 Listing repo files from: {repo_id}")
    all_files = list_repo_files(repo_id, repo_type="dataset")
    by_class = {}

    for f in all_files:
        if not f.endswith(FILE_EXT):
            continue
        parts = f.split("/")
        if len(parts) < 2:
            continue
        cls = parts[0]
        by_class.setdefault(cls, []).append(f)

    print(f"✅ Found {len(by_class)} classes with video files.")
    return by_class, all_files


def sample_and_download(by_class: Dict[str, List[str]], repo_id: str, 
                       local_dir: Path, max_per_class: int) -> Dict[str, str]:
    """Sample random videos per class and download them."""
    random.seed(RANDOM_SEED)
    manifest = {}
    total_downloaded = 0

    for cls, vids in by_class.items():
        class_dir = local_dir / cls
        class_dir.mkdir(parents=True, exist_ok=True)

        sample = random.sample(vids, min(len(vids), max_per_class))
        print(f"🎥 {cls}: {len(sample)} sampled of {len(vids)} available")

        for rel_path in sample:
            filename = os.path.basename(rel_path)
            target_path = class_dir / filename

            while True:
                try:
                    cached_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=rel_path,
                        repo_type="dataset",
                    )
                    shutil.copy2(cached_path, target_path)
                    manifest[str(target_path)] = cls
                    total_downloaded += 1
                    break
                except Exception as e:
                    if "429" in str(e) or "Too Many Requests" in str(e):
                        print(f"⚠️ Rate limit hit. Waiting ~3 minutes before retrying {rel_path}...")
                        time.sleep(200)
                    else:
                        print(f"⚠️ Failed to download {rel_path}: {e}")
                        break

    print(f"\n✅ Download complete: {total_downloaded} videos.")
    return manifest


def download_ground_truth(repo_id: str, local_dir: Path, all_files: List[str]) -> Path:
    """Download the fine_grained_labels.json ground truth file."""
    candidates = [f for f in all_files if f.endswith(GROUND_TRUTH_FILE)]
    if not candidates:
        print(f"⚠️ No {GROUND_TRUTH_FILE} found in repo.")
        return None

    gt_path = local_dir / GROUND_TRUTH_FILE
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=candidates[0],
            local_dir=str(local_dir),
            repo_type="dataset",
        )
        print(f"🧠 Ground truth file downloaded to: {gt_path}")
        return gt_path
    except Exception as e:
        print(f"⚠️ Failed to download {GROUND_TRUTH_FILE}: {e}")
        return None


def save_manifest(manifest: Dict[str, str], local_dir: Path) -> Path:
    """Save manifest.json mapping downloaded videos to their class."""
    manifest_path = local_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"📝 Manifest saved to: {manifest_path}")
    return manifest_path


def download_dataset(max_per_class: int = MAX_PER_CLASS):
    """Main function to download videos and ground truth."""
    print(f"📊 Using MAX_PER_CLASS = {max_per_class}")
    print(f"📁 Download directory: {LOCAL_DIR}")
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Collect videos by class
        by_class, all_files = collect_videos(REPO_ID)
    except Exception as e:
        print(f"\n❌ Failed to access HuggingFace repository: {e}")
        print("\n💡 Make sure you're logged in to HuggingFace:")
        print("   huggingface-cli login")
        print("   # OR")
        print("   hf auth login\n")
        sys.exit(1)

    # Step 2: Sample and download
    manifest = sample_and_download(by_class, REPO_ID, LOCAL_DIR, max_per_class)

    # Step 3: Save manifest
    save_manifest(manifest, LOCAL_DIR)

    # Step 4: Download ground truth
    gt_result = download_ground_truth(REPO_ID, LOCAL_DIR, all_files)
    
    if not gt_result:
        print("\n⚠️ Warning: Ground truth file not downloaded!")
        print("   You won't be able to run 'prepare' without it.")
    
    print("\n🏁 Dataset download completed.")
    print(f"\n✅ Next step: python dataset.py prepare")


# ============================================
# Step 2: Prepare Dataset (Train/Val/Test Splits)
# ============================================
def load_manifest(manifest_path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
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


def load_fine_labels(labels_path: Path) -> List[Dict]:
    """Load fine-grained labels JSON."""
    with open(labels_path, 'r') as f:
        return json.load(f)


def prepare_output_data(fine_labels: List[Dict], filename_to_path: Dict[str, str],
                       filename_to_exercise: Dict[str, str]) -> List[Dict]:
    """Convert raw labels into conversation format."""
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


def split_dataset(output_data: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
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


def save_json(data: List[Dict], output_path: Path):
    """Save data to JSON file."""
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def prepare_dataset():
    """Main function to prepare train/val/test splits."""
    # Check if required files exist
    if not MANIFEST_JSON.exists():
        print(f"❌ Error: {MANIFEST_JSON} not found.")
        print("\n💡 You need to download the dataset first:")
        print("   python dataset.py download\n")
        sys.exit(1)
    
    if not FINE_LABELS_JSON.exists():
        print(f"❌ Error: {FINE_LABELS_JSON} not found.")
        print("\n💡 You need to download the dataset first:")
        print("   python dataset.py download\n")
        sys.exit(1)
    
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
    print(f"Output files saved in {LOCAL_DIR}")
    print(f"{'='*60}")


# ============================================
# Step 3: Clean Dataset (Quality Filtering)
# ============================================
def analyze_video_quality(video_path: Path, num_frames: int, frame_stride: int, 
                          exercise_name: str) -> Tuple[Dict, List[str]]:
    """Analyze a video's brightness, sharpness, and motion."""
    issues = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {}, ["corrupted_file"]

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    sampled_brightness = []
    sampled_sharpness = []
    frame_index = 0
    samples_collected = 0

    prev_gray_for_motion = None
    active_count = 0
    motion_pairs = 0
    change_ratios = []

    # Load motion flags
    if MOTION_FLAGS_FILE.exists():
        with open(MOTION_FLAGS_FILE, "r") as f:
            motion_flags = json.load(f)
    else:
        motion_flags = {}
    
    motion_flag = motion_flags.get(exercise_name, False)

    while samples_collected < num_frames and frame_index < max(frame_count, num_frames * frame_stride + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            frame_index += frame_stride
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sampled_brightness.append(float(gray.mean()))
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = float(lap.var())
        sampled_sharpness.append(sharpness)

        # Motion detection
        if prev_gray_for_motion is not None and motion_flag:
            diff = cv2.absdiff(prev_gray_for_motion, gray)
            changed_pixels = np.sum(diff > MOTION_DIFF_THRESHOLD)
            total_pixels = diff.size
            change_ratio = changed_pixels / total_pixels
            change_ratios.append(change_ratio)
            motion_pairs += 1
            if change_ratio > MOTION_MIN_PIXEL_CHANGE_RATIO:
                active_count += 1

        prev_gray_for_motion = gray.copy()
        samples_collected += 1
        frame_index += frame_stride

    cap.release()

    # Check resolution
    if width < MIN_FRAME_WIDTH or height < MIN_FRAME_HEIGHT:
        issues.append("low_resolution")

    # Check brightness
    avg_brightness = np.mean(sampled_brightness) if sampled_brightness else 0
    if avg_brightness < MIN_BRIGHTNESS:
        issues.append("too_dark")
    elif avg_brightness > MAX_BRIGHTNESS:
        issues.append("too_bright")

    # Check sharpness
    avg_sharpness = np.mean(sampled_sharpness) if sampled_sharpness else 0
    if avg_sharpness < MIN_SHARPNESS_SCORE:
        issues.append("blurry")

    # Check motion
    active_frame_pct = (active_count / motion_pairs) if motion_pairs > 0 else 0
    if motion_flag and active_frame_pct < MOTION_MIN_ACTIVE_FRAME_PCT:
        issues.append("insufficient_motion")

    metrics = {
        "width": width,
        "height": height,
        "avg_brightness": avg_brightness,
        "avg_sharpness": avg_sharpness,
        "frame_count": frame_count,
        "active_frame_pct": active_frame_pct,
        "motion_pairs": motion_pairs,
        "active_count": active_count
    }

    return metrics, issues


def clean_dataset():
    """Main function to clean dataset by filtering low-quality videos."""
    print("🧹 Starting dataset cleaning...")
    
    if not LOCAL_DIR.exists():
        print(f"❌ Dataset path {LOCAL_DIR} does not exist.")
        print("\n💡 You need to download the dataset first:")
        print("   python dataset.py download\n")
        return

    CLEANED_DATASET_PATH.mkdir(parents=True, exist_ok=True)

    video_files = list(LOCAL_DIR.glob("**/*.mp4"))
    print(f"Found {len(video_files)} videos to analyze.")

    good_videos = []
    rejected_videos = []
    detailed_log = []

    for video_path in tqdm(video_files, desc="Analyzing videos"):
        exercise_name = video_path.parent.name
        metrics, issues = analyze_video_quality(
            video_path, NUM_SAMPLED_FRAMES, FRAME_STRIDE, exercise_name
        )

        log_entry = {
            "video": str(video_path),
            "exercise": exercise_name,
            **metrics,
            "issues": ", ".join(issues) if issues else "none"
        }
        detailed_log.append(log_entry)

        if not issues:
            good_videos.append(video_path)
            # Copy to cleaned dataset
            relative_path = video_path.relative_to(LOCAL_DIR)
            dest_path = CLEANED_DATASET_PATH / relative_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(video_path, dest_path)
        else:
            rejected_videos.append((video_path, issues))

    # Generate reports
    df = pd.DataFrame(detailed_log)
    df.to_csv("exercise_analysis_report.csv", index=False)
    print(f"📊 Detailed report saved to exercise_analysis_report.csv")

    summary = df.groupby("exercise").agg({
        "video": "count",
        "avg_brightness": "mean",
        "avg_sharpness": "mean"
    }).round(2)
    summary.to_csv("cleaning_report.csv")
    print(f"📋 Summary report saved to cleaning_report.csv")

    print(f"\n{'='*60}")
    print("✅ Dataset Cleaning Summary")
    print(f"Total videos analyzed: {len(video_files)}")
    print(f"Good videos: {len(good_videos)}")
    print(f"Rejected videos: {len(rejected_videos)}")
    print(f"Cleaned dataset saved to: {CLEANED_DATASET_PATH}")
    print(f"{'='*60}")


# ============================================
# Step 4: Copy Videos for Inference
# ============================================
def copy_for_inference(num_videos: int = 5, output_folder: str = "videos"):
    """
    Copy N random videos from dataset to a flat folder for batch inference.
    
    Args:
        num_videos: Number of videos to copy (default: 5)
        output_folder: Destination folder name (default: videos)
    """
    print(f"\n{'='*60}")
    print(f"📋 Copying {num_videos} videos for batch inference")
    print(f"{'='*60}\n")
    
    # Check if dataset exists
    if not LOCAL_DIR.exists():
        print(f"❌ Error: Dataset folder '{LOCAL_DIR}' not found.")
        print("   Please run 'python dataset.py download' first.")
        return
    
    # Collect all video files
    video_files = []
    for video_path in LOCAL_DIR.rglob(f"*{FILE_EXT}"):
        if video_path.is_file():
            video_files.append(video_path)
    
    if not video_files:
        print(f"❌ Error: No video files found in '{LOCAL_DIR}'")
        return
    
    print(f"✅ Found {len(video_files)} videos in dataset")
    
    # Sample random videos
    num_to_copy = min(num_videos, len(video_files))
    random.seed(RANDOM_SEED)
    selected_videos = random.sample(video_files, num_to_copy)
    
    # Create output folder
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Copy videos
    print(f"📂 Copying to: {output_path}/\n")
    copied = []
    
    for i, video_src in enumerate(selected_videos, 1):
        # Create unique filename: classname_originalname.mp4
        class_name = video_src.parent.name
        video_dst = output_path / f"{class_name}_{video_src.name}"
        
        try:
            shutil.copy2(video_src, video_dst)
            copied.append(str(video_dst))
            print(f"  [{i}/{num_to_copy}] ✓ {video_src.name} → {video_dst.name}")
        except Exception as e:
            print(f"  [{i}/{num_to_copy}] ✗ Failed to copy {video_src.name}: {e}")
    
    print(f"\n{'='*60}")
    print(f"✅ Copied {len(copied)}/{num_to_copy} videos to '{output_path}/'")
    print(f"\nRun batch inference with:")
    print(f"  python gemma3n_batch_inference.py \\")
    print(f"    --video_folder {output_folder} \\")
    print(f"    --output results/batch_results.csv")
    print(f"{'='*60}")


# ============================================
# Main CLI
# ============================================
def main():
    parser = argparse.ArgumentParser(
        description="QVED Dataset Management for Gemma-3N-E2B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dataset.py download                    # Download 5 videos per class
  python dataset.py download --max-per-class 10
  python dataset.py prepare                     # Create train/val/test splits
  python dataset.py clean                       # Filter low-quality videos
  python dataset.py copy                        # Copy 5 random videos to videos/
  python dataset.py copy --num-videos 10        # Copy 10 videos
  python dataset.py copy --output my_videos     # Copy to custom folder
  python dataset.py all                         # Run all steps
        """
    )
    
    parser.add_argument(
        "command",
        choices=["download", "prepare", "clean", "copy", "all"],
        help="Command to execute"
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=MAX_PER_CLASS,
        help=f"Maximum videos per class for download (default: {MAX_PER_CLASS})"
    )
    parser.add_argument(
        "--num-videos",
        type=int,
        default=5,
        help="Number of videos to copy for inference (default: 5)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="videos",
        help="Output folder for copied videos (default: videos)"
    )

    args = parser.parse_args()

    if args.command == "download":
        download_dataset(args.max_per_class)
    elif args.command == "prepare":
        prepare_dataset()
    elif args.command == "clean":
        clean_dataset()
    elif args.command == "copy":
        copy_for_inference(args.num_videos, args.output)
    elif args.command == "all":
        download_dataset(args.max_per_class)
        prepare_dataset()
        clean_dataset()


if __name__ == "__main__":
    main()
