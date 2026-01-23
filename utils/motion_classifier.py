#!/usr/bin/env python3
"""
Exercise Motion Analyzer

Features:
- Detects motion in exercise videos
- Randomly samples N videos per exercise
- Saves results to CSV and JSON files
- Configurable dataset/output paths (compatible with gemma-3n-E2B pipelines)
"""

import os
import random
import csv
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone

import cv2
import numpy as np
from tqdm import tqdm

# ================= Configuration =================
ROOT_DIR = Path("dataset")  # Root folder containing exercise videos
OUTPUT_DIR = Path(__file__).parent / "exercise_motion_reports"
CSV_FILENAME = "exercise_motion_report.csv"
JSON_FILENAME = "dataset_motion_overview.json"

CSV_PATH = OUTPUT_DIR / CSV_FILENAME
JSON_PATH = OUTPUT_DIR / JSON_FILENAME

# Sampling & Motion Parameters
N = 30  # Number of videos per exercise to sample
SEED = 42

NUM_FRAMES = 15
FRAME_STRIDE = 15
MOTION_DIFF_THRESHOLD = 20
MOTION_MIN_PIXEL_CHANGE_RATIO = 0.01
MOTION_MIN_ACTIVE_FRAME_PCT = 0.3
EXERCISE_MOTION_THRESHOLD = 0.7
# =================================================


def analyze_motion(video_path: Path, num_frames: int, frame_stride: int) -> bool:
    """
    Analyze motion in a video.

    Args:
        video_path: Path to the video file
        num_frames: Number of frames to sample
        frame_stride: Frame interval between sampled frames

    Returns:
        True if motion is detected, else False
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("corrupted_file")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    prev_gray = None
    active_count = 0
    motion_pairs = 0
    samples_collected = 0
    frame_index = 0
    limit = max(frame_count, num_frames * frame_stride + 1)

    while samples_collected < num_frames and frame_index < limit:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            frame_index += frame_stride
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

        if prev_gray is not None:
            diff = cv2.absdiff(gray_blur, prev_gray)
            _, thresh = cv2.threshold(diff, MOTION_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
            ratio = np.count_nonzero(thresh) / float(thresh.size)
            motion_pairs += 1
            if ratio >= MOTION_MIN_PIXEL_CHANGE_RATIO:
                active_count += 1

        prev_gray = gray_blur
        samples_collected += 1
        frame_index += frame_stride

    cap.release()

    if motion_pairs == 0:
        raise RuntimeError("no_valid_frames")

    return (active_count / motion_pairs) >= MOTION_MIN_ACTIVE_FRAME_PCT


def list_videos_by_exercise(root: Path) -> dict:
    """
    Scan the dataset and list videos by exercise folder.

    Returns:
        Dictionary {exercise_name: [list of video paths]}
    """
    by_exercise = defaultdict(list)
    print(f"\nScanning for videos in {root} ...")
    for dirpath, _, filenames in os.walk(root):
        exercise = Path(dirpath).name
        mp4_files = [fn for fn in filenames if fn.lower().endswith(".mp4")]
        if mp4_files:
            by_exercise[exercise].extend(str(Path(dirpath) / fn) for fn in mp4_files)

    print(f"\nTotal exercises found: {len(by_exercise)}\n")
    return by_exercise


def main(dataset_path: str = None, output_path: str = None) -> None:
    """
    Run motion analysis and save results to CSV and JSON.

    Args:
        dataset_path: Optional custom dataset folder
        output_path: Optional custom output folder
    """
    root_dir = Path(dataset_path) if dataset_path else ROOT_DIR
    output_dir = Path(output_path) if output_path else OUTPUT_DIR

    if not root_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {root_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)

    # List and sample videos
    exercises = list_videos_by_exercise(root_dir)
    if not exercises:
        print("No videos found in dataset.")
        return

    sampled = {}
    for ex, vids in exercises.items():
        if vids:
            k = min(len(vids), N)
            sampled[ex] = random.sample(vids, k)
            print(f"Sampling {k} of {len(vids)} from {ex}")

    # Analyze motion
    per_ex_counts = defaultdict(lambda: {"motion": 0, "no_motion": 0})
    all_items = [(ex, vp) for ex, vids in sampled.items() for vp in vids]

    print("\nStarting motion analysis...\n")
    for ex, video_path in tqdm(all_items, desc="Analyzing videos", unit="video"):
        try:
            motion = analyze_motion(Path(video_path), NUM_FRAMES, FRAME_STRIDE)
        except Exception as e:
            tqdm.write(f"[WARN] Skipping {video_path} (error: {e})")
            continue
        if motion:
            per_ex_counts[ex]["motion"] += 1
        else:
            per_ex_counts[ex]["no_motion"] += 1

    # Generate CSV summary
    summary_rows = [("exercise", "motion_videos", "no_motion_videos", "total_videos", "motion_fraction", "is_motion_exercise")]
    exercise_flags = {}
    print("\nPer-exercise summary:")
    for ex in sorted(per_ex_counts.keys()):
        m = per_ex_counts[ex]["motion"]
        nm = per_ex_counts[ex]["no_motion"]
        total = m + nm
        motion_fraction = m / total if total > 0 else 0.0
        is_motion = motion_fraction >= EXERCISE_MOTION_THRESHOLD
        exercise_flags[ex] = is_motion
        summary_rows.append((ex, m, nm, total, f"{motion_fraction:.2f}", is_motion))
        print(f"{ex:20s} motion={m:<3} non_motion={nm:<3} fraction={motion_fraction:.2f} is_motion={is_motion}")

    # Save CSV
    csv_path = output_dir / CSV_FILENAME
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(summary_rows)
    print(f"\nCSV saved to {csv_path}")

    # Save JSON
    json_path = output_dir / JSON_FILENAME
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(exercise_flags, jf, ensure_ascii=False, indent=2)
    print(f"JSON saved to {json_path}")


if __name__ == "__main__":
    main()
