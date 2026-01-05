#!/usr/bin/env python3
"""
Dataset Cleaning + Gemma3N-E2B Inference

Features:
- Checks video quality (resolution, brightness, sharpness, motion)
- Copies only good videos to a cleaned dataset
- Keeps folder structure
- Optionally runs Gemma3N-E2B inference on accepted videos
- Creates:
    - cleaning_report.csv
    - exercise_analysis_report.csv
    - rejected_videos.json
"""

import cv2
import numpy as np
import shutil
import pandas as pd
import os
from pathlib import Path
from tabulate import tabulate
from tqdm import tqdm
import json
import torch

# -------------------------------
# Dataset paths
# -------------------------------
DATASET_PATH = Path("dataset")
CLEANED_DATASET_PATH = Path("cleaned_dataset")

# -------------------------------
# Video quality thresholds
# -------------------------------
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

# Motion flags per exercise
MOTION_FLAGS_FILE = Path("utils/exercise_motion_overview.json")
if MOTION_FLAGS_FILE.exists():
    with open(MOTION_FLAGS_FILE, "r") as f:
        MOTION_FLAGS = json.load(f)
else:
    MOTION_FLAGS = {}

VIDEO_LOG = []
REJECTED_VIDEOS = []

# -------------------------------
# Helper functions
# -------------------------------
def ensure_directory_exists(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)

def copy_video_with_structure(source_file: Path, source_root: Path, destination_root: Path) -> None:
    relative_path = source_file.relative_to(source_root)
    destination_file = destination_root / relative_path
    ensure_directory_exists(destination_file.parent)
    shutil.copy2(source_file, destination_file)

def analyze_video_quality(video_path: Path, num_frames: int, frame_stride: int, exercise_name: str):
    """Analyze a video’s brightness, sharpness, and motion."""
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

    motion_flag = MOTION_FLAGS.get(exercise_name, False)

    while samples_collected < num_frames and frame_index < max(frame_count, num_frames * frame_stride + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            frame_index += frame_stride
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sampled_brightness.append(float(gray.mean()))
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        sampled_sharpness.append(float(lap.var()))

        if motion_flag:
            gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
            if prev_gray_for_motion is not None:
                diff = cv2.absdiff(gray_blur, prev_gray_for_motion)
                _, thresh = cv2.threshold(diff, MOTION_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
                ratio = np.count_nonzero(thresh) / thresh.size
                change_ratios.append(ratio)
                motion_pairs += 1
                if ratio >= MOTION_MIN_PIXEL_CHANGE_RATIO:
                    active_count += 1
            prev_gray_for_motion = gray_blur

        samples_collected += 1
        frame_index += frame_stride

    cap.release()

    if samples_collected == 0:
        issues.append("corrupted_file")

    motion_detected = False
    motion_active_frame_pct = float("nan")
    motion_mean_change_ratio = float("nan")
    motion_max_change_ratio = float("nan")

    if motion_flag and motion_pairs > 0:
        motion_active_frame_pct = active_count / motion_pairs
        motion_mean_change_ratio = float(np.mean(change_ratios)) if change_ratios else float("nan")
        motion_max_change_ratio = float(np.max(change_ratios)) if change_ratios else float("nan")
        motion_detected = motion_active_frame_pct >= MOTION_MIN_ACTIVE_FRAME_PCT

    metrics = {
        "width": width,
        "height": height,
        "mean_brightness": np.mean(sampled_brightness) if sampled_brightness else float("nan"),
        "sharpness_score": np.mean(sampled_sharpness) if sampled_sharpness else float("nan"),
        "motion_flag": motion_flag,
        "motion_detected": motion_detected,
        "motion_pairs": motion_pairs if motion_flag else 0,
        "motion_active_pairs": active_count if motion_flag else 0,
        "motion_active_frame_pct": motion_active_frame_pct,
        "motion_mean_change_ratio": motion_mean_change_ratio,
        "motion_max_change_ratio": motion_max_change_ratio,
    }
    return metrics, issues

def evaluate_video_acceptance(metrics: dict, issues: list, stats: dict):
    width, height = metrics.get("width", 0), metrics.get("height", 0)
    brightness = metrics.get("mean_brightness", float("nan"))
    sharpness = metrics.get("sharpness_score", float("nan"))
    motion_detected = metrics.get("motion_detected", False)
    motion_flag = metrics.get("motion_flag", False)

    reasons = []
    accepted = True

    if ("corrupted_file" in issues) or np.isnan(brightness) or np.isnan(sharpness):
        reasons.append("corrupted_file")
        return False, reasons

    if width < MIN_FRAME_WIDTH or height < MIN_FRAME_HEIGHT:
        reasons.append("low_resolution")
        accepted = False
    if brightness < MIN_BRIGHTNESS:
        reasons.append("too_dark")
        accepted = False
    elif brightness > MAX_BRIGHTNESS:
        reasons.append("too_bright")
        accepted = False
    if sharpness < MIN_SHARPNESS_SCORE:
        reasons.append("blurry")
        accepted = False
    if (not motion_detected) and motion_flag:
        reasons.append("insufficient_motion")
        accepted = False

    return accepted, reasons

def default_stats(exercise_name: str) -> dict:
    return {
        "Exercise": exercise_name,
        "total_videos": 0,
        "accepted_videos": 0,
        "rejected_videos": 0,
        "corrupted_files": 0,
        "low_resolution": 0,
        "too_dark": 0,
        "too_bright": 0,
        "blurry": 0,
        "insufficient_motion": 0,
    }

# -------------------------------
# Gemma3N-E2B inference
# -------------------------------
def run_gemma3n_inference(model, tokenizer, video_path: Path, num_frames=20):
    """Run Gemma3N-E2B inference on a sampled subset of frames."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, frame_count // num_frames)

    for i in range(0, frame_count, stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        frames.append(frame)
        if len(frames) >= num_frames:
            break
    cap.release()

    # Convert frames to PIL images for Gemma3N
    from PIL import Image
    pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]

    messages = [{"role":"user", "content":[{"type":"image", "image":img} for img in pil_frames] + [{"type":"text","text":"Describe the exercise."}]}]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")

    from transformers import TextStreamer
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    output = model.generate(**inputs, max_new_tokens=256, temperature=0.1, do_sample=True, streamer=streamer)
    return output

# -------------------------------
# Main cleaning loop
# -------------------------------
def clean_dataset(source_root: Path, destination_root: Path, run_inference=False):
    ensure_directory_exists(destination_root)
    video_extensions = (".mp4", ".avi", ".mov")

    # Load Gemma3N if requested
    if run_inference:
        from unsloth import FastModel
        model, tokenizer = FastModel.from_pretrained("google/gemma-3n-E2B")
        print("[INFO] Gemma3N-E2B model loaded for inference")
    else:
        model = tokenizer = None

    overall_stats = []
    totals = default_stats("ALL_EXERCISES")

    for root, _, files in os.walk(source_root):
        if not files:
            continue
        root_path = Path(root)
        folder_name = root_path.name
        stats = default_stats(folder_name)

        with tqdm(total=len(files), desc=f"Processing {folder_name}", ncols=80, leave=False) as pbar:
            for file in files:
                if not file.lower().endswith(video_extensions):
                    continue
                stats["total_videos"] += 1
                video_path = root_path / file

                metrics, issues = analyze_video_quality(video_path, NUM_SAMPLED_FRAMES, FRAME_STRIDE, folder_name)
                accepted, reasons = evaluate_video_acceptance(metrics, issues, stats)

                decision = "accepted" if accepted else "rejected"

                VIDEO_LOG.append({
                    "exercise": folder_name,
                    "file": file,
                    "width": metrics.get("width"),
                    "height": metrics.get("height"),
                    "brightness": metrics.get("mean_brightness"),
                    "sharpness": metrics.get("sharpness_score"),
                    "motion_flag": metrics.get("motion_flag"),
                    "motion_detected": metrics.get("motion_detected"),
                    "decision": decision,
                    "reasons": ", ".join(reasons) if reasons else "passed_all_checks"
                })

                if accepted:
                    copy_video_with_structure(video_path, source_root, destination_root)
                    stats["accepted_videos"] += 1
                    if run_inference:
                        result = run_gemma3n_inference(model, tokenizer, video_path)
                        tqdm.write(f"[Gemma3N Output] {file}: {result}")
                else:
                    stats["rejected_videos"] += 1
                    REJECTED_VIDEOS.append(str(video_path))
                pbar.update(1)

        overall_stats.append(stats)
        for key in totals:
            if key != "Exercise":
                totals[key] += stats[key]

    # Generate reports
    df_summary = pd.DataFrame(overall_stats)
    df_summary.loc[len(df_summary)] = totals
    df_summary.to_csv(destination_root / "cleaning_report.csv", index=False)
    pd.DataFrame(VIDEO_LOG).to_csv(destination_root / "exercise_analysis_report.csv", index=False)
    with open(destination_root / "rejected_videos.json", "w") as f:
        json.dump(REJECTED_VIDEOS, f, indent=2)

    print(f"\n[INFO] Dataset cleaning completed. Reports saved in {destination_root}")

# -------------------------------
# Main entry
# -------------------------------
if __name__ == "__main__":
    if not DATASET_PATH.exists():
        print(f"[ERROR] Dataset not found: {DATASET_PATH}")
    else:
        clean_dataset(DATASET_PATH, CLEANED_DATASET_PATH, run_inference=True)
