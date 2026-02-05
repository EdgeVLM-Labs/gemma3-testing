#!/usr/bin/env python3
"""
Video Augmentation & Gemma3N-E2B Inference Script

This script:
1. Applies video augmentations using vidaug.
2. Saves augmented videos.
3. Optionally runs Gemma3N-E2B inference on augmented frames.
4. Updates JSON files to reflect new augmented videos.

Usage:
    python augment_videos.py \
        --dataset_dir dataset \
        --folders 1,3 \
        --augmentations 1,3,5 \
        --run_inference
"""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

# Fix for vidaug compatibility with NumPy >= 1.20
# vidaug uses deprecated np.float, np.int aliases
import numpy as np
if not hasattr(np, 'float'):
    np.float = np.float64
if not hasattr(np, 'int'):
    np.int = np.int_
if not hasattr(np, 'bool'):
    np.bool = np.bool_

import vidaug.augmentors as va
from unsloth import FastModel
import torch

# ----------------------------
# Augmentation Options
# ----------------------------
# Note: Some augmentations may fail with certain parameter combinations
# due to vidaug compatibility issues with newer NumPy versions
AUGMENTATION_OPTIONS = {
    1: ("Horizontal Flip", va.HorizontalFlip()),
    2: ("Vertical Flip", va.VerticalFlip()),
    3: ("Random Rotate (±10°)", va.RandomRotate(degrees=10)),
    # 4: ("Random Resize (±20%)", va.RandomResize(rate=0.2)),  # Disabled - NumPy compatibility issue
    5: ("Gaussian Blur", va.GaussianBlur(sigma=1.5)),
    6: ("Add Brightness (+30)", va.Add(value=30)),
    7: ("Multiply Brightness (1.2x)", va.Multiply(value=1.2)),
    # 8: ("Random Translate (±15px)", va.RandomTranslate(x=15, y=15)),  # Disabled - NumPy compatibility issue
    # 9: ("Random Shear", va.RandomShear(x=0.1, y=0.1)),  # Disabled - NumPy compatibility issue
    10: ("Invert Color", va.InvertColor()),
    11: ("Salt Noise", va.Salt(ratio=100)),
    12: ("Pepper Noise", va.Pepper(ratio=100)),
    # 13: ("Temporal Downsample (0.8x)", va.Downsample(ratio=0.8)),  # Disabled - NumPy compatibility issue
    # 14: ("Elastic Transformation", va.ElasticTransformation(alpha=10, sigma=3)),  # Disabled - NumPy compatibility issue
}

# ----------------------------
# Helper Functions
# ----------------------------
def load_video_frames(video_path):
    """Load video frames as PIL Images."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames

def save_video_frames(frames, output_path, fps=30):
    """Save PIL frames as a video file."""
    if not frames:
        return False
    first_frame = np.array(frames[0])
    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    for frame in frames:
        out.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
    out.release()
    return True

def get_video_fps(video_path):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 30

def augment_video(video_path, augmentor, output_path):
    """Apply augmentation to video and save the result."""
    frames = load_video_frames(video_path)
    if not frames:
        print(f"❌ Failed to load frames for {video_path.name}")
        return False
    try:
        augmented_frames = augmentor(frames)
        # Ensure frames are valid after augmentation
        if not augmented_frames or len(augmented_frames) == 0:
            print(f"❌ Augmentation produced no frames")
            return False
    except Exception as e:
        print(f"❌ Augmentation failed: {e}")
        return False
    
    fps = get_video_fps(video_path)
    success = save_video_frames(augmented_frames, output_path, fps)
    if not success:
        print(f"❌ Failed to save augmented video")
    return success

def update_json_files(augmented_videos_info, base_dir, update_manifest=True):
    """Update JSON files with augmented video info."""
    ground_truth_file = base_dir / "fine_grained_labels.json"
    manifest_file = base_dir / "manifest.json"
    output_ground_truth_file = base_dir / "ground_truth.json"

    with open(ground_truth_file, 'r') as f:
        fine_grained_labels = json.load(f)
    if manifest_file.exists() and update_manifest:
        with open(manifest_file, 'r') as f:
            manifest = json.load(f)
    else:
        manifest = {}

    if output_ground_truth_file.exists():
        with open(output_ground_truth_file, 'r') as f:
            ground_truth = json.load(f)
    else:
        ground_truth = {}

    for aug_info in augmented_videos_info:
        orig, aug = aug_info['original_path'], aug_info['augmented_path']
        if orig in fine_grained_labels:
            fine_grained_labels[aug] = fine_grained_labels[orig].copy()
        if orig in manifest:
            manifest[aug] = manifest[orig]
        else:
            manifest[aug] = aug.split('/')[0]
        if orig in ground_truth:
            ground_truth[aug] = ground_truth[orig].copy()

    with open(ground_truth_file, 'w') as f:
        json.dump(fine_grained_labels, f, indent=2)
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    if ground_truth:
        with open(output_ground_truth_file, 'w') as f:
            json.dump(ground_truth, f, indent=2)

# ----------------------------
# Gemma3N Inference
# ----------------------------
def run_gemma3n_inference(model, tokenizer, frames, prompt="Analyze the video."):
    """Run inference on list of PIL frames."""
    from transformers import TextStreamer

    messages = [{"role":"user", "content":[{"type":"image", "image":img} for img in frames] + [{"type":"text","text":prompt}]}]
    
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
    
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.1, do_sample=True, streamer=streamer)
    return outputs

# ----------------------------
# Main Function
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Video Augmentation Tool for Gemma3N-E2B")
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="Dataset base directory")
    parser.add_argument("--folders", type=str, default=None, help="Comma-separated folder indices to augment")
    parser.add_argument("--augmentations", type=str, default=None, help="Comma-separated augmentation indices")
    parser.add_argument("--run_inference", action="store_true", help="Run Gemma3N-E2B inference on augmented videos")
    args = parser.parse_args()

    base_dir = Path(args.dataset_dir)
    exercise_folders = sorted([d for d in base_dir.iterdir() if d.is_dir() and any(d.glob("*.mp4"))])
    if not exercise_folders:
        print("❌ No exercise folders found")
        return

    if args.folders:
        try:
            indices = [int(x)-1 for x in args.folders.split(',')]
            selected_folders = [exercise_folders[i] for i in indices if 0 <= i < len(exercise_folders)]
        except Exception:
            print("❌ Invalid folder indices")
            return
    else:
        selected_folders = exercise_folders

    if args.augmentations:
        try:
            aug_indices = [int(x) for x in args.augmentations.split(',') if int(x) in AUGMENTATION_OPTIONS]
        except Exception:
            print("❌ Invalid augmentation indices")
            return
    else:
        aug_indices = list(AUGMENTATION_OPTIONS.keys())

    # Load Gemma3N model if requested
    if args.run_inference:
        model, tokenizer = FastModel.from_pretrained("unsloth/gemma-3n-E2B")
        print("✅ Gemma3N-E2B loaded for inference")
    else:
        model = tokenizer = None

    all_augmented_videos = []

    for folder in selected_folders:
        videos = sorted(folder.glob("*.mp4"))
        for video_path in videos:
            for idx in aug_indices:
                aug_name, augor = AUGMENTATION_OPTIONS[idx]
                seq = va.Sequential([augor])
                output_file = folder / f"{video_path.stem}_aug{idx}.mp4"
                success = augment_video(video_path, seq, output_file)
                if success:
                    all_augmented_videos.append({
                        "original_path": str(Path(folder.name) / video_path.name),
                        "augmented_path": str(Path(folder.name) / output_file.name)
                    })
                    # Optionally run Gemma3N inference
                    if args.run_inference:
                        frames = load_video_frames(output_file)
                        result = run_gemma3n_inference(model, tokenizer, frames)
                        print(f"🎯 Gemma3N Output ({output_file.name}): {result}")

    if all_augmented_videos:
        update_json_files(all_augmented_videos, base_dir)
        print(f"\n✓ Augmentation complete! {len(all_augmented_videos)} videos created.")

if __name__ == "__main__":
    main()
