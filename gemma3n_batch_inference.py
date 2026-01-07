"""
Gemma-3N Batch Video Inference Script
Processes multiple videos and saves results to CSV
"""
import os
import warnings
import logging
import argparse
import csv
from pathlib import Path
from typing import List, Optional

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

logging.getLogger("mmengine").setLevel(logging.CRITICAL)
logging.getLogger("transformers").setLevel(logging.CRITICAL)
logging.getLogger("transformers.modeling_utils").setLevel(logging.CRITICAL)

import torch
import cv2
from PIL import Image
from unsloth import FastModel
from transformers import TextStreamer


def extract_frames(video_path: str, num_frames: int = 8) -> List[Image.Image]:
    """
    Extract evenly-spaced frames from a video file.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
    
    Returns:
        List of PIL Image objects
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"  ⚠️  Error: Could not open video file {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        print(f"  ⚠️  Error: Video has no frames {video_path}")
        return []
    
    # Calculate the step size
    step = total_frames // num_frames
    frames = []

    for i in range(num_frames):
        frame_idx = i * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR (OpenCV) to RGB (PIL)
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frames.append(img)

    cap.release()
    return frames


def load_model(model_name: str = "unsloth/gemma-3n-E4B-it", device: str = "cuda"):
    """Load Gemma-3N model using Unsloth FastModel."""
    print(f"Loading model: {model_name}...")
    
    torch._dynamo.config.recompile_limit = 64
    
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        dtype=None,  # None for auto detection
        max_seq_length=1024,
        load_in_4bit=False,
        full_finetuning=False,
        # token="hf_...",  # use one if using gated models
    )
    
    model.to(device)
    print("✓ Model loaded successfully!")
    return model, tokenizer


def do_inference(
    model,
    tokenizer,
    messages: List[dict],
    max_new_tokens: int = 128,
    show_stream: bool = False
) -> str:
    """
    Run inference on Gemma-3N model with chat messages.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        messages: List of message dicts with role/content
        max_new_tokens: Maximum tokens to generate
        show_stream: Whether to show streaming output
    
    Returns:
        Generated response text
    """
    # 1. Prepare inputs
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")

    # 2. Generate
    streamer = TextStreamer(tokenizer, skip_prompt=True) if show_stream else None
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.1,
        do_sample=True,
        streamer=streamer,
    )

    # 3. Decode only the NEW tokens (the answer)
    input_length = inputs.input_ids.shape[1]
    new_tokens = outputs[0][input_length:]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return response_text


def process_videos(
    model,
    tokenizer,
    video_folder: str,
    output_csv: str,
    prompt: str,
    num_frames: int = 8,
    max_new_tokens: int = 256,
    show_stream: bool = False
):
    """
    Process all videos in a folder and save results to CSV.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        video_folder: Path to folder containing videos
        output_csv: Path to output CSV file
        prompt: Text prompt for inference
        num_frames: Number of frames to extract per video
        max_new_tokens: Maximum tokens to generate
        show_stream: Whether to show streaming output
    """
    results = []
    
    # Get list of all video files
    video_path = Path(video_folder)
    if not video_path.exists():
        print(f"❌ Error: Video folder not found: {video_folder}")
        return
    
    video_files = [f for f in video_path.iterdir() if f.suffix.lower() in ['.mp4', '.mov', '.avi', '.mkv']]
    
    if not video_files:
        print(f"❌ Error: No video files found in {video_folder}")
        return
    
    print(f"\n{'='*60}")
    print(f"Found {len(video_files)} videos. Starting processing...")
    print(f"{'='*60}\n")

    for idx, video_file in enumerate(video_files, 1):
        print(f"[{idx}/{len(video_files)}] Processing: {video_file.name}")
        
        # 1. Extract frames
        video_frames = extract_frames(str(video_file), num_frames=num_frames)
        
        if not video_frames:
            print(f"  ⚠️  Skipping - could not extract frames\n")
            results.append([video_file.name, "ERROR: Could not extract frames"])
            continue

        # 2. Build the message
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": img} for img in video_frames],
                    {"type": "text", "text": prompt}
                ],
            },
        ]

        # 3. Inference
        try:
            response = do_inference(
                model,
                tokenizer,
                messages,
                max_new_tokens=max_new_tokens,
                show_stream=show_stream
            )
            results.append([video_file.name, response])
            print(f"  ✓ Completed\n")
        except Exception as e:
            print(f"  ❌ Error: {e}\n")
            results.append([video_file.name, f"ERROR: {str(e)}"])

    # Save to CSV
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_name", "model_output"])
        writer.writerows(results)

    print(f"\n{'='*60}")
    print(f"✅ Success! Processed {len(results)} videos")
    print(f"📄 Results saved to: {output_csv}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Batch video inference with Gemma-3N for assistive navigation"
    )
    parser.add_argument(
        "--model",
        default="unsloth/gemma-3n-E4B-it",
        help="Model name or path (default: unsloth/gemma-3n-E4B-it)"
    )
    parser.add_argument(
        "--video_folder",
        default="sample_videos",
        help="Path to folder containing videos (default: sample_videos)"
    )
    parser.add_argument(
        "--output",
        default="results/gemma3n_batch_inference_results.csv",
        help="Output CSV file path (default: results/gemma3n_batch_inference_results.csv)"
    )
    parser.add_argument(
        "--prompt",
        default=(
            "You are an assistive navigation system for a visually impaired user. "
            "Analyze the provided video from the user's forward perspective. "
            "Identify all the immediate, high-risk obstructions. "
            "State the obstruction's location using the 12-hour clock face. "
            "Process the provided video generate a single, actionable safety alert."
        ),
        help="Prompt text for inference"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=8,
        help="Number of frames to extract per video (default: 8)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate (default: 256)"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to use: cuda or cpu (default: cuda)"
    )
    parser.add_argument(
        "--show_stream",
        action="store_true",
        help="Show streaming output during inference"
    )
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.model, args.device)
    
    # Process videos
    process_videos(
        model=model,
        tokenizer=tokenizer,
        video_folder=args.video_folder,
        output_csv=args.output,
        prompt=args.prompt,
        num_frames=args.num_frames,
        max_new_tokens=args.max_new_tokens,
        show_stream=args.show_stream
    )


if __name__ == "__main__":
    main()
