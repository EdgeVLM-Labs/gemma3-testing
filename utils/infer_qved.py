#!/usr/bin/env python3
"""
QVED Inference with Google/Gemma-3n-E2B (Unsloth)

This script runs inference on a single video using the Google/Gemma-3n-E2B
model via Unsloth FastModel.

Usage:
    python infer_qved.py \
        --video_path sample_videos/00000340.mp4 \
        --prompt "Describe this video"
"""

import os
import warnings
import logging
import argparse
from typing import List

os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings("ignore")

logging.getLogger('mmengine').setLevel(logging.CRITICAL)
logging.getLogger('transformers').setLevel(logging.CRITICAL)
logging.getLogger('transformers.modeling_utils').setLevel(logging.CRITICAL)

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
    print(f"📦 Loading model: {model_name}...")
    
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
    print("✅ Model loaded successfully!")
    return model, tokenizer


def run_inference(
    model,
    tokenizer,
    video_path: str,
    prompt: str,
    device: str = "cuda",
    num_frames: int = 8,
    max_new_tokens: int = 512,
    show_stream: bool = False
) -> str:
    """
    Run inference on a video file.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        video_path: Path to video file
        prompt: Text prompt for inference
        device: Device to use (cuda/cpu)
        num_frames: Number of frames to extract
        max_new_tokens: Maximum tokens to generate
        show_stream: Whether to show streaming output
    
    Returns:
        Generated response text
    """
    # Extract frames
    video_frames = extract_frames(video_path, num_frames=num_frames)
    
    if not video_frames:
        return "ERROR: Could not extract frames from video"

    # Build the message
    messages = [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": img} for img in video_frames],
                {"type": "text", "text": prompt}
            ],
        },
    ]

    # Try to use chat template if available, otherwise use direct processing
    try:
        # Check if tokenizer has chat template
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device)
        else:
            raise ValueError("No chat template available")
    except (ValueError, AttributeError):
        # Fallback: direct processing with images
        print("⚠️  Chat template not available, using direct processing...")
        text_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # Process with images
        inputs = tokenizer(
            text=text_prompt,
            images=video_frames,
            return_tensors="pt",
            padding=True,
        )
        # Move all tensors to device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # Generate
    streamer = TextStreamer(tokenizer, skip_prompt=True) if show_stream else None
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.1,
        do_sample=True,
        streamer=streamer,
    )

    # Decode only the NEW tokens (the answer)
    if 'input_ids' in inputs:
        input_length = inputs['input_ids'].shape[1]
    elif hasattr(inputs, 'input_ids'):
        input_length = inputs.input_ids.shape[1]
    else:
        input_length = 0
        
    new_tokens = outputs[0][input_length:]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return response_text


def main():
    parser = argparse.ArgumentParser(
        description="QVED Inference with Google/Gemma-3n-E2B (Unsloth)",
        epilog="Note: For best results, use 'unsloth/gemma-3n-E4B-it' which includes proper chat templates."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="unsloth/gemma-3n-E4B-it",
        help="Path to the model (HuggingFace model ID). Recommended: unsloth/gemma-3n-E4B-it or unsloth/gemma-3n-E2B-it"
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="sample_videos/00000340.mp4",
        help="Path to the input video file"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Please evaluate the exercise form shown. What mistakes, if any, are present, and what corrections would you recommend?",
        help="Prompt for the model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (cuda/cpu)"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=8,
        help="Number of frames to extract from video"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--show_stream",
        action="store_true",
        help="Show streaming output during inference"
    )

    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"❌ Error: Video file not found: {args.video_path}")
        return

    # Load model
    model, tokenizer = load_model(args.model_path, device=args.device)

    print(f"🎥 Processing video: {args.video_path}")
    print(f"💬 Prompt: {args.prompt}")
    print("\n" + "="*80)

    # Run inference
    output = run_inference(
        model,
        tokenizer,
        args.video_path,
        args.prompt,
        device=args.device,
        num_frames=args.num_frames,
        max_new_tokens=args.max_new_tokens,
        show_stream=args.show_stream
    )

    print("🤖 Gemma-3N Output:")
    print(output)
    print("="*80)


if __name__ == "__main__":
    main()
