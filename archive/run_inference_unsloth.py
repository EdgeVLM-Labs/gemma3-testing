#!/usr/bin/env python3
"""
Inference script for fine-tuned Gemma-3N model

Usage:
    python scripts/run_inference_unsloth.py \
        --model_path outputs/gemma3n_finetune_YYYYMMDD_HHMMSS_merged_16bit \
        --video_path sample_videos/test.mp4 \
        --num_frames 8
"""

import argparse
import cv2
import numpy as np
from PIL import Image
import torch
from unsloth import FastVisionModel
from transformers import TextStreamer


def downsample_video(video_path: str, num_frames: int = 8):
    """Extract evenly spaced frames for VLM context."""
    
    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        raise ValueError(f"Could not read video: {video_path}")
    
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    
    for i in indices:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, image = vidcap.read()
        if success:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(image))
    
    vidcap.release()
    
    if not frames:
        raise ValueError(f"Could not extract frames from: {video_path}")
    
    return frames


def run_inference(model, processor, video_path: str, num_frames: int = 8, 
                 instruction: str = None, temperature: float = 1.0, 
                 top_p: float = 0.95, top_k: int = 64, max_new_tokens: int = 128):
    """Run inference on a video."""
    
    # Default instruction
    if instruction is None:
        instruction = (
            "Given the visual input from the user's forward perspective, generate exactly one short sentence "
            "to guide a visually impaired user by identifying critical obstacles or landmarks, describing their "
            "locations using clock directions relative to the user (12 o'clock is straight ahead), including "
            "relevant details such as size, material, or distance, and giving one clear action, while prioritizing "
            "immediate safety and avoiding any extra explanation."
        )
    
    print(f"📹 Processing video: {video_path}")
    
    # Extract frames
    frames = downsample_video(video_path, num_frames)
    print(f"✅ Extracted {len(frames)} frames")
    
    # Construct the multimodal message
    messages = [{"role": "user", "content": [{"type": "text", "text": instruction}]}]
    
    # Interleave the extracted frames into the message content
    for img in frames:
        messages[0]["content"].append({"type": "image", "image": img})
    
    # Apply chat template and process inputs
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    
    # Feed BOTH the text AND the actual images to the processor
    inputs = processor(
        images=frames,
        text=input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")
    
    print("\n" + "="*80)
    print("🤖 Model Response:")
    print("="*80)
    
    # Generate with streaming
    text_streamer = TextStreamer(processor, skip_prompt=True)
    result = model.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k
    )
    
    print("="*80 + "\n")
    
    # Decode full response
    response = processor.decode(result[0], skip_special_tokens=True)
    return response


def main():
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned Gemma-3N")
    
    # Model configuration
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to fine-tuned model (merged model directory)")
    parser.add_argument("--video_path", type=str, required=True,
                       help="Path to video file")
    
    # Processing configuration
    parser.add_argument("--num_frames", type=int, default=8,
                       help="Number of frames to extract from video")
    parser.add_argument("--instruction", type=str, default=None,
                       help="Custom instruction prompt")
    
    # Generation configuration
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95,
                       help="Top-p sampling")
    parser.add_argument("--top_k", type=int, default=64,
                       help="Top-k sampling")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                       help="Maximum tokens to generate")
    
    args = parser.parse_args()
    
    print("="*80)
    print("Gemma-3N Inference with Unsloth")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Video: {args.video_path}")
    print(f"Frames: {args.num_frames}")
    print("="*80 + "\n")
    
    # Load model
    print("📦 Loading model...")
    model, processor = FastVisionModel.from_pretrained(
        args.model_path,
        use_gradient_checkpointing="unsloth",
    )
    
    # Enable inference mode
    FastVisionModel.for_inference(model)
    print("✅ Model loaded\n")
    
    # Run inference
    response = run_inference(
        model=model,
        processor=processor,
        video_path=args.video_path,
        num_frames=args.num_frames,
        instruction=args.instruction,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens
    )
    
    print("✅ Inference completed!")


if __name__ == "__main__":
    main()
