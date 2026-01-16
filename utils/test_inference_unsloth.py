#!/usr/bin/env python3
"""
Test Inference Script for QVED Dataset using Unsloth FastVisionModel

This script runs inference on videos from the QVED test set using a finetuned Gemma-3N model.
It loads videos from qved_test.json and generates predictions.

Usage:
    python utils/test_inference_unsloth.py --model_path outputs/gemma3n_finetune_20260108_162806_merged_16bit
    python utils/test_inference_unsloth.py --model_path unsloth/gemma-3n-E4B-it --output test_predictions.json
"""

import os
import warnings
import logging
import argparse
import json
import time
from pathlib import Path
from typing import List

os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings("ignore")

logging.getLogger('mmengine').setLevel(logging.CRITICAL)
logging.getLogger('transformers').setLevel(logging.CRITICAL)
logging.getLogger('transformers.modeling_utils').setLevel(logging.CRITICAL)

import torch
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from unsloth import FastVisionModel


def extract_frames(video_path: str, num_frames: int = 8) -> List[Image.Image]:
    """Extract evenly-spaced frames from a video file using numpy.linspace."""
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Video has no frames: {video_path}")
    
    # Use numpy.linspace for evenly-spaced frame indices
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR (OpenCV) to RGB (PIL)
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frames.append(img)

    cap.release()
    
    if len(frames) != num_frames:
        raise ValueError(f"Expected {num_frames} frames, got {len(frames)}")
    
    return frames


def load_model(model_path: str, device: str = "cuda"):
    """Load Gemma-3N model using Unsloth FastVisionModel."""
    print(f"📦 Loading model from: {model_path}")
    
    # Determine base model for tokenizer
    base_model = "unsloth/gemma-3n-E2B-it"
    
    # Check if this is a custom fine-tuned model
    is_custom_model = "/" in model_path and not model_path.startswith("unsloth/")
    
    if is_custom_model:
        print(f"🔍 Detected fine-tuned model. Will use base model tokenizer: {base_model}")
        
        # Load tokenizer from base model
        from transformers import AutoTokenizer
        print(f"📥 Loading tokenizer from base model: {base_model}")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model, 
            trust_remote_code=True,
        )
        print("✅ Tokenizer loaded from base model!")
        
        # Load the fine-tuned model weights
        print(f"📥 Loading fine-tuned model weights from: {model_path}")
        from transformers import AutoModelForVision2Seq
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map=device,
        )
        print("✅ Fine-tuned model loaded!")
        
        # Wrap with Unsloth for inference optimization
        try:
            FastVisionModel.for_inference(model)
        except:
            print("⚠ Could not apply Unsloth inference optimization, continuing without it...")
        
    else:
        # Standard Unsloth loading for base models
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name=model_path,
            dtype=None,
            max_seq_length=50000,
            load_in_4bit=False,
            trust_remote_code=True,
        )
        FastVisionModel.for_inference(model)
    
    model.to(device)
    print("✅ Model ready for inference!")
    return model, tokenizer


def run_inference(
    model,
    tokenizer,
    video_path: str,
    prompt: str,
    device: str = "cuda",
    max_new_tokens: int = 256,
    num_frames: int = 8
):
    """Run inference on a single video."""
    # Extract frames
    frames = extract_frames(video_path, num_frames=num_frames)
    
    # Prepare messages
    messages = [
        {
            "role": "user",
            "content": [
                *[{"type": "image"} for _ in range(num_frames)],
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    # Tokenize with chat template
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)
    
    input_token_count = inputs['input_ids'].shape[1]
    
    # Generate
    with torch.inference_mode():
        start_time = time.time()
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            use_cache=True,
        )
        end_time = time.time()
        generation_time = end_time - start_time
    
    # Decode output - only the NEW tokens (the answer)
    input_length = inputs['input_ids'].shape[1]
    new_tokens = output_ids[0][input_length:]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    generated_token_count = len(new_tokens)
    tokens_per_second = generated_token_count / generation_time if generation_time > 0 else 0
    
    return response_text, {
        'generated_tokens': generated_token_count,
        'generation_time': generation_time,
        'tokens_per_second': tokens_per_second
    }


def warmup_gpu(model, tokenizer, test_video: str, device: str = "cuda", max_new_tokens: int = 256):
    """Warm up GPU with a sample video."""
    if not os.path.exists(test_video):
        return
    
    print("\n🔥 Warming up GPU...")
    try:
        _ = run_inference(
            model, 
            tokenizer, 
            test_video, 
            "Test prompt", 
            device, 
            max_new_tokens=32  # Short warmup
        )
        print("✓ GPU warmup complete")
    except Exception as e:
        print(f"⚠ Warmup warning: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run inference on QVED test set with Unsloth FastVisionModel")
    parser.add_argument("--model_path", type=str, required=True, help="Path to finetuned model or HuggingFace model name")
    parser.add_argument("--test_json", type=str, default="dataset/qved_test.json", help="Path to test set JSON")
    parser.add_argument("--data_path", type=str, default="videos", help="Base path for video files")
    parser.add_argument("--output", type=str, default=None, help="Output file for predictions")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of new tokens")
    parser.add_argument("--num_frames", type=int, default=8, help="Number of frames to extract per video")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for testing")

    args = parser.parse_args()

    # Set default output path
    if args.output is None:
        model_name = Path(args.model_path).name.replace('/', '_')
        output_dir = Path("results") / f"test_inference_{model_name}"
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(output_dir / "test_predictions.json")
    
    print("=" * 60)
    print("🎬 Gemma-3N Test Inference (Unsloth)")
    print("=" * 60)
    print(f"Model:           {args.model_path}")
    print(f"Test JSON:       {args.test_json}")
    print(f"Data path:       {args.data_path}")
    print(f"Output:          {args.output}")
    print(f"Device:          {args.device}")
    print(f"Max tokens:      {args.max_new_tokens}")
    print(f"Frames/video:    {args.num_frames}")
    if args.limit:
        print(f"Sample limit:    {args.limit}")
    print("=" * 60)

    # Load model
    model, tokenizer = load_model(args.model_path, device=args.device)

    # Load test data
    print(f"\n📋 Loading test data from: {args.test_json}")
    with open(args.test_json, 'r') as f:
        test_data = json.load(f)

    if args.limit:
        test_data = test_data[:args.limit]
        print(f"Limited to {args.limit} samples")
    print(f"Total test samples: {len(test_data)}")

    # GPU warmup
    if test_data and args.device == "cuda":
        first_video = str(Path(args.data_path) / test_data[0]['video'])
        warmup_gpu(model, tokenizer, first_video, args.device, args.max_new_tokens)

    # Run inference
    results = []
    throughput_stats = []
    print("\n🎬 Running inference...")

    for item in tqdm(test_data, desc="Processing videos"):
        video_rel_path = item['video']
        # Extract just the filename (remove any subdirectories)
        video_filename = Path(video_rel_path).name
        video_path = str(Path(args.data_path) / video_filename)
        conversations = item['conversations']
        prompt = conversations[0]['value']
        ground_truth = conversations[1]['value']

        try:
            prediction, metrics = run_inference(
                model,
                tokenizer,
                video_path,
                prompt,
                args.device,
                args.max_new_tokens,
                args.num_frames
            )
            throughput_stats.append(metrics['tokens_per_second'])

            results.append({
                "video_path": video_rel_path,
                "prompt": prompt,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "generated_tokens": metrics['generated_tokens'],
                "generation_time": round(metrics['generation_time'], 4),
                "tokens_per_second": round(metrics['tokens_per_second'], 2),
                "status": "success"
            })
        except Exception as e:
            results.append({
                "video_path": video_rel_path,
                "prompt": prompt,
                "ground_truth": ground_truth,
                "prediction": "",
                "status": "error",
                "error": str(e)
            })
            print(f"\n✗ Error processing {video_rel_path}: {str(e)}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - successful

    print(f"\n{'='*60}")
    print("✅ Inference Complete!")
    print(f"Total: {len(results)} | Successful: {successful} | Failed: {failed}")
    
    if throughput_stats:
        avg_throughput = np.mean(throughput_stats)
        median_throughput = np.median(throughput_stats)
        min_throughput = np.min(throughput_stats)
        max_throughput = np.max(throughput_stats)
        print(f"📊 Throughput (tokens/sec):")
        print(f"    Mean: {avg_throughput:.2f} | Median: {median_throughput:.2f}")
        print(f"    Min: {min_throughput:.2f} | Max: {max_throughput:.2f}")
    
    print(f"📁 Results saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
