#!/usr/bin/env python3
"""
Gemma-3N QVED Test Inference Script (Transformers) - FIXED
Uses native transformers library for physiotherapy exercise video analysis
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple
import warnings

import cv2
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor

warnings.filterwarnings("ignore")


def resize_image(img: Image.Image, target_width: int = 640, target_height: int = 640) -> Image.Image:
    """
    Resize image to target dimensions while preserving aspect ratio.
    
    Args:
        img: PIL Image
        target_width: Target width in pixels
        target_height: Target height in pixels
    
    Returns:
        Resized PIL Image
    """
    max_size = (target_width, target_height)
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    return img


def extract_frames(video_path: str, num_frames: int = 8) -> List[Tuple[Image.Image, float]]:
    """
    Extract evenly spaced frames from a video file.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
    
    Returns:
        List of tuples containing (PIL Image, timestamp in seconds)
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file: {video_path}")
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames == 0 or fps == 0:
        print(f"❌ Error: Invalid video file: {video_path}")
        cap.release()
        return []
    
    # Calculate step size to evenly distribute frames
    step = max(1, total_frames // num_frames)
    frames = []
    
    for i in range(num_frames):
        frame_idx = min(i * step, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Convert BGR to RGB
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img = resize_image(img)
        timestamp = round(frame_idx / fps, 2)
        frames.append((img, timestamp))
    
    cap.release()
    return frames


def get_video_inference(
    video_frames: List[Tuple[Image.Image, float]],
    prompt: str,
    model,
    processor,
    max_new_tokens: int = 256,
    frames_dir: str = "temp_frames"
) -> str:
    """
    Run inference on physiotherapy exercise video frames using Gemma-3 model.
    Properly handles Gemma3n multimodal architecture.
    
    Args:
        video_frames: List of (frame, timestamp) tuples
        prompt: Text prompt/question about exercise
        model: Loaded Gemma3nModel
        processor: Gemma3Processor
        max_new_tokens: Maximum tokens to generate
        frames_dir: Directory to temporarily save frames
    
    Returns:
        Model response string
    """
    if not video_frames:
        return "[ERROR: No frames extracted]"
    
    # Extract images
    images = [img for img, _ in video_frames]
    
    # Method 1: Try simple processor call
    try:
        # Simple text prompt
        text_prompt = f"{prompt}"
        
        # Process inputs
        inputs = processor(
            text=text_prompt,
            images=images,
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Generate
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        
        # Decode full output first
        full_response = processor.decode(outputs[0], skip_special_tokens=True)
        
        # Try to extract just the answer part (remove the input prompt)
        if text_prompt in full_response:
            response = full_response.replace(text_prompt, "").strip()
        else:
            response = full_response
        
        return response if response else "[No response generated]"
    
    except Exception as e1:
        print(f"  Method 1 failed: {e1}")
        
        # Method 2: Try chat template
        try:
            if hasattr(processor, 'apply_chat_template'):
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt}
                        ] + [{"type": "image"} for _ in images]
                    }
                ]
                
                inputs = processor.apply_chat_template(
                    messages,
                    images=images,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt"
                )
                
                inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                
                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                    )
                
                response = processor.decode(outputs[0], skip_special_tokens=True)
                return response.strip() if response.strip() else "[No response generated]"
            else:
                raise AttributeError("No apply_chat_template method")
                
        except Exception as e2:
            print(f"  Method 2 failed: {e2}")
            
            # Method 3: Try manual forward pass
            try:
                # Process text and images separately
                text_inputs = processor.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True
                ).to(model.device)
                
                if hasattr(processor, 'image_processor'):
                    image_inputs = processor.image_processor(
                        images=images,
                        return_tensors="pt"
                    )
                    image_inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                                   for k, v in image_inputs.items()}
                    
                    # Combine inputs
                    inputs = {**text_inputs, **image_inputs}
                else:
                    inputs = text_inputs
                
                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                    )
                
                response = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Clean up the response
                if prompt in response:
                    response = response.replace(prompt, "").strip()
                
                return response if response else "[No response generated]"
                
            except Exception as e3:
                print(f"  Method 3 failed: {e3}")
                import traceback
                traceback.print_exc()
                return f"[ERROR: All methods failed - {str(e1)[:50]}]"


def load_test_data(test_json: str) -> List[dict]:
    """Load test dataset from JSON file."""
    with open(test_json, 'r') as f:
        data = json.load(f)
    return data


def save_predictions(predictions: List[dict], output_path: str):
    """Save predictions to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"✓ Saved predictions to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Gemma-3N QVED Test Inference (Transformers)")
    parser.add_argument("--model_path", type=str, default="google/gemma-3-4b-it",
                        help="Path to model or HuggingFace model ID")
    parser.add_argument("--test_json", type=str, required=True,
                        help="Path to QVED test set JSON")
    parser.add_argument("--data_path", type=str, default="videos",
                        help="Base path for exercise video files")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for predictions JSON")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum tokens to generate")
    parser.add_argument("--num_frames", type=int, default=8,
                        help="Number of frames to extract from videos")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples to process")
    
    args = parser.parse_args()
    
    # Validate device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        args.device = "cpu"
    
    print("=" * 60)
    print("Gemma-3N QVED Inference (Transformers)")
    print("Exercise Video Analysis")
    print("=" * 60)
    print(f"Model:           {args.model_path}")
    print(f"Test JSON:       {args.test_json}")
    print(f"Video path:      {args.data_path}")
    print(f"Device:          {args.device}")
    print(f"Max new tokens:  {args.max_new_tokens}")
    print(f"Frames/video:    {args.num_frames}")
    if args.limit:
        print(f"Sample limit:    {args.limit}")
    print("=" * 60)
    
    # Load model and processor
    print("\n📦 Loading model and processor...")
    try:
        # Load config to understand the model structure
        config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        print(f"  Model type: {config.model_type}")
        if hasattr(config, 'architectures'):
            print(f"  Architecture: {config.architectures[0]}")
        
        # Import AutoModelForCausalLM which should handle generation properly
        from transformers import AutoModelForCausalLM
        
        # Try loading with AutoModelForCausalLM first
        print("  Attempting to load with AutoModelForCausalLM...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.float16 if args.device == "cuda" else torch.float32
            )
            print(f"  ✓ Loaded with AutoModelForCausalLM: {type(model).__name__}")
        except Exception as e:
            print(f"  AutoModelForCausalLM failed: {e}")
            print("  Falling back to AutoModel...")
            
            from transformers import AutoModel
            model = AutoModel.from_pretrained(
                args.model_path,
                trust_remote_code=True,
                device_map="auto"
            )
            print(f"  Loaded model class: {type(model).__name__}")
            
            # For Gemma3n, manually add generation capability
            if not hasattr(model, 'generate'):
                print("  Adding generation capability...")
                from transformers.generation import GenerationMixin
                
                # Make the model inherit from GenerationMixin
                model.__class__ = type(
                    model.__class__.__name__,
                    (model.__class__, GenerationMixin),
                    {}
                )
                
                # Ensure the model can generate
                if hasattr(model, 'language_model'):
                    # Route generate calls through language_model if it exists
                    original_prepare = model.prepare_inputs_for_generation if hasattr(model, 'prepare_inputs_for_generation') else None
                    
                    def prepare_inputs_for_generation(self, input_ids, **kwargs):
                        if original_prepare:
                            return original_prepare(input_ids, **kwargs)
                        return {"input_ids": input_ids, **kwargs}
                    
                    model.prepare_inputs_for_generation = prepare_inputs_for_generation.__get__(model)
                
                print("  ✓ Generation capability added")
        
        model = model.eval()
        
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        print("✓ Model and processor loaded successfully")
        print(f"  Has generate: {hasattr(model, 'generate')}")
        print(f"  Processor type: {type(processor).__name__}")
        print(f"  Processor has apply_chat_template: {hasattr(processor, 'apply_chat_template')}")
        
        # Check if processor has image_processor
        if hasattr(processor, 'image_processor'):
            print(f"  Has image_processor: True")
        if hasattr(processor, 'tokenizer'):
            print(f"  Has tokenizer: True")
            print(f"  Pad token ID: {processor.tokenizer.pad_token_id}")
            print(f"  EOS token ID: {processor.tokenizer.eos_token_id}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Load test data
    print(f"\n📂 Loading QVED test data from: {args.test_json}")
    test_data = load_test_data(args.test_json)
    
    if args.limit:
        test_data = test_data[:args.limit]
    
    print(f"✓ Loaded {len(test_data)} exercise video samples")
    
    # Run inference
    print("\n🔍 Running inference on exercise videos...")
    predictions = []
    
    # Create a temporary directory for frames
    frames_temp_dir = os.path.join(os.path.dirname(args.output), "temp_frames")
    
    for idx, sample in enumerate(tqdm(test_data, desc="Processing videos")):
        # Handle both dataset formats
        video_path = sample.get("video", "")
        
        # Check if using conversations format (like Unsloth)
        if "conversations" in sample:
            conversations = sample["conversations"]
            question = conversations[0].get("value", "")
            ground_truth = conversations[1].get("value", "")
        else:
            # Flat format
            question = sample.get("question", "")
            ground_truth = sample.get("answer", "")
        
        # Construct full path
        full_path = os.path.join(args.data_path, video_path)
        
        if not os.path.exists(full_path):
            print(f"\n⚠️  Video not found: {full_path}")
            prediction = "[ERROR: Video file not found]"
        else:
            # Extract frames and run inference
            frames = extract_frames(full_path, args.num_frames)
            if frames:
                prediction = get_video_inference(
                    frames, question, model, processor, 
                    args.max_new_tokens, frames_temp_dir
                )
            else:
                prediction = "[ERROR: Failed to extract frames from video]"
        
        predictions.append({
            "video_path": video_path,
            "question": question,
            "ground_truth": ground_truth,
            "prediction": prediction
        })
    
    # Save predictions
    print(f"\n💾 Saving predictions...")
    save_predictions(predictions, args.output)
    
    # Clean up temporary frames directory
    if os.path.exists(frames_temp_dir):
        try:
            os.rmdir(frames_temp_dir)
        except:
            pass
    
    print("\n" + "=" * 60)
    print("✅ Inference complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()