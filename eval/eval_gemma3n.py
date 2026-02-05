#!/usr/bin/env python3
"""
Evaluation Script for Gemma-3N Models

Evaluate fine-tuned or base Gemma-3N models on video datasets.
Supports both single video evaluation and batch evaluation.

Usage:
    # Evaluate base model
    python scripts/eval_gemma3n.py \
        --model_path google/gemma-3n-E2B-it \
        --eval_json dataset/qved_val.json \
        --output_file results/eval_base.json
    
    # Evaluate fine-tuned model
    python scripts/eval_gemma3n.py \
        --model_path outputs/gemma3n_finetune_merged_16bit \
        --eval_json dataset/qved_val.json \
        --output_file results/eval_finetuned.json
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

import torch
import cv2
import numpy as np
from PIL import Image
from unsloth import FastVisionModel
from transformers import TextStreamer

# Metrics
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


def extract_frames(video_path: str, num_frames: int = 8) -> List[Image.Image]:
    """Extract evenly spaced frames from video."""
    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        return []
    
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    for i in indices:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, image = vidcap.read()
        if success:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(image))
    vidcap.release()
    return frames


def load_model(model_path: str, max_seq_length: int = 50000):
    """Load Gemma-3N model."""
    print(f"📦 Loading model: {model_path}...")
    
    model, processor = FastVisionModel.from_pretrained(
        model_path,
        load_in_4bit=False,
    )
    
    model.to("cuda")
    print("✅ Model loaded\n")
    return model, processor


def run_inference(model, processor, messages: List[dict], max_new_tokens: int = 256) -> str:
    """Run inference on model."""
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=False,  # Deterministic for evaluation
        )
    
    # Decode only new tokens
    input_length = inputs.input_ids.shape[1]
    new_tokens = outputs[0][input_length:]
    response = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    return response.strip()


def calculate_metrics(reference: str, prediction: str) -> Dict[str, float]:
    """Calculate evaluation metrics."""
    # BLEU score
    ref_tokens = reference.lower().split()
    pred_tokens = prediction.lower().split()
    smoothing = SmoothingFunction().method1
    bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)
    
    # ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference, prediction)
    
    return {
        'bleu': bleu,
        'rouge1': rouge_scores['rouge1'].fmeasure,
        'rouge2': rouge_scores['rouge2'].fmeasure,
        'rougeL': rouge_scores['rougeL'].fmeasure,
    }


def evaluate_dataset(
    model,
    processor,
    eval_json: str,
    num_frames: int = 8,
    max_new_tokens: int = 256,
    max_samples: int = None
):
    """Evaluate model on dataset."""
    
    # Load evaluation data
    with open(eval_json, 'r') as f:
        data = json.load(f)
    
    if max_samples:
        data = data[:max_samples]
    
    results = []
    total_metrics = {'bleu': 0, 'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    skipped = 0
    
    print(f"📊 Evaluating on {len(data)} samples...\n")
    
    for idx, item in enumerate(tqdm(data, desc="Evaluating")):
        video_path = item.get('video', '')
        
        # Handle relative paths
        if not os.path.isabs(video_path):
            video_filename = os.path.basename(video_path)
            video_path = os.path.join('videos', video_filename)
        
        if not os.path.exists(video_path):
            skipped += 1
            continue
        
        # Extract frames
        frames = extract_frames(video_path, num_frames)
        if not frames:
            skipped += 1
            continue
        
        # Get ground truth
        conversations = item.get('conversations', [])
        if len(conversations) < 2:
            skipped += 1
            continue
        
        user_msg = conversations[0].get('value', '')
        ground_truth = conversations[1].get('value', '')
        
        # Build messages
        user_content = [{"type": "text", "text": user_msg}]
        for img in frames:
            user_content.append({"type": "image", "image": img})
        
        messages = [{"role": "user", "content": user_content}]
        
        # Run inference
        try:
            prediction = run_inference(model, processor, messages, max_new_tokens)
        except Exception as e:
            print(f"\n⚠️ Error on sample {idx}: {e}")
            skipped += 1
            continue
        
        # Calculate metrics
        metrics = calculate_metrics(ground_truth, prediction)
        
        # Accumulate
        for key in total_metrics:
            total_metrics[key] += metrics[key]
        
        # Store result
        results.append({
            'sample_id': idx,
            'video': video_path,
            'prompt': user_msg,
            'ground_truth': ground_truth,
            'prediction': prediction,
            'metrics': metrics
        })
    
    # Calculate averages
    num_evaluated = len(results)
    avg_metrics = {k: v / num_evaluated for k, v in total_metrics.items()}
    
    return results, avg_metrics, skipped


def main():
    parser = argparse.ArgumentParser(description="Evaluate Gemma-3N model on video dataset")
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model (base or fine-tuned)")
    parser.add_argument("--eval_json", type=str, default="dataset/qved_val.json",
                       help="Path to evaluation JSON file")
    parser.add_argument("--output_file", type=str, default="results/eval_results.json",
                       help="Path to save evaluation results")
    parser.add_argument("--num_frames", type=int, default=8,
                       help="Number of frames to extract from video")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                       help="Maximum tokens to generate")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to evaluate (for quick testing)")
    parser.add_argument("--max_seq_length", type=int, default=50000,
                       help="Maximum sequence length")
    
    args = parser.parse_args()
    
    print("="*80)
    print("Gemma-3N Model Evaluation")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Eval data: {args.eval_json}")
    print(f"Output: {args.output_file}")
    print("="*80 + "\n")
    
    # Load model
    model, processor = load_model(args.model_path, args.max_seq_length)
    
    # Run evaluation
    results, avg_metrics, skipped = evaluate_dataset(
        model,
        processor,
        args.eval_json,
        args.num_frames,
        args.max_new_tokens,
        args.max_samples
    )
    
    # Print results
    print("\n" + "="*80)
    print("📊 Evaluation Results")
    print("="*80)
    print(f"Samples evaluated: {len(results)}")
    print(f"Samples skipped: {skipped}")
    print(f"\nAverage Metrics:")
    print(f"  BLEU:    {avg_metrics['bleu']:.4f}")
    print(f"  ROUGE-1: {avg_metrics['rouge1']:.4f}")
    print(f"  ROUGE-2: {avg_metrics['rouge2']:.4f}")
    print(f"  ROUGE-L: {avg_metrics['rougeL']:.4f}")
    print("="*80)
    
    # Save results
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    output_data = {
        'model_path': args.model_path,
        'eval_json': args.eval_json,
        'num_samples': len(results),
        'skipped': skipped,
        'average_metrics': avg_metrics,
        'samples': results
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Results saved to: {args.output_file}")
    
    # Show some examples
    if results:
        print("\n" + "="*80)
        print("📝 Sample Predictions (first 3)")
        print("="*80)
        for i, result in enumerate(results[:3]):
            print(f"\n--- Sample {i+1} ---")
            print(f"Prompt: {result['prompt'][:100]}...")
            print(f"Ground Truth: {result['ground_truth']}")
            print(f"Prediction: {result['prediction']}")
            print(f"Metrics: BLEU={result['metrics']['bleu']:.3f}, ROUGE-L={result['metrics']['rougeL']:.3f}")


if __name__ == "__main__":
    main()
