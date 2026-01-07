#!/usr/bin/env python3
"""
Gemma-3N Fine-tuning with Unsloth

Fine-tune gemma-3n-E4B-it on QVED dataset using Unsloth FastModel.

Usage:
    python gemma3_finetune_unsloth.py \
        --model_name unsloth/gemma-3n-E4B-it \
        --train_json dataset/qved_train.json \
        --val_json dataset/qved_val.json \
        --output_dir results/gemma3n_E4B_finetune
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict
import warnings

warnings.filterwarnings("ignore")

import torch
from unsloth import FastModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset
import wandb

# Video processing
import cv2
from PIL import Image


def extract_frames(video_path: str, num_frames: int = 8) -> List[Image.Image]:
    """Extract evenly-spaced frames from video."""
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"⚠️ Warning: Could not open {video_path}")
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        return []
    
    step = max(1, total_frames // num_frames)
    frames = []
    
    for i in range(num_frames):
        frame_idx = i * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frames.append(img)
    
    cap.release()
    return frames


def load_qved_dataset(json_path: str, num_frames: int = 8) -> List[Dict]:
    """Load QVED dataset and convert to chat format."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    dataset = []
    for item in data:
        video_path = item.get('video', '')
        if not os.path.exists(video_path):
            print(f"⚠️ Skipping missing video: {video_path}")
            continue
        
        # Extract frames
        frames = extract_frames(video_path, num_frames)
        if not frames:
            continue
        
        # Get conversations
        conversations = item.get('conversations', [])
        if len(conversations) < 2:
            continue
        
        # Build prompt and response
        user_msg = conversations[0].get('value', '')
        assistant_msg = conversations[1].get('value', '')
        
        # Format for Unsloth
        dataset.append({
            'images': frames,
            'prompt': user_msg,
            'response': assistant_msg
        })
    
    return dataset


def formatting_func(examples):
    """Format examples for SFT training."""
    texts = []
    images_list = []
    
    for i in range(len(examples['prompt'])):
        # Build message format
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": img} for img in examples['images'][i]],
                    {"type": "text", "text": examples['prompt'][i]}
                ]
            },
            {
                "role": "assistant",
                "content": examples['response'][i]
            }
        ]
        
        texts.append(messages)
        images_list.append(examples['images'][i])
    
    return {"text": texts, "images": images_list}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma-3N with Unsloth")
    parser.add_argument("--model_name", type=str, default="unsloth/gemma-3n-E4B-it")
    parser.add_argument("--train_json", type=str, default="dataset/qved_train.json")
    parser.add_argument("--val_json", type=str, default="dataset/qved_val.json")
    parser.add_argument("--output_dir", type=str, default="results/gemma3n_E4B_finetune")
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--wandb_project", type=str, default="gemma3n-finetune")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    
    args = parser.parse_args()
    
    # Initialize wandb
    if args.wandb_run_name:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb.init(project=args.wandb_project)
    
    print("="*80)
    print("Gemma-3N Fine-tuning with Unsloth")
    print("="*80)
    print(f"Model: {args.model_name}")
    print(f"Training data: {args.train_json}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size} x {args.gradient_accumulation} = {args.batch_size * args.gradient_accumulation}")
    print("="*80 + "\n")
    
    # Load model
    print("Loading model...")
    torch._dynamo.config.recompile_limit = 64
    
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,  # Use 4-bit quantization for memory efficiency
    )
    
    # Configure LoRA
    model = FastModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=False,
        loftq_config=None,
    )
    
    print("✅ Model loaded with LoRA\n")
    
    # Load datasets
    print(f"Loading training data from {args.train_json}...")
    train_data = load_qved_dataset(args.train_json, args.num_frames)
    print(f"✅ Loaded {len(train_data)} training examples\n")
    
    eval_data = None
    if args.val_json and os.path.exists(args.val_json):
        print(f"Loading validation data from {args.val_json}...")
        eval_data = load_qved_dataset(args.val_json, args.num_frames)
        print(f"✅ Loaded {len(eval_data)} validation examples\n")
    
    # Convert to HF Dataset
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data) if eval_data else None
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        warmup_steps=50,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        save_strategy="epoch",
        evaluation_strategy="epoch" if eval_dataset else "no",
        report_to="wandb",
        load_best_model_at_end=True if eval_dataset else False,
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        formatting_func=formatting_func,
        max_seq_length=args.max_seq_length,
        args=training_args,
    )
    
    # Train
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")
    
    trainer.train()
    
    # Save model
    print("\n" + "="*80)
    print("Saving model...")
    print("="*80 + "\n")
    
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Save merged model (optional)
    merged_dir = f"{args.output_dir}_merged"
    print(f"Saving merged model to {merged_dir}...")
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
    
    print("\n" + "="*80)
    print("✅ Fine-tuning completed!")
    print(f"LoRA adapter saved to: {args.output_dir}")
    print(f"Merged model saved to: {merged_dir}")
    print("="*80)
    
    wandb.finish()


if __name__ == "__main__":
    main()
