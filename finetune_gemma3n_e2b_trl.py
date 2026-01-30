#!/usr/bin/env python3
"""
Gemma-3n-E2B-it Fine-tuning Script for QVED Dataset
Uses TRL SFTTrainer with video frame extraction for physiotherapy exercise analysis

USAGE:
    Basic training with default hyperparameters:
        python finetune_gemma3n_e2b_trl.py \
            --train_json dataset/qved_train.json \
            --val_json dataset/qved_val.json \
            --data_path videos/ \
            --output_dir ./outputs/gemma3n-e2b-qved-ft

    Custom hyperparameters:
        python finetune_gemma3n_e2b_trl.py \
            --train_json dataset/qved_train.json \
            --val_json dataset/qved_val.json \
            --data_path videos/ \
            --output_dir ./outputs/custom-run \
            --num_train_epochs 5 \
            --learning_rate 3e-4 \
            --lora_r 128

    With wandb logging:
        python finetune_gemma3n_e2b_trl.py \
            --train_json dataset/qved_train.json \
            --val_json dataset/qved_val.json \
            --data_path videos/ \
            --output_dir ./outputs/run1 \
            --wandb_project "gemma3n-qved-finetuning" \
            --run_name "gemma3n-e2b-lr2e4-r64"

    Resume from checkpoint:
        python finetune_gemma3n_e2b_trl.py \
            --train_json dataset/qved_train.json \
            --val_json dataset/qved_val.json \
            --data_path videos/ \
            --output_dir ./outputs/gemma3n-e2b-qved-ft \
            --resume_from_checkpoint ./outputs/gemma3n-e2b-qved-ft/checkpoint-30

REQUIREMENTS:
    - torch, transformers>=4.49.0, trl, peft, timm
    - opencv-python, Pillow
    - wandb (optional, for experiment tracking)
    - Install: pip install -U torch transformers trl peft timm opencv-python Pillow wandb

DATASET FORMAT:
    JSON file with the following structure:
    [
        {
            "video": "path/to/video.mp4",
            "conversations": [
                {"from": "human", "value": "Question about exercise?"},
                {"from": "gpt", "value": "Answer about exercise form and corrections"}
            ]
        },
        ...
    ]
"""

import argparse
import io
import json
import math
import os
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import torch
from datasets import Dataset
from PIL import Image
from peft import LoraConfig
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from trl import SFTConfig, SFTTrainer

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not installed. Run: pip install wandb")


def resize_image(img: Image.Image, target_width: int = 224, target_height: int = 224) -> Image.Image:
    """
    Resize image to target dimensions while preserving aspect ratio.
    
    Args:
        img: PIL Image
        target_width: Target width in pixels (default: 224 for Gemma-3n encoder)
        target_height: Target height in pixels (default: 224 for Gemma-3n encoder)
    
    Returns:
        Resized PIL Image
    """
    max_size = (target_width, target_height)
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    return img


def extract_frames(video_path: str, num_frames: int = 8) -> List[Image.Image]:
    """
    Extract evenly spaced frames from a video file.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract (default: 8)
    
    Returns:
        List of PIL Images
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
        img = resize_image(img, target_width=224, target_height=224)
        frames.append(img)
    
    cap.release()
    return frames


def load_qved_dataset(json_path: str, data_path: str, num_frames: int = 8) -> Dataset:
    """
    Load QVED dataset from JSON and extract video frames.
    
    Args:
        json_path: Path to JSON file with dataset annotations
        data_path: Base path for video files
        num_frames: Number of frames to extract per video
    
    Returns:
        HuggingFace Dataset with extracted frames and formatted messages
    """
    print(f"📂 Loading dataset from: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    formatted_data = []
    skipped = 0
    
    for idx, sample in enumerate(data):
        video_path = sample.get("video", "")
        full_video_path = os.path.join(data_path, video_path)
        
        # Extract frames from video
        if not os.path.exists(full_video_path):
            print(f"⚠️  Video not found: {full_video_path}")
            skipped += 1
            continue
        
        frames = extract_frames(full_video_path, num_frames)
        
        if not frames:
            print(f"⚠️  Failed to extract frames from: {full_video_path}")
            skipped += 1
            continue
        
        # Handle both conversation formats
        if "conversations" in sample:
            conversations = sample["conversations"]
            # Convert to expected format
            question = ""
            answer = ""
            for conv in conversations:
                if conv.get("from") == "human":
                    question = conv.get("value", "")
                elif conv.get("from") == "gpt":
                    answer = conv.get("value", "")
        else:
            question = sample.get("question", "")
            answer = sample.get("answer", "")
        
        # Format messages for Gemma-3n chat template
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an expert physiotherapy assistant analyzing exercise videos.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question}
                ] + [{"type": "image", "image": frame} for frame in frames]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}]
            },
        ]
        
        formatted_data.append({
            "messages": messages,
            "video_path": video_path
        })
        
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(data)} samples...")
    
    print(f"✓ Loaded {len(formatted_data)} samples (skipped {skipped})")
    
    return Dataset.from_list(formatted_data)


def process_vision_info(messages: list) -> List[Image.Image]:
    """
    Extract images from message content for processing.
    
    Args:
        messages: List of message dictionaries
    
    Returns:
        List of PIL Images
    """
    image_inputs = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]
        
        for element in content:
            if isinstance(element, dict) and ("image" in element or element.get("type") == "image"):
                if "image" in element:
                    image = element["image"]
                else:
                    image = element
                
                if image is not None:
                    # Handle dictionary with bytes
                    if isinstance(image, dict) and "bytes" in image:
                        pil_image = Image.open(io.BytesIO(image["bytes"]))
                        image_inputs.append(pil_image.convert("RGB"))
                    # Handle PIL Image objects
                    elif hasattr(image, "convert"):
                        image_inputs.append(image.convert("RGB"))
    
    return image_inputs


def create_collate_fn(processor):
    """
    Create data collator function for batching samples.
    
    Args:
        processor: AutoProcessor for Gemma-3n
    
    Returns:
        Collate function
    """
    def collate_fn(examples):
        texts = []
        images_list = []
        
        for example in examples:
            # Apply chat template to get text
            text = processor.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False
            ).strip()
            texts.append(text)
            
            # Extract images from messages
            images = process_vision_info(example["messages"])
            images_list.append(images)
        
        # Tokenize the texts and process the images
        batch = processor(
            text=texts, images=images_list, return_tensors="pt", padding=True
        )
        
        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()
        
        # Use Gemma3n specific token masking
        labels[labels == processor.tokenizer.pad_token_id] = -100
        if hasattr(processor.tokenizer, 'image_token_id'):
            labels[labels == processor.tokenizer.image_token_id] = -100
        if hasattr(processor.tokenizer, "audio_token_id"):
            labels[labels == processor.tokenizer.audio_token_id] = -100
        if hasattr(processor.tokenizer, 'boi_token_id'):
            labels[labels == processor.tokenizer.boi_token_id] = -100
        if hasattr(processor.tokenizer, 'eoi_token_id'):
            labels[labels == processor.tokenizer.eoi_token_id] = -100
        
        batch["labels"] = labels
        return batch
    
    return collate_fn


def calculate_steps(total_samples: int, batch_size: int, gradient_accumulation: int, num_epochs: int) -> dict:
    """
    Calculate training steps for logging and saving.
    
    Args:
        total_samples: Total number of training samples
        batch_size: Per-device batch size
        gradient_accumulation: Gradient accumulation steps
        num_epochs: Number of training epochs
    
    Returns:
        Dictionary with calculated steps
    """
    effective_batch_size = batch_size * gradient_accumulation
    steps_per_epoch = math.ceil(total_samples / effective_batch_size)
    total_steps = steps_per_epoch * num_epochs
    
    # Calculate logging steps (log ~20 times per epoch)
    logging_steps = max(1, steps_per_epoch // 20)
    
    # Calculate eval steps (eval ~5 times per epoch)
    eval_steps = max(1, steps_per_epoch // 5)
    
    return {
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "logging_steps": logging_steps,
        "eval_steps": eval_steps
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma-3n-E2B-it on QVED dataset")
    
    # Data arguments
    parser.add_argument("--model_path", type=str, default="google/gemma-3n-E2B-it",
                        help="Path to model or HuggingFace model ID")
    parser.add_argument("--train_json", type=str, required=True,
                        help="Path to training JSON file")
    parser.add_argument("--val_json", type=str, default=None,
                        help="Path to validation JSON file")
    parser.add_argument("--data_path", type=str, default="videos",
                        help="Base path for video files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for checkpoints and model")
    
    # Training hyperparameters
    parser.add_argument("--num_frames", type=int, default=8,
                        help="Number of frames to extract from videos (default: 8)")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs (default: 3)")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate (default: 2e-4)")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8,
                        help="Training batch size per device (default: 8)")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8,
                        help="Evaluation batch size per device (default: 8)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Maximum sequence length (default: 2048)")
    
    # LoRA configuration
    parser.add_argument("--lora_r", type=int, default=64,
                        help="LoRA attention dimension (default: 64)")
    parser.add_argument("--lora_alpha", type=int, default=128,
                        help="LoRA alpha parameter (default: 128)")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout (default: 0.05)")
    
    # Training configuration
    parser.add_argument("--warmup_ratio", type=float, default=0.05,
                        help="Warmup ratio (default: 0.05)")
    parser.add_argument("--save_steps", type=int, default=30,
                        help="Save checkpoint every N steps (default: 30)")
    parser.add_argument("--eval_strategy", type=str, default="steps",
                        choices=["no", "steps", "epoch"],
                        help="Evaluation strategy (default: steps)")
    parser.add_argument("--dataloader_num_workers", type=int, default=2,
                        help="Number of dataloader workers (default: 2)")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                        help="Enable gradient checkpointing (default: True)")
    
    # Wandb configuration
    parser.add_argument("--wandb_project", type=str, default="gemma3n-qved-finetuning",
                        help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default="fyp-21",
                        help="Wandb entity/team name (default: fyp-21)")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Wandb run name (default: auto-generated)")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable wandb logging")
    
    # Other arguments
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Resume training from checkpoint")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    # Validate CUDA availability
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available. Training will be very slow on CPU.")
        print("    Consider using a GPU-enabled environment.")
    
    # Setup wandb
    if not args.no_wandb and WANDB_AVAILABLE:
        if args.run_name is None:
            args.run_name = f"gemma3n-e2b-lr{args.learning_rate}-r{args.lora_r}-epochs{args.num_train_epochs}"
        
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args)
        )
        report_to = ["wandb"]
    else:
        report_to = []
        if not args.no_wandb and not WANDB_AVAILABLE:
            print("⚠️  Wandb not available. Install with: pip install wandb")
    
    print("=" * 70)
    print("Gemma-3n-E2B-it Fine-tuning on QVED Dataset")
    print("=" * 70)
    print(f"Model:                     {args.model_path}")
    print(f"Training data:             {args.train_json}")
    print(f"Validation data:           {args.val_json or 'None'}")
    print(f"Video base path:           {args.data_path}")
    print(f"Output directory:          {args.output_dir}")
    print(f"Frames per video:          {args.num_frames}")
    print(f"Epochs:                    {args.num_train_epochs}")
    print(f"Learning rate:             {args.learning_rate}")
    print(f"Batch size:                {args.per_device_train_batch_size}")
    print(f"Gradient accumulation:     {args.gradient_accumulation_steps}")
    print(f"Max sequence length:       {args.max_seq_length}")
    print(f"LoRA r:                    {args.lora_r}")
    print(f"LoRA alpha:                {args.lora_alpha}")
    print(f"Warmup ratio:              {args.warmup_ratio}")
    print(f"Save steps:                {args.save_steps}")
    print(f"Eval strategy:             {args.eval_strategy}")
    print(f"Gradient checkpointing:    {args.gradient_checkpointing}")
    print(f"Wandb logging:             {not args.no_wandb and WANDB_AVAILABLE}")
    if not args.no_wandb and WANDB_AVAILABLE:
        print(f"Wandb project:             {args.wandb_project}")
        print(f"Run name:                  {args.run_name}")
    print("=" * 70)
    
    # Load model and processor
    print("\n📦 Loading model and processor...")
    try:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        model = Gemma3nForConditionalGeneration.from_pretrained(
            args.model_path,
            device_map="auto",
            torch_dtype=dtype,
            trust_remote_code=True
        )
        
        processor = AutoProcessor.from_pretrained(
            args.model_path,
            trust_remote_code=True
        )
        processor.tokenizer.padding_side = "right"
        
        print(f"✓ Model loaded: {type(model).__name__}")
        print(f"  Model dtype: {model.dtype}")
        print(f"✓ Processor loaded: {type(processor).__name__}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Load datasets
    print("\n📂 Loading and processing datasets...")
    train_dataset = load_qved_dataset(args.train_json, args.data_path, args.num_frames)
    
    val_dataset = None
    if args.val_json and args.eval_strategy != "no":
        val_dataset = load_qved_dataset(args.val_json, args.data_path, args.num_frames)
        print(f"✓ Validation set: {len(val_dataset)} samples")
    
    print(f"✓ Training set: {len(train_dataset)} samples")
    
    # Calculate steps
    steps_info = calculate_steps(
        len(train_dataset),
        args.per_device_train_batch_size,
        args.gradient_accumulation_steps,
        args.num_train_epochs
    )
    
    print(f"\n📊 Training schedule:")
    print(f"  Steps per epoch:  {steps_info['steps_per_epoch']}")
    print(f"  Total steps:      {steps_info['total_steps']}")
    print(f"  Logging steps:    {steps_info['logging_steps']}")
    if args.eval_strategy == "steps" and val_dataset:
        print(f"  Eval steps:       {steps_info['eval_steps']}")
    
    # Configure LoRA
    print("\n🔧 Configuring LoRA...")
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_r,
        target_modules="all-linear",
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_rslora=False,
        use_dora=False,
        modules_to_save=None,
    )
    print(f"✓ LoRA config: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    
    # Configure training
    print("\n⚙️  Configuring training...")
    training_args = SFTConfig(
        output_dir=args.output_dir,
        eval_strategy=args.eval_strategy,
        eval_steps=steps_info['eval_steps'] if args.eval_strategy == "steps" else None,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=steps_info['logging_steps'],
        save_steps=args.save_steps,
        save_strategy="steps",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
        report_to=report_to,
        run_name=args.run_name if not args.no_wandb else None,
        dataset_kwargs={'skip_prepare_dataset': True, 'max_seq_length': args.max_seq_length},
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        seed=args.seed,
        save_total_limit=3,  # Keep only last 3 checkpoints
        load_best_model_at_end=True if args.eval_strategy != "no" and val_dataset else False,
        metric_for_best_model="eval_loss" if args.eval_strategy != "no" and val_dataset else None,
    )
    
    # Create collate function
    collate_fn = create_collate_fn(processor)
    
    # Initialize trainer
    print("\n🚀 Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor.tokenizer,
        peft_config=peft_config,
    )
    
    print("✓ Trainer initialized")
    
    # Train
    print("\n" + "=" * 70)
    print("🎯 Starting training...")
    print("=" * 70)
    
    try:
        if args.resume_from_checkpoint:
            print(f"📂 Resuming from checkpoint: {args.resume_from_checkpoint}")
            trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        else:
            trainer.train()
        
        print("\n" + "=" * 70)
        print("✅ Training completed successfully!")
        print("=" * 70)
        
        # Save final model
        print(f"\n💾 Saving final model to: {args.output_dir}")
        trainer.save_model(args.output_dir)
        processor.save_pretrained(args.output_dir)
        print("✓ Model and processor saved")
        
        # Save training arguments
        with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        print("✓ Training arguments saved")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        print(f"💾 Saving checkpoint to: {args.output_dir}/interrupted")
        trainer.save_model(f"{args.output_dir}/interrupted")
        processor.save_pretrained(f"{args.output_dir}/interrupted")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        if not args.no_wandb and WANDB_AVAILABLE:
            wandb.finish()


if __name__ == "__main__":
    main()
