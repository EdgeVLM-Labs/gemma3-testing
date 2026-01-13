#!/usr/bin/env python3
"""
Gemma-3N Fine-tuning with Unsloth - Production Ready
Fine-tune gemma-3n-E4B-it on video dataset using Unsloth FastVisionModel.

Usage:
    python gemma3n_finetune_unsloth.py \
        --train_json dataset/qved_train.json \
        --val_json dataset/qved_val.json \
        --video_dir test_videos \
        --output_dir outputs/gemma3n_finetune
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import gc

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np
import cv2
from PIL import Image
from datasets import Dataset
from huggingface_hub import login
from unsloth import FastVisionModel, get_chat_template
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

# ============================================================================
# Video Processing with Robust Error Handling
# ============================================================================

def downsample_video(
    video_path: str, 
    num_frames: int = 16, 
    resolution: int = 224,
    timeout_seconds: int = 10
) -> List[Image.Image]:
    """
    Extract evenly spaced frames from video with robust error handling.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract (default: 16)
        resolution: Target resolution for frames (default: 224x224)
        timeout_seconds: Maximum time to spend on one video
        
    Returns:
        List of PIL Images, empty list if failed
    """
    start_time = time.time()
    frames = []
    vidcap = None
    
    try:
        # Check if file exists
        if not os.path.exists(video_path):
            return []
        
        # Check file size (skip if too small/large)
        file_size = os.path.getsize(video_path)
        if file_size < 1024 or file_size > 500 * 1024 * 1024:  # <1KB or >500MB
            return []
        
        vidcap = cv2.VideoCapture(video_path)
        
        if not vidcap.isOpened():
            return []
        
        # Get video properties
        total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        
        # Validate video properties
        if total_frames <= 0 or fps <= 0 or total_frames > 100000:
            vidcap.release()
            return []
        
        # Calculate frame indices to sample
        indices = np.linspace(0, total_frames - 1, min(num_frames, total_frames), dtype=int)
        
        # Sequential reading (faster than seeking)
        current_frame = 0
        target_indices = sorted(set(indices))  # Remove duplicates and sort
        
        for target_idx in target_indices:
            # Check timeout
            if time.time() - start_time > timeout_seconds:
                break
            
            # Skip frames to reach target
            while current_frame < target_idx:
                ret = vidcap.grab()
                if not ret:
                    break
                current_frame += 1
            
            # Read and process the target frame
            success, image = vidcap.retrieve()
            if success and image is not None:
                # Resize to target resolution
                image = cv2.resize(
                    image, 
                    (resolution, resolution), 
                    interpolation=cv2.INTER_AREA
                )
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(image))
            
            current_frame += 1
            
            # Early exit if we have enough frames
            if len(frames) >= num_frames:
                break
        
        vidcap.release()
        return frames
        
    except Exception as e:
        if vidcap is not None:
            vidcap.release()
        return []
    finally:
        # Ensure video capture is released
        if vidcap is not None and vidcap.isOpened():
            vidcap.release()


def find_video_path(video_path: str, video_dir: Optional[str] = None) -> Optional[str]:
    """
    Find video file by trying multiple path combinations.
    
    Args:
        video_path: Original video path from JSON
        video_dir: Base directory to search in
        
    Returns:
        Valid path if found, None otherwise
    """
    # Extract filename and relative path
    filename = os.path.basename(video_path)
    
    # Try different path combinations
    paths_to_try = []
    
    # 1. Original path as-is
    paths_to_try.append(video_path)
    
    # 2. User-specified video_dir
    if video_dir:
        paths_to_try.extend([
            os.path.join(video_dir, video_path),
            os.path.join(video_dir, filename),
            os.path.join(os.path.abspath(video_dir), video_path),
            os.path.join(os.path.abspath(video_dir), filename),
        ])
    
    # 3. Common directory names
    for base_dir in ['test_videos', 'videos', 'data/videos']:
        paths_to_try.extend([
            os.path.join(base_dir, video_path),
            os.path.join(base_dir, filename),
        ])
    
    # 4. Current directory
    paths_to_try.append(filename)
    
    # Try each path
    for path in paths_to_try:
        if os.path.exists(path):
            return path
    
    return None


def process_single_sample(
    item: Dict,
    num_frames: int,
    resolution: int,
    video_dir: Optional[str],
    idx: int
) -> Tuple[bool, Optional[Dict], Optional[str]]:
    """
    Process a single video sample.
    
    Returns:
        (success, dataset_entry, error_message)
    """
    try:
        # Get video path
        video_path = item.get('video', '')
        if not video_path:
            return False, None, "no_video_path"
        
        # Find actual video file
        actual_path = find_video_path(video_path, video_dir)
        if actual_path is None:
            return False, None, f"not_found:{os.path.basename(video_path)}"
        
        # Extract frames
        frames = downsample_video(actual_path, num_frames, resolution)
        if not frames or len(frames) < num_frames // 2:  # Need at least half the frames
            return False, None, f"insufficient_frames:{len(frames)}"
        
        # Get conversations
        conversations = item.get('conversations', [])
        if len(conversations) < 2:
            return False, None, "no_conversations"
        
        # Extract prompt and response
        user_msg = conversations[0].get('value', '').strip()
        assistant_msg = conversations[1].get('value', '').strip()
        
        if not user_msg or not assistant_msg:
            return False, None, "empty_conversation"
        
        # Build message format
        user_content = [{"type": "text", "text": user_msg}]
        for img in frames:
            user_content.append({"type": "image", "image": img})
        
        entry = {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": [{"type": "text", "text": assistant_msg}]}
            ]
        }
        
        return True, entry, None
        
    except Exception as e:
        return False, None, f"exception:{str(e)[:50]}"


# ============================================================================
# Dataset Loading with Parallel Processing
# ============================================================================

def load_dataset_parallel(
    json_path: str,
    num_frames: int = 16,
    resolution: int = 224,
    video_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
    num_workers: int = 8
) -> Dataset:
    """
    Load dataset from JSON with parallel video processing.
    
    Args:
        json_path: Path to JSON file
        num_frames: Number of frames to extract per video
        resolution: Frame resolution (224x224)
        video_dir: Directory containing videos
        max_samples: Limit number of samples (for testing)
        num_workers: Number of parallel workers
        
    Returns:
        HuggingFace Dataset
    """
    print(f"\n{'='*80}")
    print(f"Loading Dataset from {json_path}")
    print(f"{'='*80}")
    
    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    total_samples = len(data)
    print(f"📊 Total samples in JSON: {total_samples}")
    
    if max_samples:
        data = data[:max_samples]
        print(f"⚠️  Limited to {max_samples} samples for testing")
    
    if video_dir:
        print(f"📁 Video directory: {video_dir}")
    
    print(f"🎬 Extracting {num_frames} frames @ {resolution}x{resolution} per video")
    print(f"🚀 Using {num_workers} parallel workers\n")
    
    # Process videos in parallel
    dataset = []
    errors = {}
    error_examples = {}
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(
                process_single_sample,
                item,
                num_frames,
                resolution,
                video_dir,
                idx
            ): idx for idx, item in enumerate(data)
        }
        
        # Process results
        completed = 0
        last_print = 0
        
        for future in as_completed(futures):
            idx = futures[future]
            completed += 1
            
            # Progress updates
            if completed - last_print >= 100 or completed == len(data):
                success_count = len(dataset)
                failed_count = completed - success_count
                success_rate = (success_count / completed * 100) if completed > 0 else 0
                
                print(f"  📹 {completed}/{len(data)} | "
                      f"✅ {success_count} | "
                      f"❌ {failed_count} | "
                      f"Success: {success_rate:.1f}%")
                last_print = completed
            
            try:
                success, entry, error_msg = future.result()
                
                if success:
                    dataset.append(entry)
                else:
                    # Track errors
                    error_type = error_msg.split(':')[0] if error_msg else "unknown"
                    errors[error_type] = errors.get(error_type, 0) + 1
                    
                    # Keep examples of each error type
                    if error_type not in error_examples:
                        error_examples[error_type] = error_msg
                        
            except Exception as e:
                errors['exception'] = errors.get('exception', 0) + 1
                if 'exception' not in error_examples:
                    error_examples['exception'] = str(e)[:100]
    
    # Summary
    print(f"\n{'='*80}")
    print(f"Dataset Loading Complete")
    print(f"{'='*80}")
    print(f"✅ Loaded: {len(dataset)} samples")
    print(f"❌ Skipped: {len(data) - len(dataset)} samples")
    print(f"📊 Success rate: {len(dataset)/len(data)*100:.1f}%")
    
    if errors:
        print(f"\n⚠️  Error Summary:")
        for error_type, count in sorted(errors.items(), key=lambda x: -x[1]):
            example = error_examples.get(error_type, "")
            print(f"   - {error_type}: {count} | Example: {example}")
    
    print(f"{'='*80}\n")
    
    # Validation
    if len(dataset) == 0:
        raise ValueError(
            f"No valid samples loaded from {json_path}. "
            f"Check video paths and video_dir parameter."
        )
    
    if len(dataset) < len(data) * 0.1:
        print(f"⚠️  WARNING: Only {len(dataset)/len(data)*100:.1f}% of samples loaded!")
        print(f"   Consider checking video paths or filtering the JSON file.\n")
    
    print(f"⏳ Converting to HuggingFace Dataset (this may take a moment)...", flush=True)
    
    # Create dataset without deep copying images to avoid memory issues
    # Disable pyarrow to prevent serialization issues with PIL images
    import datasets
    datasets.disable_caching()
    
    hf_dataset = Dataset.from_list(dataset)
    
    # Disable format transforms to keep PIL images as-is
    hf_dataset.set_format(type=None)
    
    print(f"✅ HuggingFace Dataset created\n", flush=True)
    
    return hf_dataset


# ============================================================================
# Training Configuration
# ============================================================================

def create_trainer(
    model,
    processor,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset],
    args: argparse.Namespace
) -> SFTTrainer:
    """Create SFTTrainer with optimized configuration."""
    
    # Calculate steps
    steps_per_epoch = len(train_dataset) // (args.batch_size * args.gradient_accumulation)
    total_steps = steps_per_epoch * args.num_epochs
    
    # Auto-calculate eval_steps and logging_steps
    eval_steps = args.eval_steps if args.eval_steps > 0 else max(1, steps_per_epoch // 10)
    logging_steps = args.logging_steps if args.logging_steps > 0 else 1
    
    print(f"\n{'='*80}")
    print(f"Training Configuration")
    print(f"{'='*80}")
    print(f"📊 Dataset: {len(train_dataset)} training samples")
    if eval_dataset:
        print(f"📊 Validation: {len(eval_dataset)} samples")
    print(f"📈 Steps per epoch: {steps_per_epoch}")
    print(f"📈 Total steps: {total_steps}")
    print(f"📈 Eval every: {eval_steps} steps")
    print(f"📈 Log every: {logging_steps} steps")
    print(f"💾 Save every: {args.save_steps} steps")
    print(f"{'='*80}\n")
    
    # Training arguments
    training_args = SFTConfig(
        # Batch configuration
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        
        # Optimization
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        
        # Training duration
        num_train_epochs=args.num_epochs,
        
        # Learning rates
        learning_rate=args.learning_rate,
        
        # Optimizer
        optim="adamw_torch_fused",
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        
        # Logging
        logging_steps=logging_steps,
        logging_first_step=True,
        logging_strategy="steps",
        
        # Saving
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        
        # Evaluation
        eval_strategy="steps" if args.run_eval and eval_dataset else "no",
        eval_steps=eval_steps if args.run_eval and eval_dataset else None,
        eval_on_start=False,  # Disabled to prevent hanging on large datasets
        load_best_model_at_end=True if args.run_eval and eval_dataset else False,
        metric_for_best_model="eval_loss" if args.run_eval and eval_dataset else None,
        greater_is_better=False,
        
        # Other
        seed=3407,
        output_dir=args.output_dir,
        report_to="wandb" if args.use_wandb else "none",
        dataloader_num_workers=args.dataloader_num_workers,
        
        # Unsloth specific
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=args.max_seq_length,
        packing=False,  # Disable packing for video datasets
    )
    
    print("⏳ Step 1/3: Creating data collator...", flush=True)
    data_collator = UnslothVisionDataCollator(
        model, 
        processor, 
        max_seq_length=args.max_seq_length
    )
    print("✅ Data collator created", flush=True)
    
    print("⏳ Step 2/3: Creating SFTTrainer instance...", flush=True)
    print("   (This may take several minutes for large datasets)", flush=True)
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if args.run_eval else None,
        processing_class=processor.tokenizer,
        data_collator=data_collator,
        args=training_args,
    )
    
    print("✅ Step 3/3: SFTTrainer instance created", flush=True)
    
    return trainer


# ============================================================================
# Main Training Loop
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Gemma-3N with Unsloth",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Model
    parser.add_argument("--model_name", type=str, default="unsloth/gemma-3n-E2B-it")
    parser.add_argument("--load_in_4bit", action="store_true")
    
    # Dataset
    parser.add_argument("--train_json", type=str, required=True)
    parser.add_argument("--val_json", type=str, default=None)
    parser.add_argument("--video_dir", type=str, default="test_videos")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=50)
    
    # Video processing
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=8)
    
    # Training hyperparameters
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--projector_lr", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_seq_length", type=int, default=4096)  # Increased for video frames
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--dataloader_num_workers", type=int, default=0, 
                        help="DataLoader workers (0=main process, avoids hangs with large datasets)")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    
    # Checkpointing
    parser.add_argument("--save_steps", type=int, default=30)
    parser.add_argument("--eval_steps", type=int, default=0)
    parser.add_argument("--logging_steps", type=int, default=0)
    
    # LoRA
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    
    # Evaluation
    parser.add_argument("--run_eval", action="store_true")
    
    # Logging
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="gemma3n-finetune")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    
    # HuggingFace
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--upload_to_hf", action="store_true")
    parser.add_argument("--hf_repo_name", type=str, default=None)
    
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.hf_token:
        login(token=args.hf_token)
    
    # WandB setup
    if args.use_wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        os.environ["WANDB_LOG_MODEL"] = "false"
        if args.wandb_entity:
            os.environ["WANDB_ENTITY"] = args.wandb_entity
        if args.wandb_run_name:
            os.environ["WANDB_NAME"] = args.wandb_run_name
    
    print(f"\n{'='*80}")
    print(f"Gemma-3N Fine-tuning with Unsloth")
    print(f"{'='*80}")
    print(f"🤖 Model: {args.model_name}")
    print(f"📁 Output: {args.output_dir}")
    print(f"🎬 Frames: {args.num_frames} @ {args.resolution}x{args.resolution}")
    print(f"📊 Batch: {args.batch_size} x {args.gradient_accumulation} = {args.batch_size * args.gradient_accumulation}")
    print(f"📈 LR: {args.learning_rate}, Projector LR: {args.projector_lr}")
    print(f"🔧 LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"⏱️  Epochs: {args.num_epochs}")
    print(f"{'='*80}\n")
    
    # Load model
    print("📦 Loading model...")
    model, processor = FastVisionModel.from_pretrained(
        args.model_name,
        load_in_4bit=args.load_in_4bit,
        use_gradient_checkpointing="unsloth",
    )
    
    processor = get_chat_template(processor, "gemma-3n")
    print("✅ Model loaded\n")
    
    # Configure LoRA
    print("🔧 Configuring LoRA...")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        random_state=3407,
        use_rslora=False,
        target_modules="all-linear",
    )
    print("✅ LoRA configured\n")
    
    # Load datasets
    print(f"\n{'='*80}")
    print(f"Loading Training Dataset")
    print(f"{'='*80}")
    train_dataset = load_dataset_parallel(
        args.train_json,
        num_frames=args.num_frames,
        resolution=args.resolution,
        video_dir=args.video_dir,
        max_samples=args.max_train_samples,
        num_workers=args.num_workers
    )
    print(f"✅ Training dataset loaded: {len(train_dataset)} samples\n")
    
    eval_dataset = None
    if args.run_eval and args.val_json and os.path.exists(args.val_json):
        print(f"{'='*80}")
        print(f"Loading Validation Dataset")
        print(f"{'='*80}")
        eval_dataset = load_dataset_parallel(
            args.val_json,
            num_frames=args.num_frames,
            resolution=args.resolution,
            video_dir=args.video_dir,
            max_samples=args.max_val_samples,
            num_workers=args.num_workers
        )
        print(f"✅ Validation dataset loaded: {len(eval_dataset)} samples\n")
    
    # Free memory before trainer creation
    print("🧹 Cleaning up memory...")
    gc.collect()
    torch.cuda.empty_cache()
    print("✅ Memory cleaned\n")
    
    # Enable training
    print("🔧 Enabling training mode...")
    FastVisionModel.for_training(model)
    print("✅ Training mode enabled\n")
    
    # Create trainer
    print(f"{'='*80}")
    print(f"Creating Trainer (this may take a few minutes with large datasets)...")
    print(f"{'='*80}")
    print(f"📊 Dataset size: {len(train_dataset)} training samples")
    if eval_dataset:
        print(f"📊 Validation size: {len(eval_dataset)} samples")
    print(f"⏳ Initializing trainer components...\n")
    
    trainer = create_trainer(model, processor, train_dataset, eval_dataset, args)
    print(f"✅ Trainer created successfully\n")
    
    # GPU stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_mem = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_mem = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"🖥️  GPU: {gpu_stats.name} | {max_mem} GB total | {start_mem} GB reserved\n")
    
    # Train
    print(f"{'='*80}")
    print(f"🚀 Starting Training")
    print(f"{'='*80}\n")
    
    try:
        trainer_stats = trainer.train()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n❌ GPU OUT OF MEMORY ERROR!")
            print(f"   Current memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
            print(f"   Max memory: {max_mem} GB")
            print(f"\n💡 Suggestions:")
            print(f"   - Reduce --batch_size (currently {args.batch_size})")
            print(f"   - Reduce --num_frames (currently {args.num_frames})")
            print(f"   - Reduce --max_seq_length (currently {args.max_seq_length})")
            print(f"   - Add --load_in_4bit flag")
            raise
        else:
            print(f"\n❌ Runtime Error: {e}")
            raise
    except Exception as e:
        print(f"\n❌ Unexpected Error during training: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Final stats
    used_mem = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    train_mem = round(used_mem - start_mem, 3)
    
    print(f"\n{'='*80}")
    print(f"✅ Training Complete")
    print(f"{'='*80}")
    print(f"⏱️  Time: {trainer_stats.metrics['train_runtime']:.0f}s ({trainer_stats.metrics['train_runtime']/60:.1f}m)")
    print(f"💾 Peak memory: {used_mem} GB ({train_mem} GB for training)")
    print(f"{'='*80}\n")
    
    # Save model
    print("💾 Saving models...")
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    
    merged_dir = f"{args.output_dir}_merged_16bit"
    model.save_pretrained_merged(merged_dir, processor.tokenizer, save_method="merged_16bit")
    
    print(f"✅ Saved:")
    print(f"   - LoRA: {args.output_dir}")
    print(f"   - Merged: {merged_dir}\n")
    
    # Upload to HF
    if args.upload_to_hf:
        from huggingface_hub import HfApi, create_repo
        from datetime import datetime
        
        repo_name = args.hf_repo_name or f"gemma3n-finetune-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        api = HfApi()
        user = api.whoami()['name']
        repo_id = f"{user}/{repo_name}"
        
        print(f"📤 Uploading to HuggingFace: {repo_id}")
        create_repo(repo_id=repo_id, private=False, exist_ok=True)
        api.upload_folder(folder_path=merged_dir, repo_id=repo_id)
        print(f"✅ Uploaded: https://huggingface.co/{repo_id}\n")
    
    print(f"{'='*80}")
    print(f"✅ All Done!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()