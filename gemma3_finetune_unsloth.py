#!/usr/bin/env python3
"""
Gemma-3N Fine-tuning with Unsloth

Fine-tune gemma-3n-E4B-it on QVED dataset or HuggingFace dataset using Unsloth FastVisionModel.

Usage:
    # Local QVED dataset
    python gemma3_finetune_unsloth.py \
        --model_name unsloth/gemma-3n-E4B-it \
        --train_json dataset/qved_train.json \
        --output_dir results/gemma3n_E4B_finetune
    
    # HuggingFace dataset (streaming)
    python gemma3_finetune_unsloth.py \
        --model_name unsloth/gemma-3n-E4B-it \
        --hf_dataset blind-assist/walk-train \
        --hf_split train \
        --num_samples 500 \
        --output_dir results/gemma3n_E4B_finetune
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict
import warnings
from itertools import islice

warnings.filterwarnings("ignore")

import torch
from unsloth import FastVisionModel, get_chat_template
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from datasets import Dataset, load_dataset, Video
from huggingface_hub import login, hf_hub_download
import wandb

# Video processing
import cv2
import numpy as np
from PIL import Image


def downsample_video(video_path: str, num_frames: int = 8) -> List[Image.Image]:
    """Extract evenly spaced frames for VLM context."""
    
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


def convert_to_conversation(sample, num_frames: int = 8, instruction: str = None):
    """Convert sample to conversation format for training."""
    frames = downsample_video(sample["video"]["path"], num_frames)
    
    if not frames:
        return None
    
    # Default instruction if not provided
    if instruction is None:
        instruction = ("Given the visual input from the user's forward perspective, generate exactly one short sentence "
                      "to guide a visually impaired user by identifying critical obstacles or landmarks, describing their "
                      "locations using clock directions relative to the user (12 o'clock is straight ahead), including "
                      "relevant details such as size, material, or distance, and giving one clear action, while prioritizing "
                      "immediate safety and avoiding any extra explanation.")
    
    # Constructing the message content with interleaved text and images
    user_content = [{"type": "text", "text": instruction}]
    for img in frames:
        user_content.append({"type": "image", "image": img})
        
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": sample["alter"]}]}
        ]
    }


def load_hf_dataset_streaming(dataset_name: str, split: str, num_samples: int, 
                              save_dir: str = "videos", num_frames: int = 8):
    """Load dataset from HuggingFace Hub with streaming and download videos locally.
    
    Videos are cached locally, so subsequent runs will skip downloading.
    """
    
    print(f"📥 Loading {num_samples} samples from {dataset_name} ({split} split)...")
    
    dataset_stream = load_dataset(
        dataset_name,
        split=split,
        streaming=True
    )
    
    dataset_stream = dataset_stream.cast_column("video", Video(decode=False))
    
    os.makedirs(os.path.join(save_dir, split), exist_ok=True)
    
    local_data = []
    skipped_count = 0
    downloaded_count = 0
    
    for idx, item in enumerate(islice(dataset_stream, num_samples)):
        if idx % 50 == 0:
            print(f"  Processed {idx}/{num_samples} samples...")
            
        video_hf_path = item["video"]["path"]
        filename = video_hf_path.split("/")[-1]
        local_path = os.path.join(save_dir, split, filename)
        
        # Download if not exists (uses local cache if available)
        if not os.path.exists(local_path):
            try:
                hf_hub_download(
                    repo_id=dataset_name,
                    filename=f"{split}/{filename}",
                    repo_type="dataset",
                    local_dir=save_dir,
                    local_dir_use_symlinks=False
                )
                downloaded_count += 1
            except Exception as e:
                print(f"⚠️ Failed to download {filename}: {e}")
                continue
        else:
            skipped_count += 1
        
        # Replace video path in the item
        item["video"]["path"] = local_path
        local_data.append(item)
    
    print(f"✅ Loaded {len(local_data)} samples ({downloaded_count} downloaded, {skipped_count} from cache)")
    
    # Convert to Dataset
    dataset = Dataset.from_list(local_data)
    return dataset


def load_qved_dataset(json_path: str, num_frames: int = 8) -> Dataset:
    """Load QVED dataset from JSON (prepared by dataset.py) and convert to conversation format."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    dataset = []
    print(f"📥 Loading QVED dataset from {json_path}...")
    skipped = 0
    
    for idx, item in enumerate(data):
        if idx % 50 == 0:
            print(f"  Processed {idx}/{len(data)} samples...")
            
        video_path = item.get('video', '')
        
        # Handle relative paths - use videos directory
        if not os.path.isabs(video_path):
            # Extract just the filename from the path
            video_filename = os.path.basename(video_path)
            video_path = os.path.join('videos', video_filename)
        
        if not os.path.exists(video_path):
            print(f"⚠️ Skipping missing video: {video_path}")
            skipped += 1
            continue
        
        # Extract frames
        frames = downsample_video(video_path, num_frames)
        if not frames:
            skipped += 1
            continue
        
        # Get conversations
        conversations = item.get('conversations', [])
        if len(conversations) < 2:
            skipped += 1
            continue
        
        # Build prompt and response (handle both 'value' and 'from' keys)
        user_msg = conversations[0].get('value', '')
        assistant_msg = conversations[1].get('value', '')
        
        # Construct message format
        user_content = [{"type": "text", "text": user_msg}]
        for img in frames:
            user_content.append({"type": "image", "image": img})
        
        dataset.append({
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": [{"type": "text", "text": assistant_msg}]}
            ]
        })
    
    print(f"✅ Loaded {len(dataset)} samples ({skipped} skipped)")
    return Dataset.from_list(dataset)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma-3N with Unsloth FastVisionModel")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="unsloth/gemma-3n-E4B-it",
                       help="Model name from HuggingFace Hub")
    parser.add_argument("--load_in_4bit", action="store_true", 
                       help="Use 4bit quantization to reduce memory")
    
    # Dataset configuration
    parser.add_argument("--train_json", type=str, default="dataset/qved_train.json",
                       help="Path to local QVED training JSON (prepared by dataset.py)")
    parser.add_argument("--val_json", type=str, default="dataset/qved_val.json",
                       help="Path to local QVED validation JSON (prepared by dataset.py)")
    parser.add_argument("--run_eval", action="store_true",
                       help="Run evaluation on validation set during training")
    parser.add_argument("--eval_steps", type=int, default=50,
                       help="Run evaluation every N steps")
    parser.add_argument("--save_eval_csv", action="store_true",
                       help="Save evaluation results as CSV after training")
    parser.add_argument("--generate_report", action="store_true",
                       help="Generate Excel test report after training")
    parser.add_argument("--hf_dataset", type=str, default=None,
                       help="HuggingFace dataset name (optional, for streaming download)")
    parser.add_argument("--hf_split", type=str, default="train",
                       help="Dataset split to use")
    parser.add_argument("--num_samples", type=int, default=500,
                       help="Number of samples to load from streaming dataset")
    parser.add_argument("--video_save_dir", type=str, default="videos",
                       help="Directory to save downloaded videos")
    parser.add_argument("--num_frames", type=int, default=8,
                       help="Number of frames to extract from each video")
    
    # Training configuration
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="Output directory for checkpoints")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Per device batch size")
    parser.add_argument("--gradient_accumulation", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--max_seq_length", type=int, default=50000,
                       help="Maximum sequence length")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                       help="Warmup ratio")
    parser.add_argument("--max_grad_norm", type=float, default=0.3,
                       help="Max gradient norm")
    parser.add_argument("--weight_decay", type=float, default=0.001,
                       help="Weight decay")
    
    # LoRA configuration
    parser.add_argument("--lora_r", type=int, default=64,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=128,
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                       help="LoRA dropout")
    parser.add_argument("--finetune_vision_layers", action="store_true", default=True,
                       help="Finetune vision layers")
    parser.add_argument("--finetune_language_layers", action="store_true", default=True,
                       help="Finetune language layers")
    parser.add_argument("--finetune_attention_modules", action="store_true", default=True,
                       help="Finetune attention modules")
    parser.add_argument("--finetune_mlp_modules", action="store_true", default=True,
                       help="Finetune MLP modules")
    
    # Wandb configuration
    parser.add_argument("--wandb_project", type=str, default="Finetune-gemma3n",
                       help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                       help="Wandb entity (username or team name)")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="Wandb run name")
    parser.add_argument("--hf_token", type=str, default=None,
                       help="HuggingFace API token")
    
    # HuggingFace upload configuration
    parser.add_argument("--upload_to_hf", action="store_true",
                       help="Upload model to HuggingFace Hub after training")
    parser.add_argument("--hf_repo_name", type=str, default=None,
                       help="HuggingFace repository name (auto-generated if not provided)")
    parser.add_argument("--hf_private", action="store_true",
                       help="Make HuggingFace repository private")
    
    # DeepSpeed configuration
    parser.add_argument("--deepspeed_config", type=str, default=None,
                       help="Path to DeepSpeed config JSON file (e.g., scripts/zero.json)")
    
    args = parser.parse_args()
    
    # Check if DeepSpeed is available
    deepspeed_config = None
    if args.deepspeed_config:
        try:
            import deepspeed
            if os.path.exists(args.deepspeed_config):
                deepspeed_config = args.deepspeed_config
                print(f"⚡ DeepSpeed config loaded: {deepspeed_config}")
            else:
                print(f"⚠️ DeepSpeed config file not found: {args.deepspeed_config}")
        except ImportError:
            print("⚠️ DeepSpeed not installed. Training without DeepSpeed optimization.")
            print("   To use DeepSpeed, install it with: pip install deepspeed")
    
    # Login to HuggingFace if token provided
    if args.hf_token:
        print("🔐 Logging into HuggingFace...")
        login(token=args.hf_token)
    
    # Initialize wandb
    run_name = args.wandb_run_name or f"gemma-3n-finetune-{args.num_samples if args.hf_dataset else 'qved'}"
    wandb_config = {
        "project": args.wandb_project,
        "name": run_name,
        "dir": args.output_dir,
    }
    if args.wandb_entity:
        wandb_config["entity"] = args.wandb_entity
    
    wandb.init(**wandb_config)
    
    print("="*80)
    print("Gemma-3N Fine-tuning with Unsloth FastVisionModel")
    print("="*80)
    print(f"Model: {args.model_name}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size} x {args.gradient_accumulation} = {args.batch_size * args.gradient_accumulation}")
    print(f"LoRA r={args.lora_r}, alpha={args.lora_alpha}")
    print("="*80 + "\n")
    
    # Load model
    print("📦 Loading model and processor...")
    model, processor = FastVisionModel.from_pretrained(
        args.model_name,
        load_in_4bit=args.load_in_4bit,
        use_gradient_checkpointing="unsloth",
    )
    print("✅ Model loaded\n")
    
    # Apply chat template
    processor = get_chat_template(
        processor,
        "gemma-3n"
    )
    print("✅ Chat template applied\n")
    
    # Configure LoRA
    print("🔧 Configuring LoRA...")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=args.finetune_vision_layers,
        finetune_language_layers=args.finetune_language_layers,
        finetune_attention_modules=args.finetune_attention_modules,
        finetune_mlp_modules=args.finetune_mlp_modules,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
        target_modules="all-linear",
    )
    print("✅ LoRA configured\n")
    
    # Load dataset
    if args.train_json and os.path.exists(args.train_json):
        # Load from local QVED JSON (prepared by dataset.py)
        print(f"📥 Loading dataset from local JSON: {args.train_json}")
        train_dataset = load_qved_dataset(args.train_json, args.num_frames)
        
        # Load validation dataset if evaluation is enabled
        eval_dataset = None
        if args.run_eval and args.val_json and os.path.exists(args.val_json):
            print(f"📥 Loading validation dataset from: {args.val_json}")
            eval_dataset = load_qved_dataset(args.val_json, args.num_frames)
            print(f"✅ Validation dataset ready: {len(eval_dataset)} samples\n")
        
    elif args.hf_dataset:
        # Load from HuggingFace Hub (streaming download)
        print(f"📥 Loading dataset from HuggingFace: {args.hf_dataset}")
        raw_dataset = load_hf_dataset_streaming(
            args.hf_dataset,
            args.hf_split,
            args.num_samples,
            args.video_save_dir,
            args.num_frames
        )
        
        # Convert to conversation format
        print("🔄 Converting to conversation format...")
        converted_data = []
        for idx, sample in enumerate(raw_dataset):
            if idx % 50 == 0:
                print(f"  Converted {idx}/{len(raw_dataset)} samples...")
            conv = convert_to_conversation(sample, args.num_frames)
            if conv:
                converted_data.append(conv)
        
        train_dataset = Dataset.from_list(converted_data)
        print(f"✅ Dataset ready: {len(train_dataset)} samples\n")
        
    else:
        raise ValueError("Dataset not found. Please run 'python dataset.py download && python dataset.py prepare' first, or provide --hf_dataset")
    
    # Enable training mode
    FastVisionModel.for_training(model)
    
    # Create trainer
    print("🏋️ Setting up trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if args.run_eval else None,
        processing_class=processor.tokenizer,
        data_collator=UnslothVisionDataCollator(model, processor, max_seq_length=args.max_seq_length),
        args=SFTConfig(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_grad_norm=args.max_grad_norm,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            logging_steps=1,
            logging_first_step=True,
            save_strategy="steps",
            save_steps=50,
            eval_strategy="steps" if args.run_eval else "no",
            eval_steps=args.eval_steps if args.run_eval else None,
            load_best_model_at_end=True if args.run_eval else False,
            metric_for_best_model="eval_loss" if args.run_eval else None,
            optim="adamw_torch_fused",
            weight_decay=args.weight_decay,
            lr_scheduler_type="cosine",
            seed=3407,
            output_dir=args.output_dir,
            report_to="wandb",
            logging_nan_inf_filter=False,
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=args.max_seq_length,
            deepspeed=deepspeed_config,
        )
    )
    print("✅ Trainer ready\n")
    
    # Show memory stats before training
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"🖥️  GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"📊 {start_gpu_memory} GB of memory reserved.\n")
    
    # Train
    print("="*80)
    print("🚀 Starting training...")
    print("="*80 + "\n")
    
    trainer_stats = trainer.train()
    
    # Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    
    print("\n" + "="*80)
    print("📈 Training Statistics")
    print("="*80)
    print(f"⏱️  Training time: {trainer_stats.metrics['train_runtime']} seconds ({round(trainer_stats.metrics['train_runtime']/60, 2)} minutes)")
    print(f"💾 Peak reserved memory: {used_memory} GB ({used_percentage}% of max)")
    print(f"💾 Peak reserved memory for training: {used_memory_for_lora} GB ({lora_percentage}% of max)")
    print("="*80 + "\n")
    
    # Save model
    print("💾 Saving model...")
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    
    # Save merged model (16-bit)
    merged_dir = f"{args.output_dir}_merged_16bit"
    print(f"💾 Saving merged 16-bit model to {merged_dir}...")
    model.save_pretrained_merged(merged_dir, processor.tokenizer, save_method="merged_16bit")
    
    # Save merged model (4-bit, optional)
    if args.load_in_4bit:
        merged_4bit_dir = f"{args.output_dir}_merged_4bit"
        print(f"💾 Saving merged 4-bit model to {merged_4bit_dir}...")
        model.save_pretrained_merged(merged_4bit_dir, processor.tokenizer, save_method="merged_4bit")
    
    print("\n" + "="*80)
    print("✅ Fine-tuning completed!")
    print(f"📁 LoRA adapter saved to: {args.output_dir}")
    print(f"📁 Merged 16-bit model saved to: {merged_dir}")
    if args.load_in_4bit:
        print(f"📁 Merged 4-bit model saved to: {merged_4bit_dir}")
    print("="*80)
    
    # Run final evaluation and save results
    if args.run_eval and eval_dataset:
        print("\n" + "="*80)
        print("📊 Running final evaluation on validation set...")
        print("="*80 + "\n")
        
        try:
            # Run evaluation
            eval_results = trainer.evaluate(eval_dataset=eval_dataset)
            
            print("\n📈 Final Evaluation Results:")
            for key, value in eval_results.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
            
            # Save as CSV if requested
            if args.save_eval_csv:
                import csv
                csv_path = f"{args.output_dir}/eval_results.csv"
                print(f"\n💾 Saving evaluation results to: {csv_path}")
                
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Metric", "Value"])
                    for key, value in eval_results.items():
                        writer.writerow([key, value])
                
                print(f"✅ Evaluation results saved to CSV")
            
            # Generate Excel report if requested
            if args.generate_report:
                print("\n📊 Generating Excel test report...")
                try:
                    # Run inference on validation set first
                    print("  Running inference on validation set...")
                    predictions = []
                    
                    for idx, sample in enumerate(eval_dataset):
                        if idx >= 50:  # Limit to 50 samples for report
                            break
                        
                        try:
                            messages = sample["messages"]
                            inputs = processor.apply_chat_template(
                                messages[:-1],  # Exclude assistant response
                                add_generation_prompt=True,
                                tokenize=True,
                                return_dict=True,
                                return_tensors="pt",
                            ).to("cuda")
                            
                            with torch.no_grad():
                                outputs = model.generate(
                                    **inputs,
                                    max_new_tokens=256,
                                    temperature=0.1,
                                    do_sample=False,
                                )
                            
                            input_length = inputs.input_ids.shape[1]
                            new_tokens = outputs[0][input_length:]
                            prediction = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
                            ground_truth = messages[-1]["content"][0]["text"]
                            
                            predictions.append({
                                "video_path": f"sample_{idx}",
                                "ground_truth": ground_truth,
                                "prediction": prediction.strip(),
                                "status": "success",
                                "error": ""
                            })
                        except Exception as e:
                            predictions.append({
                                "video_path": f"sample_{idx}",
                                "ground_truth": "",
                                "prediction": "",
                                "status": "error",
                                "error": str(e)
                            })
                    
                    # Save predictions JSON
                    predictions_path = f"{args.output_dir}/eval_predictions.json"
                    with open(predictions_path, 'w') as f:
                        json.dump(predictions, f, indent=2)
                    
                    print(f"  ✅ Predictions saved to: {predictions_path}")
                    
                    # Generate Excel report
                    report_path = f"{args.output_dir}/eval_report.xlsx"
                    
                    # Import and run report generation
                    import sys
                    sys.path.insert(0, 'utils')
                    from generate_test_report import create_excel_report
                    
                    create_excel_report(predictions, report_path, use_bert=True)
                    print(f"  ✅ Excel report generated: {report_path}")
                    
                except Exception as e:
                    print(f"  ⚠️ Warning: Failed to generate report: {e}")
                    print(f"  You can manually generate it later with:")
                    print(f"    python utils/generate_test_report.py --predictions {args.output_dir}/eval_predictions.json")
        
        except Exception as e:
            print(f"⚠️ Warning: Evaluation failed: {e}")
    
    # Upload to HuggingFace if requested
    if args.upload_to_hf:
        print("\n" + "="*80)
        print("🚀 Uploading model to HuggingFace Hub...")
        print("="*80 + "\n")
        
        try:
            from huggingface_hub import HfApi, create_repo, upload_folder
            from datetime import datetime
            
            # Generate repo name if not provided
            if args.hf_repo_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                repo_name = f"gemma3n-E4B-finetune-{timestamp}"
            else:
                repo_name = args.hf_repo_name
            
            # Get user info
            api = HfApi()
            user_info = api.whoami()
            username = user_info['name']
            repo_id = f"{username}/{repo_name}"
            
            print(f"📦 Creating repository: {repo_id}")
            create_repo(repo_id=repo_id, repo_type="model", private=args.hf_private, exist_ok=True)
            
            # Upload merged 16-bit model (recommended for inference)
            print(f"📤 Uploading merged 16-bit model from {merged_dir}...")
            upload_folder(
                folder_path=merged_dir,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Upload Gemma-3N-E4B fine-tuned model (16-bit merged)",
                ignore_patterns=["*.py", "__pycache__", "*.pyc", "runs/*", "wandb/*"],
            )
            
            repo_url = f"https://huggingface.co/{repo_id}"
            print(f"\n✅ Model uploaded successfully!")
            print(f"🔗 Repository: {repo_url}")
            print(f"\nTo use this model:")
            print(f"  from unsloth import FastVisionModel")
            print(f"  model, processor = FastVisionModel.from_pretrained('{repo_id}')")
            print("="*80)
            
        except Exception as e:
            print(f"\n⚠️ Warning: Failed to upload to HuggingFace: {e}")
            print(f"You can manually upload later using:")
            print(f"  python utils/hf_upload.py --model_path {merged_dir}")
    
    wandb.finish()


if __name__ == "__main__":
    main()
