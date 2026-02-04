#!/usr/bin/env python3
"""CLI script for fine-tuning Gemma 3n models."""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logging_utils import setup_logging
from utils.finetune import create_trainer, run_training
from utils.config import AppConfig


def main():
    """Main CLI entrypoint."""
    config = AppConfig()

    parser = argparse.ArgumentParser(
        description="Fine-tune Gemma 3n with LoRA/QLoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example JSONL format:
  {"prompt": "What is this?", "response": "This is a cat.", "image_path": "cat.jpg"}
  {"prompt": "Describe the video", "response": "A person running.", "video_path": "run.mp4"}
  {"prompt": "Explain AI", "response": "AI stands for..."}

Usage example:
  python scripts/run_finetune.py \\
    --train_jsonl data/train.jsonl \\
    --output_dir outputs/gemma3n-finetuned \\
    --method qlora \\
    --epochs 3 \\
    --batch_size 2
        """
    )

    # Model settings
    parser.add_argument(
        "--model_id",
        type=str,
        default=config.default_model_id,
        help=f"Hugging Face model ID (default: {config.default_model_id})"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        default=config.default_dtype,
        help=f"Model dtype (default: {config.default_dtype})"
    )

    # Dataset settings
    parser.add_argument(
        "--train_jsonl",
        type=str,
        required=True,
        help="Path to training JSONL file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for checkpoints and final model"
    )

    # PEFT settings
    parser.add_argument(
        "--method",
        type=str,
        choices=["lora", "qlora"],
        default="qlora",
        help="Fine-tuning method (default: qlora)"
    )
    parser.add_argument(
        "--quant",
        type=str,
        choices=["none", "4bit"],
        default="4bit",
        help="Quantization mode (default: 4bit for QLoRA)"
    )
    parser.add_argument(
        "--peft_r",
        type=int,
        default=16,
        help="LoRA rank (default: 16)"
    )
    parser.add_argument(
        "--peft_alpha",
        type=int,
        default=32,
        help="LoRA alpha (default: 32)"
    )
    parser.add_argument(
        "--peft_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout (default: 0.05)"
    )

    # Training settings
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Per-device batch size (default: 1)"
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=8,
        help="Gradient accumulation steps (default: 8)"
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
        help="Warmup ratio (default: 0.03)"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay (default: 0.0)"
    )

    # Video settings
    parser.add_argument(
        "--num_frames",
        type=int,
        default=config.default_num_frames,
        help=f"Number of frames to sample from videos (default: {config.default_num_frames})"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Sampling rate for videos (frames/sec). If set, overrides num_frames."
    )

    # Logging settings
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log every N steps (default: 10)"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=200,
        help="Save checkpoint every N steps (default: 200)"
    )

    # Other settings
    parser.add_argument(
        "--seed",
        type=int,
        default=config.seed,
        help=f"Random seed (default: {config.seed})"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase verbosity (can be repeated)"
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(verbosity=args.verbose)

    # Validate files
    if not os.path.exists(args.train_jsonl):
        logger.error(f"Training file not found: {args.train_jsonl}")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        # Create trainer
        logger.info("Setting up training...")
        trainer = create_trainer(
            model_id=args.model_id,
            train_jsonl=args.train_jsonl,
            output_dir=args.output_dir,
            method=args.method,
            quant=args.quant,
            num_train_epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            max_seq_len=args.max_seq_len,
            num_frames=args.num_frames,
            fps=args.fps,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            seed=args.seed,
            peft_r=args.peft_r,
            peft_alpha=args.peft_alpha,
            peft_dropout=args.peft_dropout,
            dtype=args.dtype
        )

        # Run training
        run_training(trainer, args.output_dir)

        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETE!")
        logger.info("="*80)
        logger.info(f"Model saved to: {args.output_dir}")
        logger.info("\nTo load the fine-tuned model:")
        logger.info("  from transformers import AutoModelForCausalLM, AutoProcessor")
        logger.info("  from peft import PeftModel")
        logger.info(f"  model = AutoModelForCausalLM.from_pretrained('{args.model_id}')")
        logger.info(f"  model = PeftModel.from_pretrained(model, '{args.output_dir}')")
        logger.info(f"  processor = AutoProcessor.from_pretrained('{args.model_id}')")
        logger.info("="*80 + "\n")

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=args.verbose >= 2)
        sys.exit(1)


if __name__ == "__main__":
    main()
