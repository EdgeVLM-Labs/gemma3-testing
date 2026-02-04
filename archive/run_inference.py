#!/usr/bin/env python3
"""CLI script for running Gemma 3n inference."""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logging_utils import setup_logging
from utils.inference import run_inference_text, run_inference_image, run_inference_video
from utils.config import AppConfig


def main():
    """Main CLI entrypoint."""
    config = AppConfig()

    parser = argparse.ArgumentParser(
        description="Run Gemma 3n inference (text/image/video)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Text-only
  python scripts/run_inference.py --mode text --prompt "Explain quantum computing"

  # Image + text
  python scripts/run_inference.py --mode image --prompt "Describe this image" --image_path photo.jpg

  # Video + text
  python scripts/run_inference.py --mode video --prompt "What happens in this video?" --video_path clip.mp4 --num_frames 8
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
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default=config.default_device,
        help=f"Device (default: {config.default_device})"
    )
    parser.add_argument(
        "--quant",
        type=str,
        choices=["none", "4bit"],
        default="none",
        help="Quantization mode (default: none)"
    )

    # Inference mode
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["text", "image", "video"],
        help="Inference mode"
    )

    # Input settings
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        help="Path to image file (required for image mode)"
    )
    parser.add_argument(
        "--video_path",
        type=str,
        help="Path to video file (required for video mode)"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=config.default_num_frames,
        help=f"Number of frames to sample from video (default: {config.default_num_frames})"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Sampling rate for video (frames/sec). If set, overrides num_frames."
    )
    parser.add_argument(
        "--video_method",
        type=str,
        choices=["opencv", "decord"],
        default="opencv",
        help="Video sampling method (default: opencv)"
    )

    # Generation settings
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=config.default_max_new_tokens,
        help=f"Maximum tokens to generate (default: {config.default_max_new_tokens})"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=config.default_temperature,
        help=f"Sampling temperature (default: {config.default_temperature})"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=config.default_top_p,
        help=f"Nucleus sampling parameter (default: {config.default_top_p})"
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Use sampling instead of greedy decoding"
    )

    # Logging
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

    # Validate required arguments by mode
    if args.mode == "image" and not args.image_path:
        parser.error("--image_path is required for image mode")

    if args.mode == "video" and not args.video_path:
        parser.error("--video_path is required for video mode")

    try:
        # Run inference based on mode
        if args.mode == "text":
            result = run_inference_text(
                model_id=args.model_id,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.do_sample,
                device=args.device,
                dtype=args.dtype,
                quant=args.quant
            )

        elif args.mode == "image":
            result = run_inference_image(
                model_id=args.model_id,
                prompt=args.prompt,
                image_path=args.image_path,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.do_sample,
                device=args.device,
                dtype=args.dtype,
                quant=args.quant
            )

        elif args.mode == "video":
            result = run_inference_video(
                model_id=args.model_id,
                prompt=args.prompt,
                video_path=args.video_path,
                num_frames=args.num_frames,
                fps=args.fps,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.do_sample,
                device=args.device,
                dtype=args.dtype,
                quant=args.quant,
                video_method=args.video_method
            )

        # Print output
        print("\n" + "="*80)
        print("GENERATED OUTPUT:")
        print("="*80)
        print(result["text"])
        print("="*80)
        print(f"Time taken: {result['time_taken']:.2f}s")
        print("="*80 + "\n")

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=args.verbose >= 2)
        sys.exit(1)


if __name__ == "__main__":
    main()
