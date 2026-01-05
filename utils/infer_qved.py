#!/usr/bin/env python3
"""
QVED Inference with Google/Gemma-3n-E2B

This script runs inference on a single video using the Google/Gemma-3n-E2B
Mobile-VideoGPT model (or a finetuned LoRA adapter based on it).

Usage:
    python infer_qved_gemma.py \
        --video_path sample_videos/00000340.mp4 \
        --prompt "Describe this video"
"""

import sys
import os
import warnings
import logging
import argparse

os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings("ignore")

logging.getLogger('mmengine').setLevel(logging.CRITICAL)
logging.getLogger('transformers').setLevel(logging.CRITICAL)
logging.getLogger('transformers.modeling_utils').setLevel(logging.CRITICAL)

import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import PeftModel

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from mobilevideogpt.utils import preprocess_input


def load_model(pretrained_path: str, device: str = "cuda", base_model: str = "google/gemma-3n-E2B"):
    """Loads the pre-trained model and tokenizer."""
    is_lora_checkpoint = False
    adapter_path = pretrained_path

    # Detect if path contains LoRA adapters
    if "checkpoint-" in pretrained_path or os.path.exists(os.path.join(pretrained_path, "adapter_config.json")):
        is_lora_checkpoint = True

    if is_lora_checkpoint:
        print(f"Loading LoRA adapters from: {adapter_path}")
        print(f"Base model: {base_model}")

        config = AutoConfig.from_pretrained(base_model)
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            config=config,
            torch_dtype=torch.float16
        )
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    else:
        print(f"Loading full model from: {pretrained_path}")
        config = AutoConfig.from_pretrained(pretrained_path)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_path,
            config=config,
            torch_dtype=torch.float16
        )

    model.to(device)
    return model, tokenizer


def run_inference(model, tokenizer, video_path: str, prompt: str, device: str = "cuda", max_new_tokens: int = 512):
    """Runs inference on the given video file."""
    input_ids, video_frames, context_frames, stop_str = preprocess_input(
        model, tokenizer, video_path, prompt
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=torch.stack(video_frames, dim=0).half().to(device),
            context_images=torch.stack(context_frames, dim=0).half().to(device),
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)].strip()

    return outputs


def main():
    parser = argparse.ArgumentParser(description="QVED Inference with Google/Gemma-3n-E2B")
    parser.add_argument(
        "--model_path",
        type=str,
        default="google/gemma-3n-E2B",
        help="Path to the model (HuggingFace model ID or local checkpoint directory)"
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="sample_videos/00000340.mp4",
        help="Path to the input video file"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Please evaluate the exercise form shown. What mistakes, if any, are present, and what corrections would you recommend?",
        help="Prompt for the model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (cuda/cpu)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="google/gemma-3n-E2B",
        help="Base model to use when loading LoRA adapters"
    )

    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"❌ Error: Video file not found: {args.video_path}")
        sys.exit(1)

    print(f"📦 Loading model from: {args.model_path}")
    model, tokenizer = load_model(args.model_path, device=args.device, base_model=args.base_model)

    print(f"🎥 Processing video: {args.video_path}")
    print(f"💬 Prompt: {args.prompt}")
    print("\n" + "="*80)

    output = run_inference(
        model,
        tokenizer,
        args.video_path,
        args.prompt,
        device=args.device,
        max_new_tokens=args.max_new_tokens
    )

    print("🤖 Mobile-VideoGPT Output:")
    print(output)
    print("="*80)


if __name__ == "__main__":
    main()
