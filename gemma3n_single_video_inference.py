import os
import warnings
import logging
import argparse
from typing import Optional, List

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

logging.getLogger("mmengine").setLevel(logging.CRITICAL)
logging.getLogger("transformers").setLevel(logging.CRITICAL)
logging.getLogger("transformers.modeling_utils").setLevel(logging.CRITICAL)

import torch
import cv2
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM


def _get_hf_token(explicit_token: Optional[str] = None) -> Optional[str]:
    if explicit_token:
        return explicit_token
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HF_ACCESS_TOKEN")
    )


def extract_frames(video_path: str, num_frames: int = 8) -> List[Image.Image]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise ValueError(f"Video has no frames: {video_path}")

    indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    frames: List[Image.Image] = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frames.append(img)

    cap.release()

    if not frames:
        raise ValueError(f"Could not decode frames from: {video_path}")
    return frames


def load_model(model_id_or_path: str, device: str = "cuda", token: Optional[str] = None):
    """Loads Gemma-3N model, tokenizer, and processor."""
    model_kwargs = {}
    if token:
        model_kwargs["token"] = token

    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, use_fast=False, **model_kwargs)
    processor = AutoProcessor.from_pretrained(model_id_or_path, **model_kwargs)

    # Prefer bf16 on modern GPUs; fall back to fp16/cpu.
    if device.startswith("cuda") and torch.cuda.is_available():
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        torch_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        torch_dtype=torch_dtype,
        device_map=None,
        **model_kwargs,
    )

    if device.startswith("cuda") and torch.cuda.is_available():
        model = model.to("cuda")
    else:
        model = model.to("cpu")

    model.eval()
    return model, tokenizer, processor


def run_inference(
    model,
    tokenizer,
    processor,
    video_path: str,
    prompt: str,
    max_new_tokens: int = 256,
    num_frames: int = 8,
):
    frames = extract_frames(video_path, num_frames=num_frames)

    inputs = processor(
        text=prompt,
        images=frames,
        return_tensors="pt",
        padding=True,
    )

    # Move inputs to the model device
    try:
        inputs = inputs.to(model.device)
    except AttributeError:
        for k, v in list(inputs.items()):
            if hasattr(v, "to"):
                inputs[k] = v.to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="Gemma-3N single-video inference")
    parser.add_argument("--model", default="google/gemma-3n-E2B", help="HF model id or local path")
    parser.add_argument("--video", default="sample_videos/00000340.mp4", help="Path to video")
    parser.add_argument(
        "--prompt",
        default=(
            "Please evaluate the exercise form shown. "
            "What mistakes, if any, are present, and what corrections would you recommend?"
        ),
        help="Prompt text",
    )
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument(
        "--hf_token",
        default=None,
        help="HF token (or set HF_TOKEN / HUGGINGFACE_HUB_TOKEN env var)",
    )
    args = parser.parse_args()

    token = _get_hf_token(args.hf_token)

    try:
        model, tokenizer, processor = load_model(args.model, device=args.device, token=token)
    except Exception as e:
        msg = str(e)
        if "gated" in msg.lower() or "403" in msg:
            raise SystemExit(
                "Cannot download model (likely gated / requires access).\n"
                "- Request access and accept terms on the model page\n"
                "- Then login: `hf auth login`\n"
                "- Or pass token: `HF_TOKEN=... python gemma3n_single_video_inference.py --model google/gemma-3n-E2B`\n"
                "- Or use a public mirror (if available): `--model unsloth/gemma-3n-E2B`\n"
            )
        raise

    output = run_inference(
        model,
        tokenizer,
        processor,
        video_path=args.video,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        num_frames=args.num_frames,
    )
    print("🤖 Gemma-3N Output:", output)


if __name__ == "__main__":
    main()
