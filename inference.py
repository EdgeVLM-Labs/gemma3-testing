import os
import sys
import warnings
import logging
from pathlib import Path
from PIL import Image
import torch
import argparse

# -----------------------------
# Suppress warnings & logs
# -----------------------------
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings("ignore")
logging.getLogger('mmengine').setLevel(logging.CRITICAL)
logging.getLogger('transformers').setLevel(logging.CRITICAL)

# -----------------------------
# Gemma3N imports
# -----------------------------
from unsloth import FastModel
from transformers import TextStreamer, AutoProcessor


def _get_hf_token(explicit_token: str | None = None) -> str | None:
    if explicit_token:
        return explicit_token
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HF_ACCESS_TOKEN")
    )

# -----------------------------
# Video processing
# -----------------------------
import cv2

def extract_frames(video_path, num_frames=8):
    """Extracts 'num_frames' evenly spaced frames from a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frames.append(img)

    cap.release()
    return frames

# -----------------------------
# Model Loading
# -----------------------------
def load_model(model_name: str, device: str = "cuda", token: str | None = None):
    """Load Gemma3N E2B model."""
    load_kwargs = dict(
        model_name=model_name,
        dtype=None,          # auto-detect
        max_seq_length=1024,
        load_in_4bit=False,  # full precision
        full_finetuning=False,
    )

    # Some hubs are gated. If the user provides a token, pass it through.
    # Different library versions use different kwarg names, so we try both.
    if token:
        try:
            model, tokenizer = FastModel.from_pretrained(**load_kwargs, token=token)
        except TypeError:
            model, tokenizer = FastModel.from_pretrained(**load_kwargs, use_auth_token=token)
    else:
        model, tokenizer = FastModel.from_pretrained(**load_kwargs)

    if token:
        try:
            processor = AutoProcessor.from_pretrained(model_name, token=token)
        except TypeError:
            processor = AutoProcessor.from_pretrained(model_name, use_auth_token=token)
    else:
        processor = AutoProcessor.from_pretrained(model_name)

    model.to(device)
    return model, tokenizer, processor

# -----------------------------
# Inference Function
# -----------------------------
def run_inference(model, tokenizer, processor, video_path, prompt, max_new_tokens=256):
    """Run Gemma3N inference on a video with a text prompt."""
    frames = extract_frames(video_path, num_frames=8)
    
    # Prepare input using processor
    inputs = processor(
        text=prompt,
        images=frames,
        return_tensors="pt",
        padding=True
    )

    # Move to model device
    try:
        inputs = inputs.to(model.device)
    except AttributeError:
        for key, value in list(inputs.items()):
            if hasattr(value, "to"):
                inputs[key] = value.to(model.device)

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            streamer=streamer
        )

    # Convert to text
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Run Gemma-3N inference on a single video")
    parser.add_argument(
        "--model",
        default="unsloth/gemma-3n-E2B",
        help="HF model id or local path. Default: unsloth/gemma-3n-E2B",
    )
    parser.add_argument(
        "--video",
        default="sample_videos/00000340.mp4",
        help="Path to input video file",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "You are an assistive navigation system for a visually impaired user. "
            "Analyze the provided video from the user's forward perspective. "
            "Identify all the immediate, high-risk obstructions and provide a single, actionable safety alert."
        ),
        help="Text prompt",
    )
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument(
        "--hf_token",
        default=None,
        help="HuggingFace token (or set HF_TOKEN/HUGGINGFACE_HUB_TOKEN env var)",
    )
    args = parser.parse_args()

    token = _get_hf_token(args.hf_token)

    try:
        model, tokenizer, processor = load_model(args.model, device=args.device, token=token)
    except Exception as e:
        msg = str(e)
        if "gated repo" in msg.lower() or "403" in msg or "GatedRepoError" in msg:
            raise SystemExit(
                "Model download blocked (gated repo / 403).\n"
                "- Option A: login and accept the model terms: `hf auth login` (or `huggingface-cli login`)\n"
                "- Option B: request access on the model page, then rerun\n"
                "- Option C: use the public mirror: `--model unsloth/gemma-3n-E2B`\n"
                "- Option D: pass a token: `HF_TOKEN=... python inference.py --model google/gemma-3n-E2B`\n"
            )
        raise

    output = run_inference(
        model,
        tokenizer,
        processor,
        video_path=args.video,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
    )
    print("\n🤖 Output:\n", output)


if __name__ == "__main__":
    main()
