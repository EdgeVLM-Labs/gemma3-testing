import os
import sys
import warnings
import logging
from pathlib import Path
from PIL import Image
import torch

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
from transformers import TextStreamer

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
def load_model(model_name="unsloth/gemma-3n-E2B", device="cuda"):
    """Load Gemma3N E2B model."""
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        dtype=None,          # auto-detect
        max_seq_length=1024,
        load_in_4bit=False,  # full precision
        full_finetuning=False
    )
    model.to(device)
    return model, tokenizer

# -----------------------------
# Inference Function
# -----------------------------
def run_inference(model, tokenizer, video_path, prompt, max_new_tokens=256):
    """Run Gemma3N inference on a video with a text prompt."""
    frames = extract_frames(video_path, num_frames=8)
    
    # Prepare messages for Gemma3N
    messages = [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": img} for img in frames],
                {"type": "text", "text": prompt}
            ],
        }
    ]

    # Tokenize and generate output
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")

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
    pretrained_model = "unsloth/gemma-3n-E2B"
    video_path = "sample_videos/00000340.mp4"
    prompt = (
        "You are an assistive navigation system for a visually impaired user. "
        "Analyze the provided video from the user's forward perspective. "
        "Identify all the immediate, high-risk obstructions and provide a single, actionable safety alert."
    )

    model, tokenizer = load_model(pretrained_model)
    output = run_inference(model, tokenizer, video_path, prompt)
    print("🤖 Gemma3N-E2B Output:\n", output)


if __name__ == "__main__":
    main()
