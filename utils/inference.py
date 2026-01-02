"""Inference utilities for Gemma 3n multimodal models."""

import logging
import time
from typing import List, Dict, Any, Optional
import torch
from PIL import Image

from .model_utils import load_model_for_inference, load_processor
from .prompt_utils import build_chat_messages, apply_chat_template_if_available
from .video_utils import sample_video_frames
from .config import parse_dtype

logger = logging.getLogger("gemma3n")


def generate(
    model: Any,
    processor: Any,
    messages: List[Dict[str, Any]],
    images: Optional[List[Image.Image]] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.9,
    do_sample: bool = False,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Generate text from model given messages and optional images.

    Args:
        model: Loaded model instance
        processor: Processor or tokenizer instance
        messages: List of message dictionaries (HF format)
        images: Optional list of PIL images
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (if do_sample=True)
        top_p: Nucleus sampling parameter (if do_sample=True)
        do_sample: Whether to use sampling (vs greedy)
        device: Device for inference

    Returns:
        Dictionary with 'text' (generated text) and 'time_taken' (seconds)

    Raises:
        RuntimeError: If generation fails
    """
    start_time = time.time()

    try:
        # Per Google documentation, use processor.apply_chat_template for Gemma 3n
        # This properly handles images embedded in the messages
        logger.debug("Using processor.apply_chat_template with embedded images")
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )

        # Move inputs to device and convert to model dtype
        inputs = inputs.to(model.device, dtype=model.dtype)

        # Prepare generation config
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }

        if do_sample:
            generation_config["temperature"] = temperature
            generation_config["top_p"] = top_p

        # Generate with mixed precision if CUDA
        use_amp = device == "cuda" and torch.cuda.is_available()

        with torch.no_grad():
            if use_amp:
                with torch.amp.autocast('cuda', dtype=model.dtype):
                    outputs = model.generate(**inputs, **generation_config)
            else:
                outputs = model.generate(**inputs, **generation_config)

        # Decode output
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        generated_text = processor.decode(generated_tokens, skip_special_tokens=True)

        time_taken = time.time() - start_time

        return {
            "text": generated_text,
            "time_taken": time_taken
        }

    except Exception as e:
        raise RuntimeError(f"Generation failed: {e}")


def run_inference_text(
    model_id: str,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.9,
    do_sample: bool = False,
    device: str = "cuda",
    dtype: str = "bf16",
    quant: str = "none"
) -> Dict[str, Any]:
    """
    Run text-only inference.

    Args:
        model_id: Hugging Face model identifier
        prompt: Text prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        do_sample: Whether to use sampling
        device: Target device
        dtype: Model dtype string
        quant: Quantization mode

    Returns:
        Dictionary with 'text' and 'time_taken'
    """
    logger.info("Running text-only inference")
    logger.info(f"Prompt: {prompt}")

    # Parse dtype
    torch_dtype = parse_dtype(dtype)

    # Load model and processor
    processor = load_processor(model_id)
    model = load_model_for_inference(
        model_id=model_id,
        device=device,
        dtype=torch_dtype,
        quant=quant
    )

    # Build messages
    messages = build_chat_messages(prompt)

    # Generate
    result = generate(
        model=model,
        processor=processor,
        messages=messages,
        images=None,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        device=device
    )

    logger.info(f"Generated in {result['time_taken']:.2f}s")
    return result


def run_inference_image(
    model_id: str,
    prompt: str,
    image_path: str,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.9,
    do_sample: bool = False,
    device: str = "cuda",
    dtype: str = "bf16",
    quant: str = "none"
) -> Dict[str, Any]:
    """
    Run image + text inference.

    Args:
        model_id: Hugging Face model identifier
        prompt: Text prompt
        image_path: Path to image file
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        do_sample: Whether to use sampling
        device: Target device
        dtype: Model dtype string
        quant: Quantization mode

    Returns:
        Dictionary with 'text' and 'time_taken'
    """
    logger.info("Running image+text inference")
    logger.info(f"Image: {image_path}")
    logger.info(f"Prompt: {prompt}")

    # Load image
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"Failed to load image: {e}")

    # Parse dtype
    torch_dtype = parse_dtype(dtype)

    # Load model and processor
    processor = load_processor(model_id)
    model = load_model_for_inference(
        model_id=model_id,
        device=device,
        dtype=torch_dtype,
        quant=quant
    )

    # Build messages with image
    messages = build_chat_messages(prompt, images=[image])

    # Generate
    result = generate(
        model=model,
        processor=processor,
        messages=messages,
        images=[image],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        device=device
    )

    logger.info(f"Generated in {result['time_taken']:.2f}s")
    return result


def run_inference_video(
    model_id: str,
    prompt: str,
    video_path: str,
    num_frames: int = 8,
    fps: Optional[float] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.9,
    do_sample: bool = False,
    device: str = "cuda",
    dtype: str = "bf16",
    quant: str = "none",
    video_method: str = "opencv"
) -> Dict[str, Any]:
    """
    Run video (frames) + text inference.

    Args:
        model_id: Hugging Face model identifier
        prompt: Text prompt
        video_path: Path to video file
        num_frames: Number of frames to sample
        fps: Optional sampling rate
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        do_sample: Whether to use sampling
        device: Target device
        dtype: Model dtype string
        quant: Quantization mode
        video_method: Video sampling method

    Returns:
        Dictionary with 'text' and 'time_taken'
    """
    logger.info("Running video+text inference")
    logger.info(f"Video: {video_path}")
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Sampling {num_frames} frames (fps={fps})")

    # Sample video frames
    frames = sample_video_frames(
        video_path=video_path,
        num_frames=num_frames,
        fps=fps,
        method=video_method
    )

    logger.info(f"Sampled {len(frames)} frames")

    # Parse dtype
    torch_dtype = parse_dtype(dtype)

    # Load model and processor
    processor = load_processor(model_id)
    model = load_model_for_inference(
        model_id=model_id,
        device=device,
        dtype=torch_dtype,
        quant=quant
    )

    # Build messages with frames
    messages = build_chat_messages(prompt, images=frames)

    # Generate
    result = generate(
        model=model,
        processor=processor,
        messages=messages,
        images=frames,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        device=device
    )

    logger.info(f"Generated in {result['time_taken']:.2f}s")
    return result
