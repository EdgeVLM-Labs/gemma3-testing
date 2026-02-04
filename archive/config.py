"""Configuration utilities for Gemma 3n inference and fine-tuning."""

from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class AppConfig:
    """Default configuration for Gemma 3n applications."""

    # Model settings
    default_model_id: str = "google/gemma-3n-E2B"
    default_dtype: str = "bf16"
    default_device: str = "cuda"

    # Video settings
    default_num_frames: int = 8
    default_fps: Optional[float] = 1.0

    # Generation settings
    default_max_new_tokens: int = 256
    default_temperature: float = 0.2
    default_top_p: float = 0.9

    # Training settings
    seed: int = 42


def parse_dtype(dtype_str: str) -> torch.dtype:
    """
    Parse dtype string to torch dtype.

    Args:
        dtype_str: One of "bf16", "fp16", "fp32", "float32", "float16", "bfloat16"

    Returns:
        torch.dtype: Corresponding PyTorch dtype

    Raises:
        ValueError: If dtype string is not recognized
    """
    dtype_map = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }

    dtype_str = dtype_str.lower()
    if dtype_str not in dtype_map:
        raise ValueError(
            f"Unsupported dtype '{dtype_str}'. "
            f"Choose from: {', '.join(dtype_map.keys())}"
        )

    return dtype_map[dtype_str]
