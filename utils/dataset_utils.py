"""Dataset utilities for fine-tuning Gemma 3n."""

import json
import logging
import os
from typing import List, Dict, Any, Optional
from torch.utils.data import Dataset
from PIL import Image

from .video_utils import sample_video_frames
from .prompt_utils import build_chat_messages

logger = logging.getLogger("gemma3n")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Load JSONL file into list of dictionaries.

    Args:
        path: Path to JSONL file

    Returns:
        List of sample dictionaries

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is not valid JSONL
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSONL file not found: {path}")

    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                sample = json.loads(line)
                samples.append(sample)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_num}: {e}")

    logger.info(f"Loaded {len(samples)} samples from {path}")
    return samples


class VLMSFTDataset(Dataset):
    """
    Dataset for vision-language model supervised fine-tuning.

    Supports three sample types:
    1. Text-only: {"prompt": "...", "response": "..."}
    2. Image: {"image_path": "...", "prompt": "...", "response": "..."}
    3. Video: {"video_path": "...", "prompt": "...", "response": "..."}

    The dataset handles loading images/videos on-the-fly and returns
    normalized dictionaries with messages, images, and text fields.
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        num_frames: int = 8,
        fps: Optional[float] = None,
        video_method: str = "opencv",
        base_path: Optional[str] = None
    ):
        """
        Initialize dataset.

        Args:
            samples: List of sample dictionaries from JSONL
            num_frames: Number of frames to sample from videos
            fps: Optional sampling rate for videos
            video_method: Video sampling method ("opencv" or "decord")
            base_path: Optional base path for resolving relative paths
        """
        self.samples = samples
        self.num_frames = num_frames
        self.fps = fps
        self.video_method = video_method
        self.base_path = base_path or ""

        logger.info(f"Created VLMSFTDataset with {len(samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample by index.

        Returns:
            Dictionary with keys:
                - messages: List of message dicts (HF format)
                - images: List of PIL Images (empty if text-only)
                - text_prompt: User prompt text
                - text_response: Assistant response text
        """
        sample = self.samples[idx]

        # Extract text fields
        prompt = sample.get("prompt", "")
        response = sample.get("response", "")

        # Load images/video if present
        images = []

        # Handle video
        if "video_path" in sample:
            video_path = self._resolve_path(sample["video_path"])
            try:
                images = sample_video_frames(
                    video_path,
                    num_frames=self.num_frames,
                    fps=self.fps,
                    method=self.video_method
                )
            except Exception as e:
                logger.error(f"Failed to load video {video_path}: {e}")
                # Continue with empty images

        # Handle image
        elif "image_path" in sample:
            image_path = self._resolve_path(sample["image_path"])
            try:
                img = Image.open(image_path).convert("RGB")
                images = [img]
            except Exception as e:
                logger.error(f"Failed to load image {image_path}: {e}")
                # Continue with empty images

        # Build chat messages
        messages = build_chat_messages(prompt, images=images if images else None)

        # Add assistant response to messages
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": response}]
        })

        return {
            "messages": messages,
            "images": images,
            "text_prompt": prompt,
            "text_response": response
        }

    def _resolve_path(self, path: str) -> str:
        """Resolve relative paths using base_path."""
        if os.path.isabs(path):
            return path
        return os.path.join(self.base_path, path)


def create_sft_dataset(
    jsonl_path: str,
    num_frames: int = 8,
    fps: Optional[float] = None,
    video_method: str = "opencv"
) -> VLMSFTDataset:
    """
    Create SFT dataset from JSONL file.

    Args:
        jsonl_path: Path to JSONL file
        num_frames: Number of frames to sample from videos
        fps: Optional sampling rate for videos
        video_method: Video sampling method

    Returns:
        VLMSFTDataset instance
    """
    samples = load_jsonl(jsonl_path)
    base_path = os.path.dirname(os.path.abspath(jsonl_path))

    return VLMSFTDataset(
        samples=samples,
        num_frames=num_frames,
        fps=fps,
        video_method=video_method,
        base_path=base_path
    )
