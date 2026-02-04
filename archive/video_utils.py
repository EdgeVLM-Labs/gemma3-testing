"""Video frame sampling utilities for Gemma 3n."""

import os
import logging
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from PIL import Image

logger = logging.getLogger("gemma3n")


def get_video_metadata(video_path: str) -> Dict[str, Any]:
    """
    Get metadata from video file.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with 'duration', 'fps', 'frame_count', 'width', 'height'

    Raises:
        FileNotFoundError: If video file doesn't exist
        RuntimeError: If video cannot be read
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    try:
        import cv2
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0

        cap.release()

        return {
            "duration": duration,
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height
        }
    except ImportError:
        raise RuntimeError("opencv-python is required. Install with: pip install opencv-python")
    except Exception as e:
        raise RuntimeError(f"Failed to read video metadata: {e}")


def ensure_evenly_spaced_indices(total_frames: int, num_frames: int) -> List[int]:
    """
    Generate evenly spaced frame indices.

    Args:
        total_frames: Total number of frames available
        num_frames: Number of frames to sample

    Returns:
        List of frame indices (0-based)
    """
    if num_frames >= total_frames:
        return list(range(total_frames))

    # Use linspace to get evenly spaced indices
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    return indices.tolist()


def sample_video_frames(
    video_path: str,
    num_frames: int = 8,
    fps: Optional[float] = None,
    start_sec: float = 0.0,
    end_sec: Optional[float] = None,
    method: str = "opencv"
) -> List[Image.Image]:
    """
    Sample frames from video file.

    Supports two sampling strategies:
    1. If fps is provided: sample at specified fps within time range
    2. Otherwise: uniformly sample num_frames across duration

    Args:
        video_path: Path to video file
        num_frames: Number of frames to sample (ignored if fps is set)
        fps: Optional sampling rate (frames per second)
        start_sec: Start time in seconds
        end_sec: Optional end time in seconds (None = end of video)
        method: Sampling method ("opencv" or "decord")

    Returns:
        List of PIL Image objects in RGB format

    Raises:
        FileNotFoundError: If video file doesn't exist
        RuntimeError: If video cannot be read or method not available
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if method == "opencv":
        return _sample_with_opencv(video_path, num_frames, fps, start_sec, end_sec)
    elif method == "decord":
        return _sample_with_decord(video_path, num_frames, fps, start_sec, end_sec)
    else:
        raise ValueError(f"Unknown sampling method: {method}. Choose 'opencv' or 'decord'.")


def _sample_with_opencv(
    video_path: str,
    num_frames: int,
    fps: Optional[float],
    start_sec: float,
    end_sec: Optional[float]
) -> List[Image.Image]:
    """Sample video frames using OpenCV."""
    try:
        import cv2
    except ImportError:
        raise RuntimeError("opencv-python is required. Install with: pip install opencv-python")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    try:
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps if video_fps > 0 else 0

        # Determine time range
        end_sec = end_sec if end_sec is not None else duration
        end_sec = min(end_sec, duration)

        if start_sec >= end_sec:
            raise ValueError(f"Invalid time range: start_sec={start_sec}, end_sec={end_sec}")

        # Convert time range to frame indices
        start_frame = int(start_sec * video_fps)
        end_frame = int(end_sec * video_fps)
        range_frames = end_frame - start_frame

        # Determine sampling strategy
        if fps is not None:
            # Sample at specified fps
            sample_interval = video_fps / fps
            num_samples = int((end_sec - start_sec) * fps)
            frame_indices = [
                start_frame + int(i * sample_interval)
                for i in range(num_samples)
            ]
            frame_indices = [idx for idx in frame_indices if idx < end_frame]
        else:
            # Uniformly sample num_frames
            relative_indices = ensure_evenly_spaced_indices(range_frames, num_frames)
            frame_indices = [start_frame + idx for idx in relative_indices]

        # Extract frames
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)
            else:
                logger.warning(f"Failed to read frame {frame_idx}, skipping")

        if not frames:
            raise RuntimeError("No frames were successfully extracted")

        logger.info(f"Sampled {len(frames)} frames from {video_path}")
        return frames

    finally:
        cap.release()


def _sample_with_decord(
    video_path: str,
    num_frames: int,
    fps: Optional[float],
    start_sec: float,
    end_sec: Optional[float]
) -> List[Image.Image]:
    """Sample video frames using Decord (faster, optional dependency)."""
    try:
        from decord import VideoReader, cpu
    except ImportError:
        raise RuntimeError(
            "decord is not installed. Install with: pip install decord\n"
            "Or use method='opencv' instead."
        )

    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        video_fps = vr.get_avg_fps()
        total_frames = len(vr)
        duration = total_frames / video_fps if video_fps > 0 else 0

        # Determine time range
        end_sec = end_sec if end_sec is not None else duration
        end_sec = min(end_sec, duration)

        if start_sec >= end_sec:
            raise ValueError(f"Invalid time range: start_sec={start_sec}, end_sec={end_sec}")

        # Convert time range to frame indices
        start_frame = int(start_sec * video_fps)
        end_frame = int(end_sec * video_fps)
        range_frames = end_frame - start_frame

        # Determine sampling strategy
        if fps is not None:
            # Sample at specified fps
            sample_interval = video_fps / fps
            num_samples = int((end_sec - start_sec) * fps)
            frame_indices = [
                start_frame + int(i * sample_interval)
                for i in range(num_samples)
            ]
            frame_indices = [idx for idx in frame_indices if idx < end_frame]
        else:
            # Uniformly sample num_frames
            relative_indices = ensure_evenly_spaced_indices(range_frames, num_frames)
            frame_indices = [start_frame + idx for idx in relative_indices]

        # Extract frames
        frame_arrays = vr.get_batch(frame_indices).asnumpy()
        frames = [Image.fromarray(arr) for arr in frame_arrays]

        logger.info(f"Sampled {len(frames)} frames from {video_path} using decord")
        return frames

    except Exception as e:
        raise RuntimeError(f"Failed to sample frames with decord: {e}")
