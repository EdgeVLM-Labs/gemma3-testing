import os
import io
import cv2
import torch
import imageio
import numpy as np
from PIL import Image
from decord import VideoReader

# ----------------------------
# Utility: uniform sampling
# ----------------------------
def uniform_sample(lst, n):
    """Uniformly sample n items from a list."""
    assert n <= len(lst), f"Cannot sample {n} from list of length {len(lst)}"
    if n == len(lst):
        return lst
    m = len(lst)
    step = m / n
    return [lst[int(i * step)] for i in range(n)]

# ----------------------------
# Decode raw video using decord
# ----------------------------
def _get_rawvideo_dec(video_path, image_processor, video_processor,
                      max_frames=16, min_frames=16,
                      image_resolution=224, video_framerate=1,
                      s=None, e=None, num_video_frames=16, num_context_images=16):
    """Read video via Decord and preprocess frames for Gemma-3N."""
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)
    
    vreader = VideoReader(video_path, num_threads=1)
    fps = vreader.get_avg_fps()
    f_start = 0 if s is None else max(int(s * fps), 0)
    f_end = len(vreader) - 1 if e is None else min(int(e * fps), len(vreader) - 1)

    t_stride = max(int(round(fps / video_framerate)), 1)
    all_pos = list(range(f_start, f_end + 1, t_stride))
    
    # Adjust number of frames
    if len(all_pos) > max_frames:
        sample_pos = [all_pos[i] for i in np.linspace(0, len(all_pos) - 1, max_frames, dtype=int)]
    elif len(all_pos) < min_frames:
        stride = max(1, (f_end - f_start) // (min_frames - 1))
        sample_pos = list(range(f_start, f_start + stride * min_frames, stride))
    else:
        sample_pos = all_pos

    frames = [Image.fromarray(frame) for frame in vreader.get_batch(sample_pos).asnumpy()]

    # Sample patch frames & context frames
    patch_images = uniform_sample(frames, min(num_video_frames, len(frames)))
    context_images = uniform_sample(frames, min(num_context_images, len(frames)))

    # Preprocess
    patch_images = video_processor.preprocess(patch_images)['pixel_values']
    context_images = [image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0] for img in context_images]

    # Pad context_images if needed
    while len(context_images) < num_context_images:
        context_images.append(torch.zeros_like(context_images[0]))

    # Pad patch_images if needed
    while len(patch_images) < num_video_frames:
        patch_images.append(torch.zeros((3, image_resolution, image_resolution)))

    return patch_images, context_images, len(patch_images)

# ----------------------------
# Read GIF
# ----------------------------
def read_gif_mod(video_path, image_processor, max_frames=16, image_resolution=224, video_framerate=25, s=None, e=None, sample_fps=1):
    """Process GIF into model-ready frames."""
    gif_reader = imageio.get_reader(video_path)
    num_frames = len(gif_reader)
    f_start = 0 if s is None else max(int(s * video_framerate), 0)
    f_end = num_frames - 1 if e is None else min(int(e * video_framerate), num_frames - 1)

    t_stride = max(int(round(video_framerate / sample_fps)), 1)
    frame_indices = range(f_start, f_end + 1, t_stride)

    frames = []
    for i, frame in enumerate(gif_reader):
        if i in frame_indices:
            img = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            frames.append(Image.fromarray(img).resize((image_resolution, image_resolution)))
        if len(frames) >= max_frames:
            break

    patch_images = image_processor.preprocess(frames)['pixel_values']
    return patch_images, len(frames)

# ----------------------------
# Read frames from folder
# ----------------------------
def read_frame_mod(video_path, image_processor, video_processor,
                   max_frames=16, image_resolution=224, video_framerate=3,
                   s=None, e=None, sample_fps=1, num_video_frames=16, num_context_images=16):
    """Read frames from a folder of images."""
    frame_files = sorted(os.listdir(video_path))
    frames = [Image.open(os.path.join(video_path, f)) for f in frame_files]
    num_frames = len(frames)

    f_start = 0 if s is None else max(int(s * video_framerate), 0)
    f_end = num_frames - 1 if e is None else min(int(e * video_framerate), num_frames - 1)
    t_stride = max(int(round(video_framerate / sample_fps)), 1)
    frames = [frames[i] for i in range(f_start, f_end + 1, t_stride)][:max_frames]

    patch_images = uniform_sample(frames, min(num_video_frames, len(frames)))
    context_images = uniform_sample(frames, min(num_context_images, len(frames)))

    patch_images = video_processor.preprocess(patch_images)['pixel_values']
    context_images = [image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0] for img in context_images]

    # Padding
    while len(context_images) < num_context_images:
        context_images.append(torch.zeros_like(context_images[0]))
    while len(patch_images) < num_video_frames:
        patch_images.append(torch.zeros((3, image_resolution, image_resolution)))

    return patch_images, context_images, len(patch_images)

# ----------------------------
# Read PIL frames
# ----------------------------
def read_pil_frames(pil_frames, image_processor, video_processor,
                    max_frames=16, image_resolution=224, video_framerate=3,
                    s=None, e=None, sample_fps=1, num_video_frames=16, num_context_images=16):
    """Read a list of PIL frames and preprocess."""
    num_frames = len(pil_frames)
    f_start = 0 if s is None else max(int(s * video_framerate), 0)
    f_end = num_frames - 1 if e is None else min(int(e * video_framerate), num_frames - 1)
    t_stride = max(int(round(video_framerate / sample_fps)), 1)
    frames = [pil_frames[i] for i in range(f_start, f_end + 1, t_stride)][:max_frames]

    patch_images = uniform_sample(frames, min(num_video_frames, len(frames)))
    context_images = uniform_sample(frames, min(num_context_images, len(frames)))

    patch_images = video_processor.preprocess(patch_images)['pixel_values']
    context_images = [image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0] for img in context_images]

    # Padding
    while len(context_images) < num_context_images:
        context_images.append(torch.zeros_like(context_images[0]))
    while len(patch_images) < num_video_frames:
        patch_images.append(torch.zeros((3, image_resolution, image_resolution)))

    return patch_images, context_images, len(patch_images)
