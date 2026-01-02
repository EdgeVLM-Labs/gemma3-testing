"""
Dataset Download Utility for Gemma 3n Training

Features:
- Downloads videos from HuggingFace dataset repository
- Randomly samples N videos per exercise class
- Preserves folder structure
- Downloads fine_grained_labels.json (ground truth annotations)
- Creates manifest.json of downloaded videos

Usage:
    python -m utils.load_dataset
    python -m utils.load_dataset 10  # Download 10 videos per class
"""

import os
import random
import json
import sys
from pathlib import Path
from huggingface_hub import list_repo_files, hf_hub_download
import shutil
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Dataset configuration
REPO_ID = "EdgeVLM-Labs/QVED-Test-Dataset"
LOCAL_DIR = Path("dataset")
MAX_PER_CLASS = 5  # Default: 5 videos per exercise class
FILE_EXT = ".mp4"
GROUND_TRUTH_FILE = "fine_grained_labels.json"
RANDOM_SEED = 42


def collect_videos(repo_id: str):
    """
    Collect all video files grouped by exercise class (subfolder).

    Args:
        repo_id: HuggingFace dataset repository ID

    Returns:
        Tuple of (videos_by_class dict, all_files list)
    """
    logger.info(f"📂 Listing repo files from: {repo_id}")
    all_files = list_repo_files(repo_id, repo_type="dataset")

    by_class = {}
    for f in all_files:
        if not f.endswith(FILE_EXT):
            continue
        parts = f.split("/")
        if len(parts) < 2:
            continue
        cls = parts[0]
        by_class.setdefault(cls, []).append(f)

    logger.info(f"✅ Found {len(by_class)} classes with video files")
    return by_class, all_files


def sample_and_download(by_class: dict, repo_id: str, local_dir: Path, max_per_class: int):
    """
    Sample random videos per class and download them.

    Downloads into structure: <local_dir>/<class>/<filename>

    Args:
        by_class: Dictionary mapping class names to list of video paths
        repo_id: HuggingFace dataset repository ID
        local_dir: Local directory for downloads
        max_per_class: Maximum videos to download per class

    Returns:
        Dictionary mapping downloaded video paths to class names (manifest)
    """
    random.seed(RANDOM_SEED)
    manifest = {}
    total_downloaded = 0

    for cls, vids in by_class.items():
        class_dir = local_dir / cls
        class_dir.mkdir(parents=True, exist_ok=True)

        # Sample videos
        sample = random.sample(vids, min(len(vids), max_per_class))
        logger.info(f"🎥 {cls}: {len(sample)} sampled of {len(vids)} available")

        for rel_path in sample:
            filename = os.path.basename(rel_path)
            target_path = class_dir / filename

            # Retry loop for rate limiting
            while True:
                try:
                    cached_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=rel_path,
                        repo_type="dataset",
                    )

                    shutil.copy2(cached_path, target_path)
                    manifest[str(target_path)] = cls
                    total_downloaded += 1
                    break

                except Exception as e:
                    if "429" in str(e) or "Too Many Requests" in str(e):
                        logger.warning(f"⚠️  Rate limit hit (429). Waiting ~3 minutes before retrying {rel_path}...")
                        time.sleep(200)
                    else:
                        logger.error(f"⚠️  Failed to download {rel_path}: {e}")
                        break

    logger.info(f"\n✅ Download complete: {total_downloaded} videos")
    return manifest


def download_ground_truth(repo_id: str, local_dir: Path, all_files: list):
    """
    Download fine_grained_labels.json if present in repository.

    Args:
        repo_id: HuggingFace dataset repository ID
        local_dir: Local directory for downloads
        all_files: List of all files in repository

    Returns:
        Path to downloaded ground truth file, or None if not found
    """
    candidates = [f for f in all_files if f.endswith(GROUND_TRUTH_FILE)]
    if not candidates:
        logger.warning(f"⚠️  No {GROUND_TRUTH_FILE} found in repo")
        return None

    gt_path = local_dir / GROUND_TRUTH_FILE
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=candidates[0],
            local_dir=str(local_dir),
            repo_type="dataset",
        )
        logger.info(f"🧠 Ground truth file downloaded to: {gt_path}")
        return gt_path
    except Exception as e:
        logger.error(f"⚠️  Failed to download {GROUND_TRUTH_FILE}: {e}")
        return None


def save_manifest(manifest: dict, local_dir: Path):
    """
    Save manifest.json mapping downloaded videos to their classes.

    Args:
        manifest: Dictionary mapping video paths to class names
        local_dir: Local directory for saving manifest

    Returns:
        Path to saved manifest file
    """
    manifest_path = local_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"📝 Manifest saved to: {manifest_path}")
    return manifest_path


def main():
    """Main execution function."""
    # Parse command line arguments
    max_per_class = MAX_PER_CLASS
    if len(sys.argv) > 1:
        try:
            max_per_class = int(sys.argv[1])
            logger.info(f"📊 Using MAX_PER_CLASS = {max_per_class} (from command line)")
        except ValueError:
            logger.warning(f"⚠️  Invalid argument. Using default MAX_PER_CLASS = {MAX_PER_CLASS}")

    # Create local directory
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    # Download dataset
    by_class, all_files = collect_videos(REPO_ID)
    manifest = sample_and_download(by_class, REPO_ID, LOCAL_DIR, max_per_class)
    save_manifest(manifest, LOCAL_DIR)
    download_ground_truth(REPO_ID, LOCAL_DIR, all_files)

    logger.info("🏁 Dataset download completed")


if __name__ == "__main__":
    main()
