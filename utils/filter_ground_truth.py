"""
Ground Truth Filtering Utility for Gemma 3n Training

Filters fine_grained_labels.json to include only ground truths
of videos actually downloaded by load_dataset.py.

Usage:
    python -m utils.filter_ground_truth
"""

import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path("dataset")
GROUND_TRUTH_FILE = BASE_DIR / "fine_grained_labels.json"
MANIFEST_FILE = BASE_DIR / "manifest.json"
OUTPUT_FILE = BASE_DIR / "ground_truth.json"


def main():
    """
    Filter ground truth annotations to match downloaded videos.

    Reads the manifest of downloaded videos and filters the complete
    ground truth file to include only entries for downloaded videos.
    """
    # Validate required files exist
    if not GROUND_TRUTH_FILE.exists() or not MANIFEST_FILE.exists():
        logger.error("⚠️  Required files missing. Please run load_dataset.py first.")
        return

    # Load manifest (downloaded files)
    logger.info("📂 Loading manifest...")
    with open(MANIFEST_FILE, "r") as f:
        manifest = json.load(f)
    downloaded_filenames = {Path(p).name for p in manifest.keys()}
    logger.info(f"📊 Found {len(downloaded_filenames)} downloaded videos")

    # Load fine_grained_labels.json (complete ground truth)
    logger.info("🧠 Loading complete ground truth file...")
    with open(GROUND_TRUTH_FILE, "r") as f:
        gt_data = json.load(f)
    logger.info(f"📋 Found {len(gt_data)} ground truth entries")

    # Filter ground truths to match downloaded videos
    logger.info("🔍 Filtering ground truth entries...")
    filtered = [
        item for item in gt_data
        if "video_path" in item and Path(item["video_path"]).name in downloaded_filenames
    ]

    # Save filtered ground truth file
    with open(OUTPUT_FILE, "w") as f:
        json.dump(filtered, f, indent=2)

    logger.info(f"✅ Filtered ground truths: {len(filtered)} entries")
    logger.info(f"📝 Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
