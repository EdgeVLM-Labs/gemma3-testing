#!/usr/bin/env python3
"""
HuggingFace Model Upload Utility for Gemma 3n

Uploads fine-tuned Gemma 3n models (with LoRA/QLoRA adapters) to HuggingFace Hub.

Usage:
    python -m utils.hf_upload --model_path outputs/gemma3n-finetuned
    python -m utils.hf_upload --model_path outputs/gemma3n-finetuned --repo_name gemma3n-qved-20260102
    python -m utils.hf_upload --model_path outputs/gemma3n-finetuned --private
"""

import os
import sys
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder, login

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Default organization name
DEFAULT_ORG = "EdgeVLM-Labs"


def get_default_repo_name() -> str:
    """Generate a default repository name with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"gemma3n-qved-finetune-{timestamp}"


def check_hf_login() -> bool:
    """Check if user is logged into HuggingFace."""
    try:
        api = HfApi()
        user_info = api.whoami()
        logger.info(f"✓ Logged in as: {user_info['name']}")
        return True
    except Exception:
        return False


def create_model_card(model_path: Path, repo_id: str, base_model: str, has_adapter: bool) -> str:
    """Create a comprehensive model card for Gemma 3n fine-tuned model."""

    # Try to load hyperparameters from saved config
    hyperparams = {}
    config_file = model_path / "hyperparameters.json"
    if not config_file.exists():
        config_file = model_path.parent / "hyperparameters.json"

    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                hyperparams = json.load(f)
        except Exception:
            pass

    # Try to load dataset info
    dataset_info = {}
    try:
        train_file = Path("dataset/gemma_train.jsonl")
        val_file = Path("dataset/gemma_val.jsonl")
        test_file = Path("dataset/gemma_test.jsonl")

        if train_file.exists() and val_file.exists() and test_file.exists():
            with open(train_file, 'r') as f:
                train_count = sum(1 for line in f if line.strip())
            with open(val_file, 'r') as f:
                val_count = sum(1 for line in f if line.strip())
            with open(test_file, 'r') as f:
                test_count = sum(1 for line in f if line.strip())

            dataset_info = {
                "train": train_count,
                "val": val_count,
                "test": test_count,
                "total": train_count + val_count + test_count
            }
    except Exception:
        pass

    # Build model card
    model_card = f"""---
tags:
- video-text-to-text
- gemma-3n
- qved
- physiotherapy
- exercise-assessment
library_name: transformers
license: apache-2.0
base_model: {base_model}
---

# Gemma 3n QVED Fine-tuned Model

This model is a fine-tuned version of [{base_model}](https://huggingface.co/{base_model}) on the QVED (Qualitative Video-based Exercise Dataset) for physiotherapy exercise assessment.

## Model Description

- **Base Model:** {base_model}
- **Architecture:** Gemma 3n Multimodal with {"LoRA adapters" if has_adapter else "full fine-tuning"}
- **Task:** Video-based exercise quality assessment and feedback generation
- **Dataset:** QVED (Physiotherapy Exercise Videos)
- **Framework:** Hugging Face Transformers + TRL + PEFT

## Training Details

### Hyperparameters

"""

    if hyperparams:
        model_card += f"""- **Epochs:** {hyperparams.get('epochs', 'N/A')}
- **Learning Rate:** {hyperparams.get('lr', hyperparams.get('learning_rate', 'N/A'))}
- **LoRA Rank:** {hyperparams.get('peft_r', hyperparams.get('lora_r', 'N/A'))}
- **LoRA Alpha:** {hyperparams.get('peft_alpha', hyperparams.get('lora_alpha', 'N/A'))}
- **Batch Size:** {hyperparams.get('batch_size', 'N/A')}
- **Gradient Accumulation Steps:** {hyperparams.get('grad_accum', hyperparams.get('gradient_accumulation_steps', 'N/A'))}
- **Max Sequence Length:** {hyperparams.get('max_seq_len', hyperparams.get('max_length', 'N/A'))}
- **Weight Decay:** {hyperparams.get('weight_decay', '0.0')}
- **Warmup Ratio:** {hyperparams.get('warmup_ratio', '0.03')}
- **LR Scheduler:** Cosine
- **Precision:** {hyperparams.get('dtype', 'bfloat16')}

"""
    else:
        model_card += "See training configuration for details.\n\n"

    model_card += """### Training Infrastructure

- **Framework:** TRL (Transformers Reinforcement Learning)
- **Mixed Precision:** bfloat16
- **Optimization:** {"LoRA (Low-Rank Adaptation)" if has_adapter else "Full Fine-tuning"}
- **Video Processing:** Frame-based sampling (uniformly sampled frames)

"""

    if dataset_info:
        model_card += f"""### Dataset Splits

- **Train:** {dataset_info['train']} samples
- **Validation:** {dataset_info['val']} samples
- **Test:** {dataset_info['test']} samples
- **Total:** {dataset_info['total']} samples

"""

    model_card += f"""### Training Configuration

- **Model:** Gemma 3n multimodal
- **Video Frames:** 8 frames sampled uniformly per video
- **Quantization:** {"4-bit (QLoRA)" if "4bit" in str(hyperparams.get('quant', '')) else "None"}

## Usage

### Loading the Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
{"from peft import PeftModel" if has_adapter else ""}
import torch

# Load base model and processor
base_model_id = "{base_model}"
processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

{"# Load LoRA adapter" if has_adapter else "# Model is already fine-tuned"}
{"model = PeftModel.from_pretrained(model, '" + repo_id + "')" if has_adapter else ""}
```

### Running Inference

```python
from utils.video_utils import sample_video_frames
from utils.prompt_utils import build_chat_messages
from PIL import Image

# Load and sample video frames
video_path = "path/to/exercise_video.mp4"
frames = sample_video_frames(video_path, num_frames=8)

# Build prompt
prompt = "Please evaluate the exercise form shown. What mistakes, if any, are present?"
messages = build_chat_messages(prompt, images=frames)

# Process and generate
inputs = processor(
    text=processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
    images=frames,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
response = processor.decode(outputs[0], skip_special_tokens=True)

print(response)
```

## Evaluation

The model is evaluated on:

- **Exercise Form Assessment:** Identifying incorrect form and technique
- **Feedback Generation:** Providing actionable corrections
- **Exercise Classification:** Recognizing exercise types from video

## Intended Use

This model is designed for:

- Automated physiotherapy exercise assessment
- Generating feedback on exercise form and technique
- Educational and research purposes in healthcare AI
- Assistive technology for remote physiotherapy

## Limitations

- Trained on a limited dataset ({dataset_info.get('total', 'specific')} samples)
- Performance may vary on exercises not seen during training
- Should not replace professional medical advice
- Video quality and camera angle significantly affect performance
- Limited to exercises present in the QVED dataset

## Training Procedure

This model was fine-tuned using:

1. **Dataset Preparation:** QVED videos with quality annotations
2. **Frame Sampling:** Uniform sampling of 8 frames per video
3. **{"LoRA Fine-tuning" if has_adapter else "Full Fine-tuning"}:** Efficient parameter updates
4. **Validation:** Continuous evaluation during training
5. **Metrics Tracking:** Loss, gradient norm, and learning rate monitoring

## Citation

If you use this model, please cite:

```bibtex
@misc{{gemma3n-qved-finetune,
  author = {{EdgeVLM Labs}},
  title = {{Gemma 3n QVED Fine-tuned Model}},
  year = {{2026}},
  publisher = {{HuggingFace}},
  url = {{https://huggingface.co/{repo_id}}}
}}
```

## Model Card Authors

EdgeVLM Labs

## Model Card Contact

For questions or feedback, please open an issue in the model repository.
"""

    return model_card


def upload_model_to_hf(
    model_path: str,
    repo_name: str = None,
    org_name: str = DEFAULT_ORG,
    base_model: str = "google/gemma-3n-E2B",
    private: bool = False,
    commit_message: str = None,
) -> str:
    """
    Upload a fine-tuned Gemma 3n model to HuggingFace Hub.

    Args:
        model_path: Path to the model directory
        repo_name: Name for the HuggingFace repository
        org_name: HuggingFace organization name
        base_model: Base model identifier
        private: Whether to create a private repository
        commit_message: Custom commit message

    Returns:
        URL of the uploaded model on HuggingFace
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    # Check for adapter files or model files
    has_adapter = (model_path / "adapter_config.json").exists()
    has_model = (model_path / "config.json").exists() or (model_path / "pytorch_model.bin").exists()

    if not has_adapter and not has_model:
        # Maybe it's a checkpoint directory
        checkpoints = list(model_path.glob("checkpoint-*"))
        if checkpoints:
            # Use the latest checkpoint
            latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.name.split("-")[1]))[-1]
            logger.info(f"Using latest checkpoint: {latest_checkpoint}")
            model_path = latest_checkpoint
            has_adapter = (model_path / "adapter_config.json").exists()
            has_model = (model_path / "config.json").exists()

    if not has_adapter and not has_model:
        raise ValueError(
            f"No model or adapter files found in {model_path}. "
            "Expected adapter_config.json or config.json"
        )

    # Generate repo name if not provided
    if repo_name is None:
        repo_name = get_default_repo_name()

    # Full repository ID
    repo_id = f"{org_name}/{repo_name}"

    logger.info(f"\n{'='*60}")
    logger.info("HuggingFace Model Upload - Gemma 3n")
    logger.info(f"{'='*60}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Repository: {repo_id}")
    logger.info(f"Private: {private}")
    logger.info(f"Type: {'LoRA Adapter' if has_adapter else 'Full Model'}")
    logger.info(f"{'='*60}\n")

    # Check login status
    if not check_hf_login():
        logger.warning("⚠ Not logged into HuggingFace. Please login first:")
        logger.info("  huggingface-cli login")
        logger.info("  or set HF_TOKEN environment variable")

        # Try to login with token from environment
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if hf_token:
            logger.info("\nFound HF_TOKEN in environment, attempting login...")
            login(token=hf_token)
        else:
            sys.exit(1)

    api = HfApi()

    # Create repository
    logger.info(f"📦 Creating repository: {repo_id}")
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True,
        )
        logger.info(f"✓ Repository created/verified: {repo_id}")
    except Exception as e:
        logger.warning(f"⚠ Warning: Could not create repository: {e}")
        logger.info("  Will try to upload anyway...")

    # Generate and save model card
    logger.info(f"\n📝 Generating model card...")
    model_card_content = create_model_card(model_path, repo_id, base_model, has_adapter)
    readme_path = model_path / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(model_card_content)
    logger.info(f"✓ Model card saved to {readme_path}")

    # Prepare commit message
    if commit_message is None:
        if has_adapter:
            commit_message = f"Upload LoRA adapters from {model_path.name}"
        else:
            commit_message = f"Upload fine-tuned model from {model_path.name}"

    # Upload model
    logger.info(f"\n🚀 Uploading model to {repo_id}...")
    logger.info("  This may take a few minutes depending on model size...")

    try:
        upload_folder(
            folder_path=str(model_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
            ignore_patterns=["*.py", "__pycache__", "*.pyc", "runs/*", "wandb/*", "*.log"],
        )
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        raise

    # Get repository URL
    repo_url = f"https://huggingface.co/{repo_id}"

    logger.info(f"\n{'='*60}")
    logger.info("✅ Upload Complete!")
    logger.info(f"{'='*60}")
    logger.info(f"Repository URL: {repo_url}")
    logger.info(f"\nTo use this model:")
    logger.info(f"  from transformers import AutoModelForCausalLM, AutoProcessor")
    logger.info(f"  processor = AutoProcessor.from_pretrained('{base_model}', trust_remote_code=True)")
    logger.info(f"  model = AutoModelForCausalLM.from_pretrained('{base_model}', trust_remote_code=True)")
    if has_adapter:
        logger.info(f"\n  # Load LoRA adapter:")
        logger.info(f"  from peft import PeftModel")
        logger.info(f"  model = PeftModel.from_pretrained(model, '{repo_id}')")
    logger.info(f"{'='*60}")

    return repo_url


def main():
    parser = argparse.ArgumentParser(
        description="Upload fine-tuned Gemma 3n model to HuggingFace Hub"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned model directory",
    )
    parser.add_argument(
        "--repo_name",
        type=str,
        default=None,
        help=f"Name for the HuggingFace repository (default: gemma3n-qved-finetune-TIMESTAMP)",
    )
    parser.add_argument(
        "--org",
        type=str,
        default=DEFAULT_ORG,
        help=f"HuggingFace organization name (default: {DEFAULT_ORG})",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="google/gemma-3n-E2B",
        help="Base model identifier (default: google/gemma-3n-E2B)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository",
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default=None,
        help="Custom commit message for the upload",
    )

    args = parser.parse_args()

    try:
        repo_url = upload_model_to_hf(
            model_path=args.model_path,
            repo_name=args.repo_name,
            org_name=args.org,
            base_model=args.base_model,
            private=args.private,
            commit_message=args.commit_message,
        )
        logger.info(f"\n🎉 Success! Model available at: {repo_url}")
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
