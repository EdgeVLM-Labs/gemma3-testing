#!/usr/bin/env python3
"""
HuggingFace Model Upload Utility

Uploads finetuned Gemma-3N-E4B models to HuggingFace Hub.

Usage:
    python utils/hf_upload.py \
    --model_path ./outputs/gemma3n-e2b-qved-ft \
    --org EdgeVLM-Labs \
    --repo_name gemma3n-e2b-qved-coach
    
    python utils/hf_upload.py --model_path outputs/gemma3n_finetune_20260108_162806_merged_16bit
    python utils/hf_upload.py --model_path outputs/gemma3n_finetune_20260108_162806_merged_16bit --repo_name my-gemma3n-finetune
    python utils/hf_upload.py --model_path outputs/gemma3n_finetune_20260108_162806 --private
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder, login

# Default organization name (will use user's account if not specified)
DEFAULT_ORG = None  # Set to None to use user's account, or specify org name


def get_default_repo_name() -> str:
    """Generate a default repository name with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"gemma3n-finetune-{timestamp}"


def check_hf_login() -> bool:
    """Check if user is logged into HuggingFace."""
    try:
        api = HfApi()
        user_info = api.whoami()
        print(f"✓ Logged in as: {user_info['name']}")
        return True
    except Exception:
        return False


def upload_model_to_hf(
    model_path: str,
    repo_name: str = None,
    org_name: str = DEFAULT_ORG,
    private: bool = False,
    commit_message: str = None,
) -> str:
    """
    Upload a finetuned model to HuggingFace Hub.

    Args:
        model_path: Path to the model directory (checkpoint or finetuned model)
        repo_name: Name for the HuggingFace repository
        org_name: HuggingFace organization name
        private: Whether to create a private repository
        commit_message: Custom commit message

    Returns:
        URL of the uploaded model on HuggingFace
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    # Detect if model contains LoRA adapters or full model
    has_adapter = (model_path / "adapter_config.json").exists()
    has_model = (model_path / "config.json").exists() or (model_path / "pytorch_model.bin").exists()

    if not has_adapter and not has_model:
        # Check for checkpoint directories
        checkpoints = list(model_path.glob("checkpoint-*"))
        if checkpoints:
            latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.name.split("-")[1]))[-1]
            print(f"Using latest checkpoint: {latest_checkpoint}")
            model_path = latest_checkpoint
            has_adapter = (model_path / "adapter_config.json").exists()
            has_model = (model_path / "config.json").exists()

    if not has_adapter and not has_model:
        raise ValueError(
            f"No model or adapter files found in {model_path}. "
            "Expected adapter_config.json or config.json / pytorch_model.bin"
        )

    # Generate repo name if not provided
    if repo_name is None:
        repo_name = get_default_repo_name()

    # Get user info to determine repo_id
    api = HfApi()
    if org_name is None:
        user_info = api.whoami()
        username = user_info['name']
        repo_id = f"{username}/{repo_name}"
    else:
        repo_id = f"{org_name}/{repo_name}"

    print(f"\n{'='*60}")
    print("HuggingFace Model Upload")
    print(f"{'='*60}")
    print(f"Model path: {model_path}")
    print(f"Repository: {repo_id}")
    print(f"Private: {private}")
    print(f"Type: {'LoRA Adapter' if has_adapter else 'Full Model'}")
    print(f"{'='*60}\n")

    # Ensure logged in
    if not check_hf_login():
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if hf_token:
            print("\nFound HF_TOKEN in environment, attempting login...")
            login(token=hf_token)
        else:
            print("⚠ Not logged in. Please run: huggingface-cli login")
            sys.exit(1)

    api = HfApi()
    # Create repository
    print(f"📦 Creating repository: {repo_id}")
    try:
        create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
        print(f"✓ Repository created/verified: {repo_id}")
    except Exception as e:
        print(f"⚠ Warning: Could not create repository: {e}")
        print("  Will try to upload anyway...")

    # Prepare commit message
    if commit_message is None:
        commit_message = f"Upload {'LoRA adapters' if has_adapter else 'finetuned model'} from {model_path.name}"

    # Upload folder
    print(f"\n🚀 Uploading model to {repo_id}...")
    try:
        upload_folder(
            folder_path=str(model_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
            ignore_patterns=["*.py", "__pycache__", "*.pyc", "runs/*", "wandb/*"],
        )
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        raise

    repo_url = f"https://huggingface.co/{repo_id}"

    print(f"\n{'='*60}")
    print("✅ Upload Complete!")
    print(f"{'='*60}")
    print(f"Repository URL: {repo_url}\n")

    # Usage instructions
    print("To use this model:")
    print(f"  from unsloth import FastVisionModel")
    print(f"  model, processor = FastVisionModel.from_pretrained('{repo_id}')")
    if has_adapter:
        print("\n  # For LoRA adapters:")
        print(f"  from peft import PeftModel")
        print(f"  base_model, processor = FastVisionModel.from_pretrained('google/gemma-3n-E2B-it')")
        print(f"  model = PeftModel.from_pretrained(base_model, '{repo_id}')")
    print(f"{'='*60}")

    return repo_url


def main():
    parser = argparse.ArgumentParser(description="Upload finetuned Gemma3N-E2B model to HuggingFace Hub")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the finetuned model directory")
    parser.add_argument("--repo_name", type=str, default=None, help="HuggingFace repository name")
    parser.add_argument("--org", type=str, default=DEFAULT_ORG, help="HuggingFace organization name")
    parser.add_argument("--private", action="store_true", help="Create a private repository")
    parser.add_argument("--commit_message", type=str, default=None, help="Custom commit message")

    args = parser.parse_args()

    try:
        repo_url = upload_model_to_hf(
            model_path=args.model_path,
            repo_name=args.repo_name,
            org_name=args.org,
            private=args.private,
            commit_message=args.commit_message,
        )
        print(f"\n🎉 Success! Model available at: {repo_url}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
