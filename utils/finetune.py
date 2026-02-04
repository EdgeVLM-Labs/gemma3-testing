"""Fine-tuning utilities for Gemma 3n with LoRA/QLoRA."""

import logging
import os
from typing import List, Dict, Any, Optional
import torch
from transformers import TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, get_peft_model

from .model_utils import load_model_for_finetune
from .dataset_utils import create_sft_dataset
from .prompt_utils import apply_chat_template_if_available
from .config import parse_dtype

logger = logging.getLogger("gemma3n")


def build_peft_config(
    method: str = "lora",
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None
) -> LoraConfig:
    """
    Build PEFT LoRA configuration.

    Args:
        method: PEFT method ("lora" or "qlora" - both use LoraConfig)
        r: LoRA rank
        alpha: LoRA alpha (scaling factor)
        dropout: Dropout probability
        target_modules: Target modules for LoRA (None = auto-detect)

    Returns:
        LoraConfig instance
    """
    # Default target modules for transformer models
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM"
    )

    logger.info(f"PEFT config: method={method}, r={r}, alpha={alpha}, dropout={dropout}")
    logger.info(f"Target modules: {target_modules}")

    return config


class MultimodalDataCollator:
    """
    Data collator for multimodal SFT.

    Handles tokenization, image processing, and label masking for
    instruction-response pairs with optional images.
    """

    def __init__(
        self,
        processor: Any,
        max_seq_len: int = 2048,
        response_template: str = "assistant"
    ):
        """
        Initialize collator.

        Args:
            processor: Model processor/tokenizer
            max_seq_len: Maximum sequence length
            response_template: String to identify assistant responses for masking
        """
        self.processor = processor
        self.max_seq_len = max_seq_len
        self.response_template = response_template

        # Get tokenizer
        if hasattr(processor, "tokenizer"):
            self.tokenizer = processor.tokenizer
        else:
            self.tokenizer = processor

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of samples.

        Args:
            batch: List of samples from VLMSFTDataset

        Returns:
            Dictionary with input_ids, attention_mask, labels, and optionally pixel_values
        """
        # Extract data from batch
        messages_list = [sample["messages"] for sample in batch]
        images_list = [sample["images"] for sample in batch]

        # Build text prompts using chat template
        texts = []
        for messages in messages_list:
            text = apply_chat_template_if_available(
                self.processor,
                messages,
                add_generation_prompt=False  # We want the full conversation
            )
            texts.append(text)

        # Tokenize
        # Try multimodal processing if any sample has images
        has_images = any(len(imgs) > 0 for imgs in images_list)

        if has_images:
            # Flatten images for batch processing
            all_images = []
            image_counts = []
            for imgs in images_list:
                all_images.extend(imgs)
                image_counts.append(len(imgs))

            try:
                # Try processing with images
                encoding = self.processor(
                    text=texts,
                    images=all_images if all_images else None,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_len
                )
            except Exception as e:
                logger.warning(f"Failed to process with images: {e}. Using text-only.")
                encoding = self.tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_len
                )
        else:
            # Text-only
            encoding = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_seq_len
            )

        # Create labels (same as input_ids for causal LM)
        labels = encoding["input_ids"].clone()

        # Mask padding tokens with -100
        if "attention_mask" in encoding:
            labels[encoding["attention_mask"] == 0] = -100

        # Optionally mask prompt tokens (only train on responses)
        # This is a simplified version - for production, use proper response masking
        # based on the chat template structure

        encoding["labels"] = labels

        return encoding


def create_trainer(
    model_id: str,
    train_jsonl: str,
    output_dir: str,
    method: str = "qlora",
    quant: str = "4bit",
    num_train_epochs: int = 1,
    lr: float = 2e-4,
    batch_size: int = 1,
    grad_accum: int = 8,
    max_seq_len: int = 2048,
    num_frames: int = 8,
    fps: Optional[float] = None,
    logging_steps: int = 10,
    save_steps: int = 200,
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.0,
    seed: int = 42,
    peft_r: int = 16,
    peft_alpha: int = 32,
    peft_dropout: float = 0.05,
    dtype: str = "bf16"
) -> SFTTrainer:
    """
    Create SFTTrainer for fine-tuning.

    Args:
        model_id: Hugging Face model identifier
        train_jsonl: Path to training JSONL file
        output_dir: Output directory for checkpoints
        method: PEFT method ("lora" or "qlora")
        quant: Quantization mode ("none" or "4bit")
        num_train_epochs: Number of training epochs
        lr: Learning rate
        batch_size: Per-device batch size
        grad_accum: Gradient accumulation steps
        max_seq_len: Maximum sequence length
        num_frames: Number of frames to sample from videos
        fps: Optional sampling rate for videos
        logging_steps: Log every N steps
        save_steps: Save checkpoint every N steps
        warmup_ratio: Warmup ratio for learning rate schedule
        weight_decay: Weight decay coefficient
        seed: Random seed
        peft_r: LoRA rank
        peft_alpha: LoRA alpha
        peft_dropout: LoRA dropout
        dtype: Model dtype string

    Returns:
        Configured SFTTrainer instance
    """
    # Parse dtype
    torch_dtype = parse_dtype(dtype)

    # Load model and processor for fine-tuning
    logger.info("Loading model for fine-tuning...")
    model, processor = load_model_for_finetune(
        model_id=model_id,
        dtype=torch_dtype,
        quant=quant,
        gradient_checkpointing=True
    )

    # Create PEFT config and apply to model
    peft_config = build_peft_config(
        method=method,
        r=peft_r,
        alpha=peft_alpha,
        dropout=peft_dropout
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load dataset
    logger.info("Loading training dataset...")
    train_dataset = create_sft_dataset(
        jsonl_path=train_jsonl,
        num_frames=num_frames,
        fps=fps
    )

    # Create data collator
    data_collator = MultimodalDataCollator(
        processor=processor,
        max_seq_len=max_seq_len
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        bf16=torch_dtype == torch.bfloat16,
        fp16=torch_dtype == torch.float16,
        remove_unused_columns=False,
        seed=seed,
        report_to=["tensorboard"],
        dataloader_num_workers=0,  # Set to 0 to avoid multiprocessing issues with images
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=processor if hasattr(processor, "tokenizer") else processor,
    )

    logger.info("Trainer created successfully")
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Effective batch size: {batch_size * grad_accum}")
    logger.info(f"Total training steps: {len(train_dataset) // (batch_size * grad_accum) * num_train_epochs}")

    return trainer


def run_training(
    trainer: SFTTrainer,
    output_dir: str
) -> None:
    """
    Run training and save results.

    Args:
        trainer: Configured SFTTrainer
        output_dir: Output directory
    """
    logger.info("Starting training...")

    # Train
    train_result = trainer.train()

    # Save final model
    logger.info("Saving final model...")
    trainer.save_model(output_dir)

    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info(f"Training complete! Model saved to {output_dir}")
    logger.info("\nTo use the fine-tuned model for inference:")
    logger.info(f"  1. Load the base model: model = AutoModelForCausalLM.from_pretrained('{trainer.model.config._name_or_path}')")
    logger.info(f"  2. Load the adapter: from peft import PeftModel; model = PeftModel.from_pretrained(model, '{output_dir}')")
    logger.info(f"  3. Run inference as usual")
