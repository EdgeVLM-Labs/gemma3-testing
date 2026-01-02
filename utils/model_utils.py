"""Model and processor loading utilities for Gemma 3n."""

import logging
from typing import Optional, Tuple, Any
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

logger = logging.getLogger("gemma3n")


def load_processor(model_id: str) -> Any:
    """
    Load processor/tokenizer for the given model.

    Args:
        model_id: Hugging Face model identifier

    Returns:
        Processor or tokenizer instance

    Raises:
        RuntimeError: If loading fails
    """
    try:
        logger.info(f"Loading processor for {model_id}...")
        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        logger.info("Processor loaded successfully")
        return processor
    except Exception as e:
        raise RuntimeError(f"Failed to load processor: {e}")


def load_model_for_inference(
    model_id: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    attn_impl: Optional[str] = None,
    quant: str = "none"
) -> Any:
    """
    Load model for inference with optional quantization.

    Args:
        model_id: Hugging Face model identifier
        device: Target device ("cuda" or "cpu")
        dtype: Model dtype (torch.bfloat16, torch.float16, torch.float32)
        attn_impl: Attention implementation ("flash_attention_2", "sdpa", or None)
        quant: Quantization mode ("none" or "4bit")

    Returns:
        Model instance in eval mode

    Raises:
        RuntimeError: If loading fails or dependencies are missing
    """
    # Validate quantization
    if quant not in ["none", "4bit"]:
        raise ValueError(f"Invalid quant mode '{quant}'. Choose 'none' or '4bit'.")

    # Check bitsandbytes availability for 4-bit quantization
    if quant == "4bit":
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise RuntimeError(
                "4-bit quantization requires bitsandbytes. "
                "Install it with: pip install bitsandbytes"
            )

    # Prepare loading arguments
    load_kwargs = {
        "pretrained_model_name_or_path": model_id,
        "trust_remote_code": True,
        "device_map": device if device == "auto" else None,
    }

    # Add quantization config if needed
    if quant == "4bit":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        logger.info("Using 4-bit quantization (NF4)")
    else:
        load_kwargs["torch_dtype"] = dtype

    # Add attention implementation if specified
    if attn_impl:
        load_kwargs["attn_implementation"] = attn_impl

    try:
        logger.info(f"Loading model {model_id}...")
        logger.info(f"  Device: {device}, dtype: {dtype}, quant: {quant}")

        model = AutoModelForCausalLM.from_pretrained(**load_kwargs)

        # Move to device if not using device_map
        if device != "auto" and quant == "none":
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                device = "cpu"
            model = model.to(device)

        model.eval()
        logger.info("Model loaded successfully and set to eval mode")
        return model

    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")


def load_model_for_finetune(
    model_id: str,
    dtype: torch.dtype = torch.bfloat16,
    quant: str = "4bit",
    gradient_checkpointing: bool = True
) -> Tuple[Any, Any]:
    """
    Load model and processor for fine-tuning with PEFT.

    Args:
        model_id: Hugging Face model identifier
        dtype: Model dtype for non-quantized parts
        quant: Quantization mode ("none" or "4bit")
        gradient_checkpointing: Enable gradient checkpointing to save memory

    Returns:
        Tuple of (model, processor)

    Raises:
        RuntimeError: If loading fails or dependencies are missing
    """
    # Load processor first
    processor = load_processor(model_id)

    # Prepare loading arguments
    load_kwargs = {
        "pretrained_model_name_or_path": model_id,
        "trust_remote_code": True,
        "device_map": "auto",
    }

    # Add quantization config if needed
    if quant == "4bit":
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise RuntimeError(
                "4-bit quantization requires bitsandbytes. "
                "Install it with: pip install bitsandbytes"
            )

        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        logger.info("Using 4-bit quantization for training (QLoRA)")
    else:
        load_kwargs["torch_dtype"] = dtype

    try:
        logger.info(f"Loading model for fine-tuning: {model_id}...")
        logger.info(f"  dtype: {dtype}, quant: {quant}, grad_ckpt: {gradient_checkpointing}")

        model = AutoModelForCausalLM.from_pretrained(**load_kwargs)

        # Enable gradient checkpointing if requested
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

        # Prepare for k-bit training if using quantization
        if quant == "4bit":
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(model)
            logger.info("Model prepared for k-bit training")

        logger.info("Model loaded successfully for fine-tuning")
        return model, processor

    except Exception as e:
        raise RuntimeError(f"Failed to load model for fine-tuning: {e}")
