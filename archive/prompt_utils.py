"""Prompt and chat template utilities for Gemma 3n."""

import logging
from typing import List, Dict, Any, Optional
from PIL import Image

logger = logging.getLogger("gemma3n")


def build_chat_messages(
    prompt: str,
    images: Optional[List[Image.Image]] = None,
    audio: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """
    Build chat messages in Hugging Face format.

    Constructs a message list compatible with the chat template format
    used by Gemma 3n and other multimodal models.

    Args:
        prompt: User text prompt
        images: Optional list of PIL images
        audio: Optional audio data (format varies by model)

    Returns:
        List of message dictionaries with role and content

    Example output:
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": <PIL.Image>},
                    {"type": "text", "text": "What's in this image?"}
                ]
            }
        ]
    """
    content = []

    # Add images first (convention for many VLMs)
    if images:
        for img in images:
            content.append({"type": "image", "image": img})

    # Add audio if provided (optional, format may vary)
    if audio is not None:
        content.append({"type": "audio", "audio": audio})

    # Add text prompt
    content.append({"type": "text", "text": prompt})

    messages = [
        {
            "role": "user",
            "content": content
        }
    ]

    return messages


def apply_chat_template_if_available(
    processor_or_tokenizer: Any,
    messages: List[Dict[str, Any]],
    add_generation_prompt: bool = True
) -> str:
    """
    Apply chat template to messages if available.

    Uses the model's chat template to format messages into a text prompt.
    Falls back to simple text extraction if chat template is not available.

    Args:
        processor_or_tokenizer: Processor or tokenizer instance
        messages: List of message dictionaries
        add_generation_prompt: Whether to add generation prompt suffix

    Returns:
        Formatted text prompt string

    Raises:
        ValueError: If messages format is invalid
    """
    # Try to get tokenizer from processor
    if hasattr(processor_or_tokenizer, "tokenizer"):
        tokenizer = processor_or_tokenizer.tokenizer
    else:
        tokenizer = processor_or_tokenizer

    # Check if chat template is available
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        try:
            # Use official chat template
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt
            )
            logger.debug("Applied chat template successfully")
            return text
        except Exception as e:
            logger.warning(f"Failed to apply chat template: {e}. Falling back to simple extraction.")

    # Fallback: extract text from messages
    text_parts = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", [])

        # Handle content as list
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
        # Handle content as string
        elif isinstance(content, str):
            text_parts.append(content)

    text = "\n".join(text_parts)

    if add_generation_prompt:
        text += "\n"  # Simple newline as generation prompt

    logger.debug("Using fallback text extraction (no chat template)")
    return text
