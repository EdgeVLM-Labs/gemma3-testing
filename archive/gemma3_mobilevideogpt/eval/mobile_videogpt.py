from typing import List, Optional, Tuple, Union
import torch
from tqdm import tqdm
from PIL import Image
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

# Gemma-3N imports
from unsloth import FastModel
from transformers import TextStreamer

# Helper inference function
def gemma_3n_inference(model, tokenizer, messages, max_new_tokens=128):
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.1,
        do_sample=True,
        streamer=TextStreamer(tokenizer, skip_prompt=True),
    )

    # Only decode new tokens
    input_len = inputs.input_ids.shape[1]
    new_tokens = outputs[0][input_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response

# Simple frame extraction (can replace with more advanced)
def extract_frames(video_path, num_frames=8):
    import cv2
    from PIL import Image
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // num_frames, 1)
    frames = []
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret: break
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames

@register_model("gemma_3n")
class Gemma3NModel(lmms):
    """
    Gemma-3N E2B Model for multimodal video/text inference
    """

    def __init__(
        self,
        pretrained: str = "google/gemma-3n-E2B",
        batch_size: int = 1,
        max_seq_length: int = 1024,
        max_frames: int = 8,
        use_cache=True,
        **kwargs,
    ):
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.max_frames = max_frames
        self.use_cache = use_cache

        # Load Gemma-3N model
        self.model, self.tokenizer = FastModel.from_pretrained(
            model_name=pretrained,
            dtype=None,
            max_seq_length=max_seq_length,
            load_in_4bit=False,
            full_finetuning=False,
        )

        self.model = self.model.to("cuda")

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def rank(self):
        return 0

    @property
    def world_size(self):
        return 1

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood not implemented for Gemma-3N")

    def generate_until(self, requests: List[Instance], max_new_tokens: int = 128) -> List[str]:
        responses = []
        pbar = tqdm(total=len(requests), desc="Gemma-3N Responding")
        for inst in requests:
            # Expect instance to have `args` as [video_path, prompt_text]
            video_path, prompt_text = inst.args
            frames = extract_frames(video_path, num_frames=self.max_frames)

            messages = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": img} for img in frames],
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]

            try:
                response = gemma_3n_inference(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    messages=messages,
                    max_new_tokens=max_new_tokens,
                )
            except Exception as e:
                response = f"[Error: {e}]"
            responses.append(response)
            pbar.update(1)
        pbar.close()
        return responses

    def generate_until_multi_round(
        self,
        prompts: List[str],
        max_new_tokens: int,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
        outputs = []
        for p in prompts:
            messages = [{"role": "user", "content": [{"type": "text", "text": p}]}]
            response = gemma_3n_inference(self.model, self.tokenizer, messages, max_new_tokens=max_new_tokens)
            outputs.append(response)
        return outputs
