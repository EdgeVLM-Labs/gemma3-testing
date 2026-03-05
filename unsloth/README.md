# Unsloth Pipeline (Alternative)

Alternative fine-tuning and inference pipeline using [Unsloth](https://github.com/unslothai/unsloth) `FastVisionModel`. This is **not** the active pipeline — see `core/` for the main transformers-based workflow.

## When to use this

- If you want faster fine-tuning with Unsloth's optimized kernels
- If you need merged 16-bit model export (Unsloth handles LoRA merging natively)

## Requirements

```bash
pip install unsloth unsloth_zoo
```

## Files

| File | Description |
|------|-------------|
| `infer_qved.py` | Single-video inference using FastVisionModel |
| `eval_gemma3n.py` | Batch evaluation on QVED val/test sets |
| `test_inference_unsloth.py` | Batch test inference with prediction JSON output |
| `run_inference.sh` | Shell wrapper: runs test inference + evaluation report |
| `augment_videos.py` | Video augmentation using vidaug + optional inference |

## Usage

```bash
# Single video inference
python unsloth/infer_qved.py \
  --model_path google/gemma-3n-E2B-it \
  --video_path sample_videos/00000340.mp4

# Batch test inference + evaluation
bash unsloth/run_inference.sh \
  --model_path outputs/gemma3n_finetune_merged_16bit \
  --test_json dataset/qved_test.json \
  --data_path dataset
```
