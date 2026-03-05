# Archive

Deprecated code from earlier development phases. Kept for reference only — **not part of the active pipeline**.

## What's here

### MobileVideoGPT (gemma3_mobilevideogpt/)
Original architecture using VideoMamba + CLIP encoder + Qwen language model. Required mamba-ssm, einops, decord, timm, and DeepSpeed for 3-stage training. Replaced by direct Gemma-3N fine-tuning.

### Old Unsloth scripts
- `gemma3_finetune_unsloth.py` / `finetune_gemma3n_unsloth.sh` — Early unsloth fine-tuning attempts
- `run_inference_unsloth.py` / `run_inference_unsloth.sh` — Early unsloth inference scripts
- `test_inference.py` — Old inference using archived gemma3 package imports

### Old utility scripts
- `config.py`, `model_utils.py`, `prompt_utils.py`, `video_utils.py`, `logging_utils.py`, `dataset_utils.py` — Helper modules from MobileVideoGPT era
- `convert_dataset.py` — Dataset format converter
- `run_finetune.py`, `finetune.py`, `inference.py` — Earlier training/inference entry points
- `quick_setup.sh`, `verify_setup.sh`, `plot_training.sh` — Old shell utilities

### Old documentation
- `FINETUNE_GUIDE.md`, `GEMMA3N_E2B_FINETUNE_GUIDE.md`, `INFERENCE_UPDATE.md`, `finetuning_updates.md` — Outdated guides
- `basic-inference.ipynb` — Old notebook
- `Gemma-3B.sh` — Original 3-stage MobileVideoGPT training script
