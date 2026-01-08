# Gemma-3N-E2B Implementation

This folder contains the Gemma-3N-E2B model implementation adapted for video understanding tasks.

## 📁 Structure

```
gemma3/
├── train/              # Training scripts
│   ├── train.py       # Main training script
│   ├── pretrain.py    # Pretraining script
│   └── trainer.py     # Custom trainer
├── model/              # Model architecture
│   ├── arch.py        # Model architecture
│   ├── builder.py     # Model builder
│   ├── dataloader.py  # Data loading utilities
│   ├── language_model/    # Language model components
│   ├── multimodal_encoder/  # Vision encoders (VideoMamba, CLIP)
│   ├── multimodal_projector/  # Projection layers
│   └── videomamba/    # VideoMamba implementation
├── config/             # Configuration files
│   └── dataset_config.py  # Dataset configurations
├── docs/               # Documentation
│   ├── QUICKSTART.md
│   ├── finetuning_updates.md
│   └── issues.md
├── eval/               # Evaluation scripts
│   ├── mobile_videogpt.py
│   └── video_encoding.py
├── constants.py        # Constants and default values
├── conversation.py     # Conversation templates
├── mm_utils.py         # Multimodal utilities
└── utils.py           # General utilities
```

## 🚀 Usage

### Training

```bash
# Fine-tune on QVED dataset
bash scripts/finetune_qved.sh

# Or use initialize script
bash scripts/initialize_dataset.sh
```

### Model Components

- **Base Model**: google/gemma-3n-E2B
- **Video Encoder**: OpenGVLab/VideoMamba
- **Image Encoder**: openai/clip-vit-base-patch16
- **Projector**: ETP (Enhanced Token Projection)

## 📝 Notes

- This implementation is based on Mobile-VideoGPT architecture
- Adapted for Gemma-3N-E2B language model
- Supports LoRA fine-tuning for efficient training
- Uses DeepSpeed ZeRO-2 for optimization (Mamba compatibility)

## 🔗 Related

- Main training scripts: `scripts/`
- Dataset preparation: `dataset.py`
- Inference scripts: `gemma3n_batch_inference.py`, `utils/infer_qved.py`
