# Gemma-3N-E2B Architecture Overview

## Model Components

### 1. Language Model
- **Base**: google/gemma-3n-E2B
- **Type**: Causal Language Model with video understanding capabilities
- **Location**: `gemma3/model/language_model/`

### 2. Vision Encoders

#### Video Encoder
- **Model**: OpenGVLab/VideoMamba
- **Purpose**: Extract temporal features from video frames
- **Location**: `gemma3/model/videomamba/`
- **Key Features**:
  - State-space model for efficient video processing
  - Temporal modeling with selective scan
  - Handles variable-length video sequences

#### Image Encoder
- **Model**: openai/clip-vit-base-patch16
- **Purpose**: Extract visual features from individual frames
- **Location**: `gemma3/model/multimodal_encoder/`
- **Key Features**:
  - Pre-trained CLIP vision transformer
  - Fixed image resolution processing
  - Rich semantic visual representations

### 3. Multimodal Projectors
- **Type**: ETP (Enhanced Token Projection)
- **Location**: `gemma3/model/multimodal_projector/`
- **Purpose**: Bridge vision and language modalities
- **Components**:
  - Video projector: Maps VideoMamba features to LLM space
  - Image projector: Maps CLIP features to LLM space
  - Learnable projection layers with LayerNorm

## Training Pipeline

### Stage 1: Video Projection Pretraining
```bash
deepspeed gemma3/train/pretrain.py \
  --tune_mm_mlp_adapter True \
  --vision_tower OpenGVLab/VideoMamba
```

**Purpose**: Train video projector to align VideoMamba features with Gemma-3N

### Stage 2: Image Projection Pretraining
```bash
deepspeed gemma3/train/pretrain.py \
  --tune_image_mm_mlp_adapter True \
  --image_vision_tower openai/clip-vit-base-patch16
```

**Purpose**: Train image projector to align CLIP features with Gemma-3N

### Stage 3: End-to-End Fine-tuning
```bash
deepspeed gemma3/train/train.py \
  --lora_enable True \
  --pretrain_mm_mlp_adapter <video_projector> \
  --pretrain_image_mm_mlp_adapter <image_projector>
```

**Purpose**: Fine-tune entire model on task-specific data (QVED)

## Configuration

### Dataset Configuration
- **Location**: `gemma3/config/dataset_config.py`
- **Supported Datasets**:
  - QVED_TRAIN: Training split
  - QVED_VAL: Validation split
  - QVED_TEST: Test split
  - PRETRAINING: Generic pretraining data

### Model Arguments
- **Version**: `gemma3n_v1` or `gemma3n_E2B`
- **Projector Type**: `etp`
- **Vision Select Layer**: -2 (second-to-last layer)
- **Max Length**: 2048 tokens

## Training Configuration

### LoRA Settings (Efficient Fine-tuning)
- **LoRA Rank (r)**: 64
- **LoRA Alpha**: 128
- **Target Modules**: Query, key, value projections
- **Trainable Parameters**: ~2% of total

### Optimization
- **Framework**: DeepSpeed ZeRO-2
- **Precision**: BF16 + TF32
- **Gradient Checkpointing**: Enabled
- **Batch Size**: 8 per GPU
- **Gradient Accumulation**: 8 steps
- **Effective Batch Size**: 64

### Learning Rates
- **LLM**: 2e-4
- **Projectors**: 1e-4
- **Scheduler**: Cosine with warmup (5%)

## Data Processing

### Video Processing
- **Frame Sampling**: 8 frames per video
- **Frame Selection**: Uniform temporal sampling
- **Resolution**: Adaptive (maintains aspect ratio)
- **Preprocessing**: Normalization, resize, padding

### Conversation Format
```json
{
  "video": "path/to/video.mp4",
  "conversations": [
    {"from": "human", "value": "<prompt>"},
    {"from": "gpt", "value": "<response>"}
  ]
}
```

## Inference

### Frame Extraction
```python
from gemma3.model.dataloader import _get_rawvideo_dec

video_frames = _get_rawvideo_dec(
    video_path, 
    num_frames=8,
    sample='uniform'
)
```

### Model Loading
```python
from gemma3.model import GemmaForCausalLM

model = GemmaForCausalLM.from_pretrained(
    "google/gemma-3n-E2B",
    vision_tower="OpenGVLab/VideoMamba",
    image_vision_tower="openai/clip-vit-base-patch16"
)
```

## Evaluation Metrics

### Supported Metrics
- **ROUGE-L**: Longest common subsequence overlap
- **METEOR**: Semantic similarity with synonyms
- **BERT Score**: Contextual embedding similarity
- **Custom**: Task-specific exercise form accuracy

## File Structure Reference

```
gemma3/
├── train/
│   ├── train.py       # Main training loop
│   ├── pretrain.py    # Projector pretraining
│   └── trainer.py     # Custom HuggingFace Trainer
├── model/
│   ├── arch.py        # Model architecture definitions
│   ├── builder.py     # Model initialization
│   ├── dataloader.py  # Video data loading
│   ├── language_model/
│   │   └── qwen.py    # Language model wrappers
│   ├── multimodal_encoder/
│   │   ├── builder.py
│   │   ├── clip_encoder.py
│   │   └── processor.py
│   ├── multimodal_projector/
│   │   └── builder.py
│   └── videomamba/
│       ├── videomamba.py
│       └── config.py
├── config/
│   └── dataset_config.py
├── constants.py       # Default values, special tokens
├── conversation.py    # Conversation templates
├── mm_utils.py       # Multimodal utilities
└── utils.py          # General utilities
```

## Best Practices

### Memory Optimization
1. Use gradient checkpointing
2. Enable ZeRO-2 optimization
3. Use BF16 precision
4. Reduce batch size if OOM
5. Clear cache between runs

### Training Tips
1. Start with pretrained projectors
2. Use LoRA for efficient fine-tuning
3. Monitor validation loss
4. Save checkpoints regularly
5. Use WandB for experiment tracking

### Inference Optimization
1. Use Unsloth FastModel for faster inference
2. Batch multiple videos together
3. Cache model weights
4. Use appropriate max_new_tokens
5. Enable streaming for real-time output

## Troubleshooting

### Common Issues

**Q: CUDA Out of Memory**
- Reduce batch size
- Enable gradient checkpointing
- Use ZeRO-3 instead of ZeRO-2
- Reduce max_length

**Q: Mamba compatibility issues**
- Use ZeRO-2 (not ZeRO-3)
- Ensure mamba-ssm==1.2.0
- Install torch before mamba-ssm

**Q: Slow training**
- Enable tf32
- Use gradient accumulation
- Reduce dataloader workers
- Enable pin_memory

## References

- [Gemma-3N-E2B Model Card](https://huggingface.co/google/gemma-3n-E2B)
- [VideoMamba Paper](https://arxiv.org/abs/2403.06977)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/)
