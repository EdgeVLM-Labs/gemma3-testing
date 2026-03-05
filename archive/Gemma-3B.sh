#!/bin/sh

# ==================================================
# Mobile-VideoGPT Training Script for Gemma-3N-E2B
# ==================================================

export DATASET_DIR=/Mobile-VideoGPT/playground/data
export PYTHONPATH="./:$PYTHONPATH"

# -----------------------
# Model & Vision Towers
# -----------------------
BASE_LLM_PATH=google/gemma-3n-E2B
VIDEO_VISION_TOWER=OpenGVLab/VideoMamba
IMAGE_VISION_TOWER=openai/clip-vit-base-patch16
PROJECTOR_TYPE=etp

# -----------------------
# Output Directories
# -----------------------
OUTPUT_VIDEO_PRETRAIN=results/pretrain/etp_gemma3n_e2b_video
OUTPUT_IMAGE_PRETRAIN=results/pretrain/etp_gemma3n_e2b_clip
OUTPUT_FINETUNE=results/mobilevideogpt_finetune_gemma3n_e2b

# -----------------------
# Stage 1: Video Projection Pretraining
# -----------------------
deepspeed gemma3/train/pretrain.py \
--deepspeed scripts/zero2.json \
--tune_mm_mlp_adapter True \
--model_name_or_path "$BASE_LLM_PATH" \
--version gemma3n_e2b \
--dataset_use PRETRAINING \
--vision_tower "$VIDEO_VISION_TOWER" \
--mm_projector_type "$PROJECTOR_TYPE" \
--mm_vision_select_layer -2 \
--mm_use_im_start_end False \
--mm_use_im_patch_token False \
--bf16 True \
--output_dir $OUTPUT_VIDEO_PRETRAIN \
--num_train_epochs 2 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 2 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 50000 \
--save_total_limit 1 \
--learning_rate 1e-3 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 True \
--model_max_length 4096 \
--gradient_checkpointing True \
--dataloader_num_workers 4 \
--lazy_preprocess True \
--report_to none \
--num_select_k_frames_in_chunk 4 \
--topk True

# -----------------------
# Stage 2: Image Projection Pretraining
# -----------------------
deepspeed gemma3/train/pretrain.py \
--deepspeed scripts/zero2.json \
--tune_image_mm_mlp_adapter True \
--model_name_or_path "$BASE_LLM_PATH" \
--version gemma3n_e2b \
--dataset_use PRETRAINING \
--image_vision_tower "$IMAGE_VISION_TOWER" \
--image_mm_projector_type "$PROJECTOR_TYPE" \
--mm_vision_select_layer -2 \
--mm_use_im_start_end False \
--mm_use_im_patch_token False \
--bf16 True \
--output_dir $OUTPUT_IMAGE_PRETRAIN \
--num_train_epochs 2 \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 2 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 50000 \
--save_total_limit 1 \
--learning_rate 1e-3 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 True \
--model_max_length 4096 \
--gradient_checkpointing True \
--dataloader_num_workers 4 \
--lazy_preprocess True \
--report_to none \
--num_select_k_frames_in_chunk 4 \
--topk True

# -----------------------
# Stage 3: Mobile-VideoGPT Finetuning
# -----------------------
PRETRAIN_VIDEO_MLP_PATH=$OUTPUT_VIDEO_PRETRAIN/mm_projector.bin
PRETRAIN_IMAGE_MLP_PATH=$OUTPUT_IMAGE_PRETRAIN/mm_projector.bin

deepspeed gemma3/train/train.py \
--lora_enable True \
--lora_r 128 \
--lora_alpha 256 \
--mm_projector_lr 2e-5 \
--deepspeed scripts/zero3.json \
--model_name_or_path "$BASE_LLM_PATH" \
--version gemma3n_e2b \
--dataset_use MobileGPT \
--vision_tower "$VIDEO_VISION_TOWER" \
--image_vision_tower "$IMAGE_VISION_TOWER" \
--mm_projector_type "$PROJECTOR_TYPE" \
--image_mm_projector_type "$PROJECTOR_TYPE" \
--pretrain_mm_mlp_adapter "$PRETRAIN_VIDEO_MLP_PATH" \
--pretrain_image_mm_mlp_adapter "$PRETRAIN_IMAGE_MLP_PATH" \
--mm_vision_select_layer -2 \
--mm_use_im_start_end False \
--mm_use_im_patch_token False \
--image_aspect_ratio pad \
--group_by_modality_length True \
--bf16 True \
--output_dir $OUTPUT_FINETUNE \
--num_train_epochs 2 \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 2 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 50000 \
--save_total_limit 1 \
--learning_rate 2e-4 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 True \
--model_max_length 4096 \
--gradient_checkpointing True \
--dataloader_num_workers 4 \
--lazy_preprocess True \
--report_to none \
--num_select_k_frames_in_chunk 4 \
--topk True
