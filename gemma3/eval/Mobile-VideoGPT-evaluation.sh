#!/bin/sh

MODEL_NAME=$1  # e.g., "google/gemma-3n-E2B"
export PYTHONPATH="../:./:$PYTHONPATH"  # Adjust PYTHONPATH for lmms-eval
export SLURM_NTASKS=8  # Number of available GPUs

# Gemma-3N Evaluation Script

# MVBench
torchrun --nproc_per_node $SLURM_NTASKS lmms_eval/__main__.py \
    --model gemma_3n \
    --model_args "model_name=$MODEL_NAME" \
    --tasks "mvbench" \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix gemma_3n \
    --output_path ./logs/

# MLVU
torchrun --nproc_per_node $SLURM_NTASKS lmms_eval/__main__.py \
    --model gemma_3n \
    --model_args "model_name=$MODEL_NAME" \
    --tasks "mlvu" \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix gemma_3n \
    --output_path ./logs/

# NeXt_QA
torchrun --nproc_per_node $SLURM_NTASKS lmms_eval/__main__.py \
    --model gemma_3n \
    --model_args "model_name=$MODEL_NAME" \
    --tasks "nextqa_mc_test" \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix gemma_3n \
    --output_path ./logs/

# EgoSchema
torchrun --nproc_per_node $SLURM_NTASKS lmms_eval/__main__.py \
    --model gemma_3n \
    --model_args "model_name=$MODEL_NAME" \
    --tasks "egoschema" \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix gemma_3n \
    --output_path ./logs/

# ActivityNetQA
torchrun --nproc_per_node $SLURM_NTASKS lmms_eval/__main__.py \
    --model gemma_3n \
    --model_args "model_name=$MODEL_NAME" \
    --tasks "activitynetqa" \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix gemma_3n \
    --output_path ./logs/

# PerceptionTest
torchrun --nproc_per_node $SLURM_NTASKS lmms_eval/__main__.py \
    --model gemma_3n \
    --model_args "model_name=$MODEL_NAME" \
    --tasks "perceptiontest_val_mc" \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix gemma_3n \
    --output_path ./logs/
