#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

# Single-node debug script for Stage 2 Finetuning (adjust parameters as needed)
# Default: Reports to Weights & Biases

# ===== Default Configuration (Adjust for Debugging) =====
# Basic parameters
NUM_IMAGE_TOKENS=64
IMAGE_TOKEN_REDUCTION="interpolation"
MODEL_MAX_LENGTH=4096
VISION_HEAD_TYPE="mlp"
LR=1.23e-5             # Finetuning LR 
BATCH_SIZE=6           # Per device batch size
SAVE_STEPS=240000      # Save steps for debug script (adjust if needed for faster debugging)
NUM_TRAIN_EPOCHS=1

# Vision and model parameters
MM_PROJECTOR_TYPE="mlp2x_gelu"
FREEZE_VISION=True     # Typically True during finetuning if vision tower is pre-trained
VISION_TOWER="siglip/CLIP-ViT-SO400M-14-384"
MODEL_VERSION="llama3" # Corresponds to PROMPT_VERSION
USE_VISION_AR=True
NORMALIZE_VISION=True
VISION_COEF=1.0
TUNE_MM_MLP_ADAPTER=False # Typically False for Stage 2 Finetuning

# Feature flags
BF16=True
FP16=False
TF32=False
GRADIENT_CHECKPOINTING=True

# Paths (!!! IMPORTANT: Set these paths correctly !!!)
MODEL_NAME_OR_PATH="PATH_TO_LLAMA3-8B"                 # Path to the base LLM
PRETRAIN_MM_MLP_ADAPTER="PATH_TO_Pretrained_Adapter"   # Path to the connector trained in Stage 1
DATA_PATH="Path_To_Data_JSONL"                        # Path to the finetuning data JSONL file
EXP_NAME="debug_finetune_$(date +%Y%m%d_%H%M%S)"      # Unique experiment name for output
OUTPUT_DIR="YOUR_DEBUG_OUTPUT_DIR/$EXP_NAME"          # Base directory for saving checkpoints and logs

# Additional vision configuration
MM_VISION_SELECT_LAYER=-1
MM_USE_IM_START_END=True
MM_USE_IM_PATCH_TOKEN=False

# Training configuration
WEIGHT_DECAY=0.
WARMUP_RATIO=0.03
LR_SCHEDULER_TYPE="cosine"
LOGGING_STEPS=1
GRADIENT_ACCUMULATION_STEPS=1
PER_DEVICE_EVAL_BATCH_SIZE=4 
EVALUATION_STRATEGY="no"
SAVE_STRATEGY="steps"
SAVE_TOTAL_LIMIT=1
LAZY_PREPROCESS=True
DATALOADER_NUM_WORKERS=2 # Adjust based on your debug machine's capability

# Set up Weight & Biases configuration (!!! IMPORTANT: Fill in your details !!!)
export WANDB_API_KEY='YOUR_WANDB_API_KEY'      # Set your W&B API Key
export WANDB_ENTITY='YOUR_WANDB_ENTITY'        # Set your W&B entity name (e.g., your username or org name)
export WANDB_PROJECT='YOUR_WANDB_PROJECT_DEBUG' # Set your W&B project name (e.g., 'metamorph-debug')

# Create necessary directories
mkdir -p "$OUTPUT_DIR"
echo "Outputting to $OUTPUT_DIR"
echo "Reporting to WandB: YES (Entity: $WANDB_ENTITY, Project: $WANDB_PROJECT)"

# ===== Launch Training with DeepSpeed =====
# Note: Adjust --num_gpus based on the GPUs available on your debug node
# If using only 1 GPU, gradient_accumulation_steps might need adjustment
# depending on memory constraints and desired global batch size.

deepspeed --num_gpus=8 metamorph/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --version "$MODEL_VERSION" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --vision_tower "$VISION_TOWER" \
    --mm_vision_select_layer $MM_VISION_SELECT_LAYER \
    --mm_use_im_start_end $MM_USE_IM_START_END \
    --mm_use_im_patch_token $MM_USE_IM_PATCH_TOKEN \
    --tune_mm_mlp_adapter $TUNE_MM_MLP_ADAPTER \
    --mm_projector_type "$MM_PROJECTOR_TYPE" \
    --bf16 $BF16 \
    --fp16 $FP16 \
    --tf32 $TF32 \
    --vision_head_type "$VISION_HEAD_TYPE" \
    --image_token_reduction "$IMAGE_TOKEN_REDUCTION" \
    --num_image_tokens $NUM_IMAGE_TOKENS \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "$EVALUATION_STRATEGY" \
    --save_strategy "$SAVE_STRATEGY" \
    --save_steps $SAVE_STEPS \
    --save_total_limit $SAVE_TOTAL_LIMIT \
    --learning_rate $LR \
    --weight_decay $WEIGHT_DECAY \
    --warmup_ratio $WARMUP_RATIO \
    --lr_scheduler_type "$LR_SCHEDULER_TYPE" \
    --logging_steps $LOGGING_STEPS \
    --model_max_length $MODEL_MAX_LENGTH \
    --gradient_checkpointing $GRADIENT_CHECKPOINTING \
    --dataloader_num_workers $DATALOADER_NUM_WORKERS \
    --lazy_preprocess $LAZY_PREPROCESS \
    --use_vision_ar $USE_VISION_AR \
    --freeze_vision $FREEZE_VISION \
    --vision_coef $VISION_COEF \
    --pretrain_mm_mlp_adapter "$PRETRAIN_MM_MLP_ADAPTER" \
    --normalize_vision $NORMALIZE_VISION 

echo "Debug finetuning script finished."