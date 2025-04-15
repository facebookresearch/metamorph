#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

#SBATCH -A fair_amaia_cw_video
#SBATCH --partition=learn
#SBATCH -q lowest
#SBATCH --output=./log/metamorph/%j.out
#SBATCH --error=./log/metamorph/%j.err
#SBATCH --nodes=4                              # Number of nodes
#SBATCH --ntasks-per-node=1                    # Number of tasks per node
#SBATCH --gpus-per-node=8
#SBATCH --mem=500GB
#SBATCH --cpus-per-task=128
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@120

# Copyright (c) Meta Platforms, Inc. and affiliates.

# Basic parameters
NUM_IMAGE_TOKENS=64
IMAGE_TOKEN_REDUCTION="interpolation"
MODEL_MAX_LENGTH=4096
VISION_HEAD_TYPE="mlp"
LR=4.9e-5
BATCH_SIZE=14
SAVE_STEPS=240000

# Vision and model parameters
MM_PROJECTOR_TYPE="mlp2x_gelu"
FREEZE_VISION=True
VISION_TOWER="siglip/CLIP-ViT-SO400M-14-384"
MODEL_VERSION="llama3"
USE_VISION_AR=False
NORMALIZE_VISION=True
VISION_COEF=1.0
TUNE_MM_MLP_ADAPTER=True

# Feature flags
BF16=True
FP16=False

# Paths
MODEL_NAME_OR_PATH="PATH_TO_LLM"
EXP_NAME="YOUR_EXP_NAME"
OUTPUT_DIR="YOUR_OUTPUT_DIR/$EXP_NAME"
DATA_PATH="YOUR_DATA_JSONL"

# Additional vision configuration
MM_VISION_SELECT_LAYER=-1
MM_USE_IM_START_END=True
MM_USE_IM_PATCH_TOKEN=False

# Set thread allocation
export OMP_NUM_THREADS=16

# Set up Weight & Biases configuration
export WANDB_API_KEY='YOURKEY'  # Set your W&B Key
export WANDB_ENTITY='NAME'      # Set your W&B entity name
export WANDB_PROJECT='PROJECT'

# Set up distributed training environment
export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901

# Create necessary directories
mkdir -p $OUTPUT_DIR

# ===== Launch Training =====
srun torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $SLURM_NNODES \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    metamorph/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --version "$MODEL_VERSION" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$EXP_NAME" \
    --vision_tower "$VISION_TOWER" \
    --mm_vision_select_layer $MM_VISION_SELECT_LAYER \
    --mm_use_im_start_end $MM_USE_IM_START_END \
    --mm_use_im_patch_token $MM_USE_IM_PATCH_TOKEN \
    --tune_mm_mlp_adapter $TUNE_MM_MLP_ADAPTER \
    --mm_projector_type "$MM_PROJECTOR_TYPE" \
    --bf16 $BF16 \
    --fp16 $FP16 \
    --vision_head_type "$VISION_HEAD_TYPE" \
    --image_token_reduction "$IMAGE_TOKEN_REDUCTION" \
    --num_image_tokens $NUM_IMAGE_TOKENS \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps $SAVE_STEPS \
    --save_total_limit 1 \
    --learning_rate $LR \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length $MODEL_MAX_LENGTH \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --use_vision_ar $USE_VISION_AR \
    --freeze_vision $FREEZE_VISION \
    --vision_coef $VISION_COEF \
    --normalize_vision $NORMALIZE_VISION \
    --report_to $REPORT_TO