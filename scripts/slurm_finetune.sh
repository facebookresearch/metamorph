#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

#SBATCH -A Your_Account_Name                    # Account name
#SBATCH -q Your_QOS                            # Quality of Service (QoS) or queue
#SBATCH --output=./log/metamorph/%j.out
#SBATCH --error=./log/metamorph/%j.err
#SBATCH --nodes=32                              # Number of nodes, Change Here and also change LR with sqrt rule
#SBATCH --ntasks-per-node=1                    # Number of tasks per node
#SBATCH --gpus-per-node=8
#SBATCH --mem=500GB
#SBATCH --cpus-per-task=128
#SBATCH --time=96:00:00
#SBATCH --signal=B:USR1@120

# ===== Default Configuration =====
# Basic parameters
NUM_IMAGE_TOKENS=64
IMAGE_TOKEN_REDUCTION="interpolation"
MODEL_MAX_LENGTH=4096
VISION_HEAD_TYPE="mlp"
LR=3.74e-5
BATCH_SIZE=6
SAVE_STEPS=500

# Vision and model parameters
MM_PROJECTOR_TYPE="mlp2x_gelu"
FREEZE_VISION=True
VISION_TOWER="siglip/CLIP-ViT-SO400M-14-384"
MODEL_VERSION="llama3"
USE_VISION_AR=True
NORMALIZE_VISION=True
VISION_COEF=1.0
TUNE_MM_MLP_ADAPTER=False # In Stage 2, both the MLP and LLM are finetuned.

# Feature flags
BF16=True
FP16=False

# Paths
MODEL_NAME_OR_PATH="PATH_TO_LLM"
PRETRAIN_MM_MLP_ADAPTER="PATH_TO_PRETRAINED_CONNECTOR"
EXP_NAME="YOUR_EXP_NAME"
OUTPUT_DIR="YOUR_OUTPUT_DIR/$EXP_NAME"
DATA_PATH="YOUR_DATA_JSONL"

# Additional vision configuration
MM_VISION_SELECT_LAYER=-1
MM_USE_IM_START_END=True
MM_USE_IM_PATCH_TOKEN=False

# ===== Command Line Argument Parsing =====
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --num_image_tokens) NUM_IMAGE_TOKENS="$2"; shift ;;
        --image_token_reduction) IMAGE_TOKEN_REDUCTION="$2"; shift ;;
        --model_max_length) MODEL_MAX_LENGTH="$2"; shift ;;
        --vision_head_type) VISION_HEAD_TYPE="$2"; shift ;;
        --lr) LR="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        --bf16) BF16="$2"; shift ;;
        --fp16) FP16="$2"; shift ;;
        --msg) MSG="$2"; shift ;;
        --save_steps) SAVE_STEPS="$2"; shift ;;
        --mm_projector_type) MM_PROJECTOR_TYPE="$2"; shift ;;
        --freeze_vision) FREEZE_VISION="$2"; shift ;;
        --vision_tower) VISION_TOWER="$2"; shift ;;
        --model_version) MODEL_VERSION="$2"; shift ;;
        --use_vision_ar) USE_VISION_AR="$2"; shift ;;
        --vision_coef) VISION_COEF="$2"; shift ;;
        --tune_mm_mlp_adapter) TUNE_MM_MLP_ADAPTER="$2"; shift ;;
        --pretrain_mm_mlp_adapter) PRETRAIN_MM_MLP_ADAPTER="$2"; shift ;;
        --normalize_vision) NORMALIZE_VISION="$2"; shift ;;
        --model_name_or_path) MODEL_NAME_OR_PATH="$2"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

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
    --deepspeed ./scripts/zero3.json \
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
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --use_vision_ar $USE_VISION_AR \
    --freeze_vision $FREEZE_VISION \
    --vision_coef $VISION_COEF \
    --pretrain_mm_mlp_adapter "$PRETRAIN_MM_MLP_ADAPTER" \
    --normalize_vision $NORMALIZE_VISION \
    --report_to wandb