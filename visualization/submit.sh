#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

#SBATCH -A your_account                    # Account name
#SBATCH -q your_qos                                 # Quality of Service (QoS) or queue
#SBATCH --output=./log/visualization/%j.out
#SBATCH --error=./log/visualization/%j.err
#SBATCH --nodes=1                                 # Number of nodes (adjust to your liking)
#SBATCH --ntasks-per-node=1                       # Number of tasks per node
#SBATCH --gpus-per-node=8                         # Number of GPUs per node
#SBATCH --mem=500GB                               # Memory per node
#SBATCH --cpus-per-task=16                        # CPUs per task
#SBATCH --time=72:00:00                          # Time limit

# Set up Weight & Biases API key
export WANDB_API_KEY=''
export WANDB_ENTITY=''
export WANDB_PROJECT=''
EXP_NAME=''

# Paths
OUTPUT_DIR="YOUR_PATH/$EXP_NAME"
mkdir $OUTPUT_DIR

# Set up distributed training environment
GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901

# Run the training script with the current set of parameters
srun torchrun \
  --nproc_per_node $GPUS_PER_NODE \
  --nnodes $SLURM_NNODES \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  visualization/train.py \
  --project_name $WANDB_PROJECT \
  --run_name $EXP_NAME \
  --output_dir $OUTPUT_DIR \
  --num_layers 2 \
  --mode "mlp" \
  --hidden_dim 2048 \
  --lr 1e-5 \
  --save_step 1000 \
  --batch_size 24 \
  --num_tokens 64 \
  --resolution 512 \
  --cfg_prob 0.8 \
  --image_jsons path/to/json \
  --unfreeze_unet