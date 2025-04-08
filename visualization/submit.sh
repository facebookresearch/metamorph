#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

#SBATCH -A your_account                    # Account name
#SBATCH -q your_qos                                 # Quality of Service (QoS) or queue
#SBATCH --output=./log/visualization/%j.out
#SBATCH --error=./log/visualization/%j.err
#SBATCH --nodes=number_of_nodes                                # Number of nodes
#SBATCH --ntasks-per-node=8                       # Number of tasks per node
#SBATCH --gpus-per-node=8                         # Number of GPUs per node
#SBATCH --mem=500GB                               # Memory per node
#SBATCH --cpus-per-task=16                        # CPUs per task
#SBATCH --time=168:00:00                          # Time limit

# Set up Weight & Biases API key
export WANDB_API_KEY=''
export WANDB_ENTITY=

# Set up environment variables
export MASTER_PORT=29500
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

# Run the training script with the current set of parameters
srun python train.py \
  --num_layers 2 \
  --mode "mlp" \
  --run_name "sd1.5_siglip" \
  --hidden_dim 2048 \
  --base_lr 1e-5 \
  --save_step 1000 \
  --batch_size 24 \
  --num_tokens 64 \
  --resolution 512 \
  --cfg_prob 0.8 \
  --unfreeze_unet