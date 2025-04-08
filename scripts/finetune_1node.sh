#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.


########### DO NOT CHANGE ###########
########### USE THIS FOR BOTH ###########
PROMPT_VERSION=llama3
########### DO NOT CHANGE ###########

deepspeed metamorph/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path PATH_TO_LLAMA3-8B \
    --version $PROMPT_VERSION \
    --pretrain_mm_mlp_adapter PATH_TO_Pretrained_Adapter\
    --image_folder PATH_TO_Images/You can also hardcode the image path directly into the json \
    --data_path Path_To_Data_JSONL\
    --output_dir PATH_TO_OUTPUT_DIR\
    --vision_tower  siglip/CLIP-ViT-SO400M-14-384\
    --use_vision_ar True \
    --mm_projector_type mlp2x_gelu \
    --normalize_vision True \
    --mm_vision_select_layer -1 \
    --mm_use_im_start_end True \
    --mm_use_im_patch_token False \
    --image_token_reduction 'interpolation' \
    --num_image_tokens 64 \
    --vision_coef 1.0 \
    --bf16 True \
    --fp16 False \
    --freeze_vision True \
    --vision_head_type "mlp2x_gelu" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 240000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
