# Training MetaMorph Guide

This guide explains how to train MetaMorph.

## Overview

MetaMorph training follows a two-stage approach:
1. **Pretraining the MLP Connector**: Connects vision and language representations.
2. **Fine-tuning**: Optimizes both the LLM and the connector together.

## Key Training Parameters

### Basic Model Configuration

| Parameter | Description | Default Value |
|-----------|-------------|------------------|
| `--model_name_or_path` | Path to the base language model | `PATH_TO_LLAMA3-8B` |
| `--version` | Conversation template version | `llama3` |
| `--model_max_length` | Maximum sequence length for training | `4096` |
| `--output_dir` | Directory to save model checkpoints | `PATH_TO_OUTPUT_DIR` |

### Vision Tower Configuration

| Parameter | Description | Default Value |
|-----------|-------------|------------------|
| `--vision_tower` | Vision model used for processing images | `siglip/CLIP-ViT-SO400M-14-384` |
| `--mm_vision_select_layer` | Which layer of vision model to use | `-1` (last layer) |
| `--freeze_vision` | Whether to freeze vision backbone | `True` |
| `--normalize_vision` | Whether to normalize vision embeddings | `True` |

### Image Token Configuration

| Parameter | Description | Default Value |
|-----------|-------------|------------------|
| `--image_token_reduction` | Method to reduce image tokens | `interpolation` |
| `--num_image_tokens` | Number of tokens used per image | `64` |
| `--mm_use_im_start_end` | Use special image start/end tokens | `True` |
| `--mm_use_im_patch_token` | Use image patch tokens | `False` |

### Multimodal Projector Configuration

| Parameter | Description | Default Value |
|-----------|-------------|------------------|
| `--mm_projector_type` | Type of projector to map vision to language | `mlp2x_gelu` |
| `--tune_mm_mlp_adapter` | Whether to tune only the adapter (for pretraining) | `True` (pretraining), `False` (finetuning) |
| `--pretrain_mm_mlp_adapter` | Path to pretrained adapter (for finetuning) | `PATH_TO_Pretrained_Adapter` |

### Visual Auto-Regressive Parameters

| Parameter | Description | Default Value |
|-----------|-------------|------------------|
| `--use_vision_ar` | Enable vision auto-regressive prediction | `False` (pretraining), `True` (finetuning) |
| `--vision_head_type` | Type of vision head for AR prediction | `mlp` |

### Optimization Parameters

| Parameter | Description | Default Value |
|-----------|-------------|------------------|
| `--learning_rate` | Base learning rate | Please adjust based on your batch size|
| `--weight_decay` | Weight decay for AdamW optimizer | `0.0` |
| `--warmup_ratio` | Ratio of steps for learning rate warmup | `0.03` |
| `--lr_scheduler_type` | Type of learning rate scheduler | `cosine` |
| `--bf16` | Use bfloat16 precision | `True` |
| `--fp16` | Use float16 precision | `False` |


### Data Parameters

| Parameter | Description | Recommended Value |
|-----------|-------------|------------------|
| `--data_path` | Path to training data | `Path_To_Data_JSONL` |
| `--image_folder` | Path to image directory | `PATH_TO_Images` |

## Training Scripts

### Debug / Non-SLURM system

1. Stage 1: Use the `scripts/pretrain_1node.sh` script to pretrain the MLP connector.
2. Stage 2: Use the `scripts/finetune_1node.sh` script to finetune the full model.

### SLURM / Multi-Node Training

1. Stage 1: Use the `scripts/slurm_pretrain.sh` script to pretrain the MLP connector.
2. Stage 2: Use the `scripts/slurm_finetune.sh` script to finetune the full model.

## Data Format

MetaMorph supports data in the following format:

```json
{
  "id": "unique_id",
  "image": "path/to/image.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image> What is shown in this image?"
    },
    {
      "from": "gpt",
      "value": "This is a detailed description of the image."
    }
  ]
}
```

## Recommended Training Strategy

1. **Pretrain the MLP connector** with `tune_mm_mlp_adapter=True` and `use_vision_ar=False`. This stage is focused on effectively connecting the vision and language models.

2. **Fine-tune the full model** with the pretrained adapter, setting `use_vision_ar=True` to enable visual generation capabilities.

3. For best results with limited resources, **adjust batch size and gradient accumulation steps** to maintain the effective global batch size. The formula is:
   - Global Batch Size = per_device_train_batch_size × gradient_accumulation_steps × num_gpus × num_nodes

## Learning Rate Calculation

When using a different batch size than the original implementation, adjust the learning rate using:

Optimal Learning Rate = Base Learning Rate * √(Batch Size / Base Batch Size)

For example, if the base learning rate is 6.93e-5 for a batch size of 1536, and you're using a batch size of 768, the optimal learning rate would be:
6.93e-5 * √(768/1536) = 4.9e-5

## Visualization Training

For training the visualization component (to generate images from SigLIP embeddings), use the scripts in the `visualization/` directory. The key parameters are explained in `visualization/Train_Visualization.md`.