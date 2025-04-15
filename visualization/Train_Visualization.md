# Finetuning Stable Diffusion 1.5 to visualize SigLIP embedding

Here, we take prelimenary step to finetune Stable Diffusion 1.5 with SigLIP embeddings. The implementation includes distributed training support with SLURM system.

## Requirements
(Not needed if you already installed the main MetaMorph repo dependencies).

```
torch
diffusers
transformers
wandb
webdataset
pillow
tqdm
```

## Setup

1. Install the required packages (not needed if you already installed the main MetaMorph repo dependencies):
```bash
pip install torch diffusers transformers wandb webdataset pillow tqdm
```

2. Configure your SLURM environment variables in `submit.sh`:
```bash
export WANDB_API_KEY='your_key_here'
export WANDB_ENTITY='your_entity'
```

3. Prepare your dataset as WebDataset tar files and create a JSON file listing their paths.

## Training

### Configuration

Key parameters in `train.py`:

- `--mode`: Projection mode ('mlp', 'xattn', 'transformer')
- `--num_tokens`: Number of tokens for SigLIP encoder (default: 64)
- `--hidden_dim`: Hidden dimension size (default: 4096, empirically we find the larger the better.)
- `--num_layers`: Number of layers in projector (default: 3, empirically, we find 2 or 3 works the best)
- `--batch_size`: Batch size per GPU
- `--lr`: Base learning rate
- `--cfg_prob`: Classifier-free guidance probability
- `--noise_offset`: Amount of noise offset to add
- `--unfreeze_unet`: Whether to train the UNet
- `--unet_from_scratch`: Train UNet from scratch
- `--image_jsons`: A json file contains the list of tars that contain image. This code by default assumes dataloading from tar vis wds. 

For reference and suggestions on learning rate and batch size, see the [Stable Diffusion v1.5 documentation](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5). This can help you optimize your training parameters for better results.

### Running Training

1. Modify the SLURM script (`submit.sh`) according to your cluster configuration:

```bash
#SBATCH --nodes=number_of_nodes
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --mem=500GB
#SBATCH --cpus-per-task=16
```

2. Submit the training job:

```bash
sbatch submit.sh
```

### Example Training Command

```bash
srun python train.py \
  --num_layers 2 \
  --mode "mlp" \
  --run_name "sd1.5_siglip" \
  --hidden_dim 2048 \
  --lr 1e-5 \
  --save_step 1000 \
  --batch_size 24 \
  --num_tokens 64 \
  --resolution 512 \
  --cfg_prob 0.8 \
  --image_jsons path/to/json \
  --unfreeze_unet
```

## Model Architecture

The implementation includes three main components:

1. **SigLIP Encoder**: Extracts visual features from images
2. **Projector**: Projects SigLIP embeddings to the UNet conditioning space
3. **UNet**: Modified Stable Diffusion UNet that can be optionally finetuned

### Projection Modes

- **MLP**: Simple multi-layer perceptron
- **Cross-attention**: Cross-attention based projection with learnable tokens
- **Transformer**: Full transformer-based projection

