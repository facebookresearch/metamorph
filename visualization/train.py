# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoProcessor, AutoModel
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler, DPMSolverMultistepScheduler
from torchvision import transforms
from PIL import Image
import logging
import wandb
from tqdm import tqdm
import argparse
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SigLIPEncoder(nn.Module):
    def __init__(self, model_name="google/siglip-so400m-patch14-384", num_tokens=64, device=None):
        super().__init__()
        self.model_name = model_name
        self.num_tokens = num_tokens
        self.hidden_size = 1152  # SigLIP-SO400M hidden size
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.load_model()

    def load_model(self):
        model = AutoModel.from_pretrained(self.model_name)
        processor = AutoProcessor.from_pretrained(self.model_name)
        
        self.vision_tower = model.vision_model.to(self.device)
        self.processor = processor
        
    def forward(self, images):
        if images.dim() == 3:
            images = images.unsqueeze(0)
        images = images.to(self.device)

        outputs = self.vision_tower(images, output_hidden_states=True)
        image_features = outputs.hidden_states[-1]
        
        b, num_tokens, dim = image_features.shape
        h = w = int(num_tokens**0.5)
        target_h = target_w = int(self.num_tokens**0.5)

        if self.num_tokens!=729:
            
            image_features = image_features.view(b, h, w, dim)
            image_features = image_features.permute(0, 3, 1, 2)
            image_features = F.interpolate(image_features, size=(target_h, target_w), mode='bilinear', align_corners=False)
            image_features = image_features.permute(0, 2, 3, 1).contiguous().view(b, self.num_tokens, dim)

        # Normalize vision if needed
        image_features = F.normalize(image_features, p=2, dim=-1)

        
        return image_features

    def encode_image(self, image):
        if not isinstance(image, torch.Tensor):
            image = self.processor(images=image, return_tensors="pt")['pixel_values']
        image = image.to(self.device)
        
        with torch.no_grad():
            features = self(image)
        
        return features


import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import webdataset as wds
import io
from PIL import Image
import math

def lr_lambda(current_step):
    target=16000
    peak=target//4
    if current_step < peak:
        # Logarithmic warm-up
        warmup_steps = peak
        lr_multiplier = math.log(current_step + 1) / math.log(warmup_steps + 1)
        return lr_multiplier
    elif current_step < target:
        # Linear decay
        decay_steps = target-peak
        decay_progress = (current_step - peak) / decay_steps
        lr_multiplier = 1.0 - decay_progress
        return lr_multiplier
    else:
        return 0.0


class CustomDataset(wds.WebDataset):
    def __init__(self, urls, siglip_processor, center_crop=True, resolution=512):
        # super().__init__(urls, resampled=True)
        super().__init__(urls, resampled=True)

        self.siglip_processor = siglip_processor
        
        size = resolution  # As specified for VAE
        self.vae_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __iter__(self):
        for item in super().__iter__():
            try:
                # Load image from WebDataset
     
                image = Image.open(io.BytesIO(item['jpeg'])).convert('RGB')
                
                # SigLIP preprocessing
                siglip_image = self.siglip_processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
                
                # VAE preprocessing
                vae_image = self.vae_transforms(image)
                
                yield siglip_image, vae_image
            except Exception as e:
                print(f"Error processing image: {str(e)}")
                continue


import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, include_ffn=False):
        super().__init__()
        self.include_ffn = include_ffn
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        if self.include_ffn:
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, 4 * embed_dim),
                nn.ReLU(),
                nn.Linear(4 * embed_dim, embed_dim)
            )
            self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, tokens, x_proj):
        # Cross-Attention
        attn_output, _ = self.cross_attn(query=tokens, key=x_proj, value=x_proj)
        tokens = self.norm1(tokens + attn_output)
        if self.include_ffn:
            # Feed-Forward Network
            ffn_output = self.ffn(tokens)
            tokens = self.norm2(tokens + ffn_output)
        return tokens

from types import SimpleNamespace

class SimplifiedSigLIPProjector(nn.Module):
    def __init__(self, input_dim=1152, hidden_dim=4096, output_dim=768, num_tokens=77, num_layers=6, num_heads=8, mode='mlp'):
        super().__init__()
        self.config = SimpleNamespace(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_tokens=num_tokens,
            num_layers=num_layers,
            num_heads=num_heads,
            mode=mode
        )

        self.mode = mode
        self.num_layers = num_layers
        
        if self.mode == 'mlp':
            self.layers = nn.ModuleList()
            self.norms = nn.ModuleList()
            
            # First layer
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
            
            # Middle layers
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.norms.append(nn.LayerNorm(hidden_dim))
            
            # Last layer
            self.layers.append(nn.Linear(hidden_dim, output_dim))
            self.norms.append(nn.LayerNorm(output_dim))
            
        elif self.mode in ['xattn', 'xattnffn']:
            self.num_tokens = num_tokens
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.num_heads = num_heads
            # Learnable token embeddings
            self.token_embeddings = nn.Parameter(torch.randn(1, num_tokens, output_dim))
            # Input projection to match output_dim
            self.proj = nn.Linear(input_dim, output_dim)
            self.input_norm = nn.LayerNorm(output_dim)
            # Cross-Attention Blocks
            include_ffn = (self.mode == 'xattnffn')
            self.cross_attn_layers = nn.ModuleList([
                CrossAttentionBlock(embed_dim=output_dim, num_heads=num_heads, include_ffn=include_ffn)
                for _ in range(num_layers)
            ])
        elif self.mode == 'transformer':
            self.fc = nn.Linear(input_dim, hidden_dim)
            self.tfm = nn.Transformer(
                batch_first=True,
                norm_first=True,
                d_model=hidden_dim,
                num_encoder_layers=num_layers,
                num_decoder_layers=num_layers,
                dim_feedforward=hidden_dim * 4,
                dropout=0.0,
                nhead=4
            )
            self.model = nn.Linear(hidden_dim, output_dim)
            self.query_embs = nn.Parameter(torch.randn(1, num_tokens, hidden_dim))
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
    @classmethod
    def from_config(cls, config):
        return cls(**vars(config))
    
    def forward(self, x):
        if self.mode == 'mlp':
            for layer, norm in zip(self.layers[:-1], self.norms[:-1]):
                x = F.relu(norm(layer(x)))
            return self.norms[-1](self.layers[-1](x))
        elif self.mode in ['xattn', 'xattnffn']:
            # x: (batch_size, seq_len, input_dim)
            batch_size = x.size(0)
            # Project input to output_dim
            x_proj = self.proj(x)  # (batch_size, seq_len, output_dim)
            x_proj = self.input_norm(x_proj)
            # Initialize tokens
            tokens = self.token_embeddings.expand(batch_size, -1, -1)  # (batch_size, num_tokens, output_dim)
            # Pass through Cross-Attention Blocks
            for layer in self.cross_attn_layers:
                tokens = layer(tokens, x_proj)
            return tokens  # (batch_size, num_tokens, output_dim)
        elif self.mode == 'transformer':
            x = self.fc(x)  # (batch_size, seq_len, hidden_dim)
            query_embs = self.query_embs.repeat(x.shape[0], 1, 1)  # (batch_size, num_tokens, hidden_dim)
            x = self.tfm(x, query_embs)  # (batch_size, num_tokens, hidden_dim)
            outputs = self.model(x)  # (batch_size, num_tokens, output_dim)
            return outputs


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = os.environ['SLURM_LAUNCH_NODE_IPADDR']
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def save_checkpoint(model, optimizer, scheduler, global_step, path, save_unet=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),  # Save scheduler state
        'global_step': global_step,
    }
    
    if not save_unet:
        checkpoint['model_config'] = vars(model.module.config)
        # Save config separately as JSON
        config_path = os.path.join(os.path.dirname(path), 'model_config.json')
        with open(config_path, 'w') as f:
            json.dump(vars(model.module.config), f, indent=2)
            
    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, scheduler, path):
    checkpoint = torch.load(path, map_location='cpu')  # Load to CPU first
    model.module.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # Load scheduler state
    return checkpoint['global_step']

def get_latest_checkpoint(save_dir):
    checkpoints = glob.glob(f"{save_dir}/checkpoint_step_*.pt")
    if not checkpoints:
        return None
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return latest_checkpoint



def unfreeze_unet(unet, learning_rate):
    for param in unet.parameters():
        param.requires_grad = True
    return torch.optim.AdamW(
        unet.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel


def load_models(args, device):
    siglip_encoder = SigLIPEncoder(num_tokens=args.num_tokens, device=device)
    siglip_processor = siglip_encoder.processor

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="vae", 
        revision=args.revision, 
        variant=args.variant
    ).to(device)

    if args.unet_from_scratch:
        # Load the configuration of the pretrained UNet
        unet_config = UNet2DConditionModel.load_config(
            args.pretrained_model_name_or_path,
            subfolder="unet",
            revision=args.non_ema_revision
        )
        # Initialize the UNet from the configuration (random weights)
        unet = UNet2DConditionModel.from_config(unet_config).to(device)
    else:
        # Load the pretrained UNet
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, 
            subfolder="unet", 
            revision=args.non_ema_revision
        ).to(device)

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="scheduler"
    )

    for model in [siglip_encoder, vae]:
        for param in model.parameters():
            param.requires_grad = False

    return siglip_encoder, siglip_processor, vae, unet, noise_scheduler

def lr_lambda_warmup_constant(current_step):
    if current_step < 10000:
        # Linear warm-up to 0.0001 over 10,000 steps
        return float(current_step) / 10000
    else:
        # Keep the learning rate constant after warm-up
        return 1.0


def train(local_rank, world_size, args):
    global_rank = int(os.environ['SLURM_PROCID'])
    setup(global_rank, world_size)
    
    global_batch_size = args.batch_size * world_size
    learning_rate = args.base_lr

    if args.unfreeze_unet:
        unfreeze = True
    else:
        unfreeze = False
    
    save_dir = os.path.join(
        args.output_dir,
        f"{args.run_name}_bs{global_batch_size}_tokens{args.num_tokens}_unfreeze{unfreeze}_cfg{args.cfg_prob}_noise{args.noise_offset}"
    )   
    os.makedirs(save_dir, exist_ok=True)
        
    if global_rank == 0:
        wandb.init(project="siglip-stable-diffusion-multi-node", 
                   config=vars(args),
                   name=os.path.basename(save_dir))
        
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    siglip_encoder, siglip_processor, vae, unet, noise_scheduler = load_models(args, device)
    
    projector = SimplifiedSigLIPProjector(mode=args.mode, hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)

    projector = DDP(projector, device_ids=[local_rank])
    unet = DDP(unet, device_ids=[local_rank])
    
    optimizer = torch.optim.AdamW(
        projector.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )

    
    
    if args.unfreeze_unet:
        unet_optimizer = unfreeze_unet(unet, learning_rate)
        if args.unet_from_scratch:
            unet_scheduler = torch.optim.lr_scheduler.LambdaLR(unet_optimizer, lr_lambda=lr_lambda_warmup_constant)
        else:
            unet_scheduler = torch.optim.lr_scheduler.LambdaLR(unet_optimizer, lr_lambda=lr_lambda)
    else:
        for param in unet.parameters():
            param.requires_grad = False
    if args.unet_from_scratch:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda_warmup_constant)
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


    # Load latest checkpoint if it exists
    global_step = 0
    latest_checkpoint = get_latest_checkpoint(save_dir)
    if latest_checkpoint:
        global_step = load_checkpoint(projector, optimizer, scheduler, latest_checkpoint)
        if args.unfreeze_unet:
            # Extract the directory and the filename separately
            directory, filename = os.path.split(latest_checkpoint)

            # Replace 'checkpoint' in the filename
            new_filename = filename.replace('checkpoint', 'unet_checkpoint')

            # Join the directory back with the new filename
            unet_latest_checkpoint = os.path.join(directory, new_filename)         
            if os.path.exists(unet_latest_checkpoint):
                try:
                    _ = load_checkpoint(unet, unet_optimizer, unet_scheduler, unet_latest_checkpoint)
                    if global_rank == 0:
                        logger.info(f"Successfully loaded UNet checkpoint")
                except Exception as e:
                    if global_rank == 0:
                        logger.error(f"Error loading UNet checkpoint: {str(e)}")
                    raise e

        if global_rank == 0:
            logger.info(f"Resuming from checkpoint: {latest_checkpoint}")
            logger.info(f"Resumed at Global step: {global_step}")


    #################################################################################
    ## The code assumes to train on a list(json) of tars and load it in WDS format ##
    #################################################################################

    # Use the default MetaCLIP path
    with open(args.image_jsons, 'r') as f:
        shard_files = json.load(f)
    
    # Sort shard files to ensure consistency across nodes
    shard_files.sort()

    # Calculate total number of shards and shards per GPU
    total_shards = len(shard_files)
    shards_per_gpu = total_shards // world_size
    
    # Assign shards to this GPU
    start_shard = local_rank * shards_per_gpu
    end_shard = start_shard + shards_per_gpu
    if local_rank == world_size - 1:  # Last GPU takes any remaining shards
        end_shard = total_shards
    
    urls = shard_files[start_shard:end_shard]

    dataset = CustomDataset(urls, siglip_processor, resolution=args.resolution)

    # Estimate number of samples (approximately 10,000 per shard)
    estimated_samples_per_shard = 10000
    total_samples = len(urls) * estimated_samples_per_shard
    estimated_steps = total_samples // args.batch_size
    
    # WebDataset-specific dataloader
    dataloader = wds.WebLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,  # WebDataset handles shuffling internally
    )

    if global_rank == 0:
        pbar = tqdm(total=estimated_steps, initial=global_step, desc="Training Progress")

    
    for batch_idx, (siglip_images, vae_images) in enumerate(dataloader):
        if batch_idx < global_step:
            continue  # Skip already processed batches

        siglip_images = siglip_images.to(device)
        vae_images = vae_images.to(device)
        
        with torch.no_grad():
            siglip_embeddings = siglip_encoder.encode_image(siglip_images)
        
        projected_embeddings = projector(siglip_embeddings)
        
        if args.num_tokens < 77:
            padding = torch.zeros(projected_embeddings.size(0), 77 - args.num_tokens, projected_embeddings.size(-1), device=device)
            padded_embeddings = torch.cat([projected_embeddings, padding], dim=1)
        else:
            padded_embeddings = projected_embeddings
        
        # Implement classifier-free guidance (CFG) by randomly dropping conditioning
        cfg_prob = args.cfg_prob  # Should be 0.1 as per your requirement
        batch_size = padded_embeddings.size(0)

        # Create a random mask for the batch
        mask = torch.rand(batch_size, device=device) < cfg_prob

        # Set embeddings to zero where mask is True
        padded_embeddings[mask] = 0

        latents = vae.encode(vae_images).latent_dist.sample()
        latents = latents * vae.config.scaling_factor



        noise = torch.randn_like(latents)
        if args.noise_offset > 0:
            noise_offset = args.noise_offset
            # Generate additional noise
            offset_noise = torch.randn(
                (latents.shape[0], latents.shape[1], 1, 1), 
                device=latents.device
            )
            # Add the noise offset to the noise tensor
            noise = noise + noise_offset * offset_noise

        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (args.batch_size,), device=device)
        # noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)


        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=padded_embeddings).sample
        
        loss = F.mse_loss(noise_pred, noise)
        
        optimizer.zero_grad()
        if args.unfreeze_unet:
            unet_optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        scheduler.step()

        if args.unfreeze_unet:
            unet_optimizer.step()
            unet_scheduler.step()

        global_step += 1
        
        if global_rank == 0:
            # Get current learning rates
            current_lr = scheduler.get_last_lr()[0]
            log_dict = {
                "loss": loss.item(),
                "global_step": global_step,
                "lr": current_lr
            }
            if args.unfreeze_unet:
                unet_lr = unet_scheduler.get_last_lr()[0]
                log_dict["unet_lr"] = unet_lr
            wandb.log(log_dict)
            pbar.update(1)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Save checkpoint every save_step steps
        if global_step % args.save_step == 0:
            if global_rank == 0:
                logger.info(f"Saving checkpoint at step {global_step}")
                save_checkpoint(
                    projector, optimizer, scheduler, global_step,
                    f"{save_dir}/checkpoint_step_{global_step}.pt"
                )
                if args.unfreeze_unet:
                    save_checkpoint(
                        unet, unet_optimizer, unet_scheduler, global_step,
                        f"{save_dir}/unet_checkpoint_step_{global_step}.pt",
                        save_unet=True
                    )

        if args.unet_from_scratch:
            if global_step >= 1000001:
                    break
        else:

            if global_step >= args.training_steps+1:
                break

    if global_rank == 0:
        pbar.close()
        logger.info(f"Training completed. Total steps: {global_step}")
        wandb.save(f"{save_dir}/siglip_projector.pth")
        wandb.save(f"{save_dir}/finetuned_unet.pth")
        wandb.finish()

    cleanup()




def parse_args():
    parser = argparse.ArgumentParser(description="SigLIP-SD Training Script")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--resolution", type=int, default=512, help="Resolution")
    parser.add_argument("--base_lr", type=float, default=1e-6, help="Base learning rate")
    parser.add_argument("--run_name", type=str, default="siglip_sd_run", help="Name of the run")
    parser.add_argument("--output_dir", type=str, default="outputs/siglip_sd_runs", help="Directory to save model checkpoints and logs")
    parser.add_argument("--image_jsons", type=str, default="non_empty_tars.json", help="The json containing list of jsons of image tars")
    parser.add_argument("--mode", type=str, default="mlp", help="Mode of connector")
    parser.add_argument("--num_tokens", type=int, default=64, help="Number of tokens for SigLIP encoder")
    parser.add_argument("--hidden_dim", type=int, default=4096, help="Hidden dim")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of xattn layers")
    parser.add_argument("--save_step", type=int, default=5000, help="Save checkpoint every n steps")
    parser.add_argument("--training_steps", type=int, default=12000, help="Save checkpoint every n steps")
    parser.add_argument("--unfreeze_unet", action="store_true", help="Unfreeze and train UNet")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stable-diffusion-v1-5/stable-diffusion-v1-5", help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--revision", type=str, default=None, help="Revision of pretrained model identifier from huggingface.co/models")
    parser.add_argument("--variant", type=str, default=None, help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16")
    parser.add_argument("--non_ema_revision", type=str, default=None, help="Revision of pretrained non-ema model identifier from huggingface.co/models")
    parser.add_argument("--cfg_prob", type=float, default=0.1, help="Probability of dropping conditioning for CFG")
    parser.add_argument("--noise_offset", type=float, default=0.0, help="Amount of noise offset to add")
    parser.add_argument("--unet_from_scratch", action="store_true", help="Train UNet from scratch when unfreeze_unet is set")
    parser.add_argument("--zero_cfg", action="store_true", help="Train cfg with zero empty string")

    return parser.parse_args()

def main():
    args = parse_args()
    world_size = int(os.environ['SLURM_NTASKS'])
    local_rank = int(os.environ['SLURM_LOCALID'])
    global_rank = int(os.environ['SLURM_PROCID'])

    if local_rank == 0:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
        logger.info(f"Starting training with world size: {world_size}")

    if global_rank == 0:
        wandb.init(project="siglip-stable-diffusion-multi-node", 
                   config=vars(args),
                   name=args.run_name)

    train(local_rank, world_size, args)

    if global_rank == 0:
        wandb.finish()

if __name__ == "__main__":
    main()