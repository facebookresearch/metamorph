import torch
from diffusers import StableDiffusionPipeline
from huggingface_hub import hf_hub_download
import os
import json
from adapter import SimplifiedSigLIPProjector


def _download_checkpoints(repo_id: str, local_dir: str, token: str = None):
    """Internal function to download checkpoints from HuggingFace"""
    os.makedirs(local_dir, exist_ok=True)
    
    files = {
        "unet": "unet_checkpoint.pt",
        "adapter": "adapter_checkpoint.pt",
        "config": "model_config.json"
    }
    
    paths = {}
    for key, filename in files.items():
        print(f"Downloading {filename}...")
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=token,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        paths[key] = path
        print(f"Successfully downloaded {filename}")
    
    return paths

def _load_projector(checkpoint_path, config_path, device='cuda', dtype=torch.float16):
    """Internal function to load the projector model"""
    print("Loading projector checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model = SimplifiedSigLIPProjector(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device=device, dtype=dtype)
    
    return model

def load_visualization(
    visualizer_repo_id: str = "MetaMorphOrg/Visualizer",
    sd_model_path: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
    local_dir: str = "./model_files",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    token: str = None
):
    """
    Load the visualization models (Stable Diffusion and Projector)
    
    Args:
        visualizer_repo_id (str): HuggingFace repo ID for the visualizer
        sd_model_path (str): Path to the Stable Diffusion model
        local_dir (str): Directory to store downloaded files
        device (str): Device to load models on ('cuda' or 'cpu')
        dtype (torch.dtype): Data type for models
        token (str, optional): HuggingFace token for private repos
    
    Returns:
        tuple: (pipeline, projector) - The loaded StableDiffusion pipeline and projector model
    """
    # Set device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
        if dtype == torch.float16:
            dtype = torch.float32
    
    # Download checkpoints
    paths = _download_checkpoints(visualizer_repo_id, local_dir, token)
    
    # Load Stable Diffusion pipeline
    print(f"Loading Stable Diffusion from {sd_model_path}...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        sd_model_path,
        torch_dtype=dtype
    ).to(device)
    
    # Disable safety checker
    pipeline.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
    
    # Load UNet into pipeline
    print("Loading fine-tuned UNet...")
    unet_state_dict = torch.load(paths["unet"], map_location=device)
    pipeline.unet.load_state_dict(unet_state_dict["model_state_dict"])
    
    # Load projector
    projector = _load_projector(paths["adapter"], paths["config"], device, dtype)
    
    print("All models loaded successfully!")
    return pipeline, projector

# Only run this if the script is run directly
if __name__ == "__main__":
    pipeline, projector = load_visualization()