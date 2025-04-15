# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from metamorph.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from metamorph.conversation import conv_templates
from metamorph.model.builder import load_pretrained_model
from metamorph.utils import disable_torch_init
from metamorph.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
import torch
import os

def load_metamorph_model(
    model_path: str = None,
    model_base: str = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16
):
    """
    Load the MetaMorph model using the original metamorph utilities
    
    Args:
        model_path: Path or HF repo ID for MetaMorph
        model_base: Base model path if needed
        device: Device to load model on ('cuda' or 'cpu')
        dtype: Data type for model weights
    
    Returns:
        tuple: (tokenizer, model, image_processor, context_len)
    """
    # Disable torch init as done in original code
    disable_torch_init()

    # Get model name from path
    model_name = get_model_name_from_path(model_path)
    print(f"Loading MetaMorph model: {model_name}")
    
    # Set device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
        if dtype == torch.float16:
            dtype = torch.float32
    
    # Load the model using metamorph's loader
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=model_base,
        model_name=model_name
    )
    
    print(f"Model loaded successfully with context length: {context_len}")
    return tokenizer, model, image_processor, context_len

if __name__ == "__main__":
    # Example usage

    tokenizer, model, image_processor, context_len = load_metamorph_model()
    
    print(f"Model info:")
    print(f"- Model name: {get_model_name_from_path('PATH-TO-MetaMorph Model')}")
    print(f"- Context length: {context_len}")
    print(f"- Model device: {model.device}")