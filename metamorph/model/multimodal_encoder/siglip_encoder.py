# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from transformers import AutoProcessor, AutoModel


class ProcessorWrapper:
    def __init__(self, transform, height=378, width=378, image_mean = [0.48145466, 0.4578275, 0.40821073]):
        self._crop_size = {
            "height": height,
            "width": width,
        }
        self._transforms = transform
        self.image_mean = image_mean

    @property
    def crop_size(self):
        return self._crop_size

    def preprocess(self, image, return_tensors='pt'):
        # Ensure image is a PIL Image
        output = {}
        output['pixel_values'] = [self._transforms(image)]
        return output

def extract_res_interp(model_name):
    valid_model_prefixes = {
        "siglip/CLIP-ViT-SO400M-14-384":"hf-hub:timm/ViT-SO400M-14-SigLIP-384",
        "timm/ViT-SO400M-14-SigLIP-384":"hf-hub:timm/ViT-SO400M-14-SigLIP-384",
        "siglip/CLIP-ViT-SO400M-14":"hf-hub:timm/ViT-SO400M-14-SigLIP",
        "timm/ViT-SO400M-14-SigLIP":"hf-hub:timm/ViT-SO400M-14-SigLIP"
    }

    res = 384 if '384' in model_name else 224
    interp = None

    for prefix in valid_model_prefixes:
        if model_name.startswith(prefix):
            base_model_name = valid_model_prefixes[prefix]
            break
    else:
        raise ValueError(f"Unknown vision tower: {model_name}")

    parts = model_name.split("-")
    for part in parts:
        if part.startswith("res"):
            res = int(part[3:])
        elif part.startswith("interp"):
            interp = int(part[6:])

    return base_model_name, res, interp


class SiglipVisionTower(nn.Module):
    def __init__(self, vision_tower_name, args, delay_load=False):
        super().__init__()
        base_model_name, res, interp = extract_res_interp(vision_tower_name)
        self.is_loaded = False

        self.select_layer =  getattr(args, 'mm_vision_select_layer', -2)
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.image_token_reduction = getattr(args, 'image_token_reduction', 'none')
        self.image_token_len = getattr(args, 'num_image_tokens', 256)
        self.freeze_vision = getattr(args, 'freeze_vision', False)
        self.vision_coef = getattr(args, 'vision_coef', 1.0)
        self.normalize_vision = getattr(args, 'normalize_vision', False)
        self.apply_softmax = getattr(args, 'apply_softmax', False)


        self.codebook = 4096


        


        self.vision_tower_name = base_model_name
        self._image_size = res if res is not None else 512
        self._interp_size = interp
        if not delay_load:
            self.load_model()
        else:
            self.hidden_size = 1152

        if self.is_loaded:
            # Feature size
            feature_size = 1152

            # Number of patches
            num_patches = 729


            if self.image_token_reduction == "mlpmixer":
                self.token_mixer = nn.Sequential(
                    nn.Linear(num_patches,  self.image_token_len),
                )
                self.channel_mixer = nn.Sequential(
                    nn.Linear(feature_size, feature_size),
                )
            elif self.image_token_reduction =="concat_interpolation":
                 self.hidden_size = 4* self.hidden_size

    def load_model(self, device_map=None):
        self.vision_model = "siglip"

        model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
        clip_processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")

        
        self.image_processor = clip_processor.image_processor
        self.image_processor.crop_size = {
            "height": 384,
            "width": 384,
        }
        self.vision_tower = model.vision_model
    
        # self.vision_tower = clip_model.visual.trunk
        # self.vision_tower.output_tokens = True

        self.hidden_size = 1152

        # self.image_size = self.vision_tower.patch_embed.img_size[0]
        # self.patch_size = 729
        # self.image_processor = ProcessorWrapper(processor, height=self.image_size, width=self.image_size)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    def forward(self, images):
        with torch.set_grad_enabled(not self.freeze_vision):

            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

            b, num_tokens, dim = image_features.shape

            if num_tokens != self.image_token_len:
                if self.image_token_len == -1:
                    image_features = torch.zeros((b, num_tokens, dim), device=image_features.device, dtype=image_features.dtype)
                    return image_features

                if self.image_token_reduction == "interpolation":
                    if self.image_token_len == 0:
                        # Randomly sample a number from 1 to 24
                        random_sample = np.random.randint(1, 25)
                        target_h = target_w = random_sample
                    else:
                        target_h = target_w = int(np.sqrt(self.image_token_len))
                    
                    h = w = int(np.sqrt(num_tokens))
                    image_features = image_features.view(b, h, w, dim)
                    image_features = image_features.permute(0, 3, 1, 2).contiguous()
                    image_features = F.interpolate(image_features.to(torch.float32), size=(target_h, target_w), mode='bilinear', align_corners=False).to(image_features.dtype)
                    image_features = image_features.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
                elif self.image_token_reduction == "mlpmixer":
                    image_features = image_features.transpose(1, 2)  # shape: (b, dim, num_tokens)
                    image_features = self.token_mixer(image_features)  # shape: (b, dim, target_token_len)
                    image_features = image_features.transpose(1, 2)  # shape: (b, target_token_len, dim)
                    image_features = self.channel_mixer(image_features)  # shape: (b, targe
                elif self.image_token_reduction == "concat_interpolation":
                    h = w = int(np.sqrt(image_features.size(1)))
                    b, num_tokens, dim = image_features.size()

                    # Calculate the intermediate size needed for final target token length
                    stride = kernel_size = 2
                    intermediate_token_len = self.image_token_len * (kernel_size ** 2)
                    intermediate_h = intermediate_w = int(np.sqrt(intermediate_token_len))


                    image_features = image_features.view(b, h, w, dim)
                    image_features = image_features.permute(0, 3, 1, 2).contiguous()
                    image_features = F.interpolate(image_features.to(torch.float32), size=(intermediate_h, intermediate_w), mode='bilinear', align_corners=False).to(image_features.dtype)

                    # Reshape to 4D tensor
                    image_features = image_features.permute(0, 2, 3, 1).contiguous()

                    # Initialize the target tensor
                    target_h = target_w = int(np.sqrt(self.image_token_len))
                    target_features = torch.zeros(b, target_h, target_w, dim * (kernel_size ** 2), dtype=image_features.dtype, device=image_features.device)

                    # Fill the target tensor with concatenated features
                    
                    for i in range(0, intermediate_h, stride):
                        for j in range(0, intermediate_w, stride):
                            sub_tensor = image_features[:, i:i+stride, j:j+stride, :].contiguous().view(b, 1, -1)  # Make sure sub_tensor has shape [b, 1, dim * 4]
                            target_i = i // stride
                            target_j = j // stride
                            target_features[:, target_i, target_j, :] = sub_tensor.view(b, dim * (stride ** 2))

                    # Permute and flatten to match the expected output shape
                    image_features = target_features.permute(0, 3, 1, 2).contiguous()
                    image_features = image_features.permute(0, 2, 3, 1).contiguous().flatten(1, 2)

                else:
                    raise NotImplementedError("Not Implemented!")

            if self.normalize_vision:

                image_features = F.normalize(image_features, p=2, dim=-1)

            if self.apply_softmax:
                image_features = F.softmax(image_features / 0.07, dim=-1)

            return image_features
    @property
    def dtype(self):
        # Dynamically infer the dtype from the first parameter, if not explicitly specified
        if hasattr(self.vision_tower, 'dtype'):
            return self.vision_tower.dtype
        else:
            params = list(self.vision_tower.parameters())
            return params[0].dtype if len(params) > 0 else torch.float32  # Default to torch.float32 if no parameters

    @property
    def device(self):
        # Dynamically infer the device from the first parameter, if not explicitly specified
        if hasattr(self.vision_tower, 'device'):
            return self.vision_tower.device
        else:
            params = list(self.vision_tower.parameters())
            return params[0].device if len(params) > 0 else torch.device("cpu")  # Default to CPU if no parameters

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only