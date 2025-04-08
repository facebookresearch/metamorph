# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
from .siglip_encoder import SiglipVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))

    return SiglipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
