"""
	"LiftFeat: 3D Geometry-Aware Local Feature Matching"
"""


import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).absolute().parents[1]
ALIKE_PATH = os.path.abspath(os.environ.get(
    'ALIKE_PATH',
    str(PROJECT_ROOT / 'third_repos' / 'ALIKE')
))
if ALIKE_PATH not in sys.path:
    sys.path.insert(0, ALIKE_PATH)

import torch
import torch.nn as nn
from alike import ALike
import cv2
import numpy as np



configs = {
    'alike-t': {'c1': 8, 'c2': 16, 'c3': 32, 'c4': 64, 'dim': 64, 'single_head': True, 'radius': 2,
                'model_path': os.path.join(ALIKE_PATH, 'models', 'alike-t.pth')},
    'alike-s': {'c1': 8, 'c2': 16, 'c3': 48, 'c4': 96, 'dim': 96, 'single_head': True, 'radius': 2,
                'model_path': os.path.join(ALIKE_PATH, 'models', 'alike-s.pth')},
    'alike-n': {'c1': 16, 'c2': 32, 'c3': 64, 'c4': 128, 'dim': 128, 'single_head': True, 'radius': 2,
                'model_path': os.path.join(ALIKE_PATH, 'models', 'alike-n.pth')},
    'alike-l': {'c1': 32, 'c2': 64, 'c3': 128, 'c4': 128, 'dim': 128, 'single_head': False, 'radius': 2,
                'model_path': os.path.join(ALIKE_PATH, 'models', 'alike-l.pth')},
}


class ALikeExtractor(nn.Module):
    def __init__(self, model_type, device) -> None:
        super().__init__()
        self.device = device
        self.net = ALike(**configs[model_type], device=device, top_k=4096, scores_th=0.1, n_limit=8000)

    @torch.inference_mode()
    def extract_alike_kpts(self, img):
        pred0 = self.net(img, sub_pixel=True)
        return pred0['keypoints']

    @torch.inference_mode()
    def extract_alike_kpts_from_tensor(self, image: torch.Tensor, sub_pixel: bool = True):
        """
        Extracts keypoints directly from a PyTorch tensor on GPU without CPU/numpy roundtrips.
        Args:
            image: Tensor of shape (B, 1, H, W) or (B, 3, H, W) or (3, H, W), float in [0.0, 1.0].
        Returns:
            list of B Tensors, each of shape (N_i, 2) on the same GPU device.
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if image.shape[1] == 1:
            image = image.expand(-1, 3, -1, -1)
        if not image.is_contiguous():
            image = image.contiguous()

        B, _, H, W = image.shape
        descriptor_map, scores_map = self.net.extract_dense_map(image)
        keypoints, _, _, _ = self.net.dkd(scores_map, descriptor_map, sub_pixel=sub_pixel)

        scale = image.new_tensor([[W - 1, H - 1]])
        kpts_scaled = [((kpt + 1) / 2) * scale for kpt in keypoints]
        return kpts_scaled


