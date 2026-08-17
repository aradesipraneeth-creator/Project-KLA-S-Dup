import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.airnet_v3 import AIRNetV3
from models.airnet_v4 import AIRNetV4
from utils.image_normalization import normalize_for_display_and_metrics

class SpatialRefinementBranch(nn.Module):
    """Multi-scale spatial residual convolution branch."""
    def __init__(self, in_channels: int = 16, num_blocks: int = 3):
        super().__init__()
        layers = []
        for _ in range(num_blocks):
            layers.extend([
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ])
        self.branch = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.branch(x)

class EdgeAwareBranch(nn.Module):
    """Sobel-guided edge enhancement branch for crisp boundary recovery."""
    def __init__(self, in_channels: int = 16):
        super().__init__()
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x_img: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(x_img, self.sobel_x, padding=1)
        gy = F.conv2d(x_img, self.sobel_y, padding=1)
        mag = torch.sqrt(gx**2 + gy**2 + 1e-8)
        edge_mask = self.edge_conv(mag)
        return feat * (1.0 + edge_mask)

class FrequencyAwareBranch(nn.Module):
    """High-frequency decomposition branch for micro-texture detail recovery."""
    def __init__(self, in_channels: int = 16):
        super().__init__()
        self.hf_conv = nn.Sequential(
            nn.Conv2d(1, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x_img: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        blurred = F.avg_pool2d(x_img, kernel_size=5, stride=1, padding=2)
        hf_residual = x_img - blurred
        hf_feat = self.hf_conv(hf_residual)
        return feat + hf_feat

class IntensityContrastBranch(nn.Module):
    """Global & local intensity distribution alignment branch."""
    def __init__(self, in_channels: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        weights = self.mlp(feat)
        return feat * weights

class BoundedResidualModule(nn.Module):
    """
    Learns a bounded residual refinement map with numerical stability.
    V5 = clamp(Foundation_Output + sigmoid(alpha) * 0.5 * residual, 0.0, 1.0)
    """
    def __init__(self, in_channels: int = 16, initial_alpha: float = 0.1):
        super().__init__()
        self.res_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=3, padding=1),
            nn.Tanh()
        )
        self.alpha_param = nn.Parameter(torch.tensor([np.log(initial_alpha / (1.0 - initial_alpha))], dtype=torch.float32))

    def forward(self, foundation_out: torch.Tensor, feat: torch.Tensor) -> tuple:
        raw_res = self.res_conv(feat)
        alpha = torch.sigmoid(self.alpha_param) * 0.5
        v5_out = torch.clamp(foundation_out + alpha * raw_res, 0.0, 1.0)
        return v5_out, raw_res, alpha

class AIRNetV5(nn.Module):
    """
    AIR-Net v5 High-Fidelity Refinement Semiconductor Image Restoration System.
    Combines trained AIR-Net v4 foundation model (or v3 base) with specialized multi-branch refinement architecture.
    Inputs: 128x128 NoisyLR -> V4 (256x256) -> V5 Multi-Branch Refinement -> 256x256 V5 Output.
    """
    def __init__(self, foundation_mode: str = "v4", norm_params: dict = None, num_channels: int = 16):
        super().__init__()
        self.foundation_mode = foundation_mode
        if foundation_mode == "v4":
            self.foundation_base = AIRNetV4(norm_params=norm_params)
        else:
            self.foundation_base = AIRNetV3(norm_params=norm_params)

        # Alias for backward compatibility
        self.v4_base = self.foundation_base
        self.v3_base = self.foundation_base.v3_base if hasattr(self.foundation_base, "v3_base") else self.foundation_base

        # Feature encoder for 256x256 foundation output
        self.encoder = nn.Sequential(
            nn.Conv2d(1, num_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Specialized Branch Modules
        self.spatial_branch = SpatialRefinementBranch(in_channels=num_channels)
        self.edge_branch = EdgeAwareBranch(in_channels=num_channels)
        self.freq_branch = FrequencyAwareBranch(in_channels=num_channels)
        self.intensity_branch = IntensityContrastBranch(in_channels=num_channels)

        # Multi-scale Feature Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(num_channels * 4, num_channels, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Bounded Residual Refinement Head
        self.refinement_head = BoundedResidualModule(in_channels=num_channels)

    def freeze_foundation(self):
        """Freezes foundation base parameters (V4 or V3) during initial V5 training."""
        for p in self.foundation_base.parameters():
            p.requires_grad = False

    def freeze_v3_base(self):
        """Alias for freeze_foundation."""
        self.freeze_foundation()

    def unfreeze_foundation(self):
        """Unfreezes foundation base parameters for joint fine-tuning."""
        for p in self.foundation_base.parameters():
            p.requires_grad = True

    def forward(self, lr_input: torch.Tensor) -> dict:
        assert lr_input.shape[-2:] == (128, 128), f"Input resolution mismatch: expected (128, 128), got {lr_input.shape[-2:]}"

        # 1. Base Foundation Forward Pass
        base_out_dict = self.foundation_base(lr_input)
        foundation_res = base_out_dict["restored"]  # (B, 1, 256, 256)
        routing_probs = base_out_dict["routing_probs"]

        assert foundation_res.shape[-2:] == (256, 256), f"Foundation output shape mismatch: {foundation_res.shape}"

        # 2. V5 Multi-Branch Feature Extraction
        feat_base = self.encoder(foundation_res)
        f_spatial = self.spatial_branch(feat_base)
        f_edge = self.edge_branch(foundation_res, feat_base)
        f_freq = self.freq_branch(foundation_res, feat_base)
        f_intensity = self.intensity_branch(feat_base)

        # 3. Feature Fusion
        f_cat = torch.cat([f_spatial, f_edge, f_freq, f_intensity], dim=1)
        f_fused = self.fusion(f_cat)

        # 4. Bounded Residual Refinement
        v5_res, raw_residual, alpha = self.refinement_head(foundation_res, f_fused)

        assert v5_res.shape[-2:] == (256, 256), f"V5 output shape mismatch: {v5_res.shape}"

        return {
            "restored": v5_res,
            "foundation_restored": foundation_res,
            "v3_restored": foundation_res,
            "residual": raw_residual,
            "alpha": alpha,
            "routing_probs": routing_probs
        }
