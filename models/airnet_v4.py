import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple

from models.airnet_v3 import AIRNetV3
from models.image_indexer import ImageIndexer

class SobelConv2d(nn.Module):
    """Deterministic 2D Sobel gradient operator for feature guidance."""
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        return torch.sqrt(gx**2 + gy**2 + 1e-8)

class GatedResidualBlock(nn.Module):
    """
    Lightweight Gated Residual Block for fine-grained structural refinement.
    Uses depthwise-separable convolutions and spatial feature gating.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.norm = nn.GroupNorm(4, channels)
        self.pw_conv1 = nn.Conv2d(channels, channels * 2, kernel_size=1)
        self.pw_conv2 = nn.Conv2d(channels, channels, kernel_size=1)
        self.sobel = SobelConv2d()
        self.sobel_proj = nn.Conv2d(1, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.dw_conv(x)
        out = self.norm(out)
        
        # Spatial Feature Gating
        gate_feats = self.pw_conv1(out)
        f1, f2 = torch.chunk(gate_feats, 2, dim=1)
        out = f1 * torch.sigmoid(f2)
        out = self.pw_conv2(out)

        # Sobel Gradient Edge Guidance
        sobel_map = self.sobel(torch.mean(residual, dim=1, keepdim=True))
        sobel_guidance = torch.tanh(self.sobel_proj(sobel_map))

        return residual + out + 0.1 * sobel_guidance

class ResidualRefinementModule(nn.Module):
    """
    Multi-Scale Residual High-Fidelity Refinement Module.
    Refines 256x256 AIR-Net v3 predictions to produce faithful final outputs.
    Output = V3_Prediction + residual_scale * Residual_Correction
    """
    def __init__(self, in_channels: int = 1, base_dim: int = 32, num_blocks: int = 4, residual_scale: float = 0.2):
        super().__init__()
        self.residual_scale = residual_scale
        
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.blocks = nn.ModuleList([
            GatedResidualBlock(base_dim) for _ in range(num_num_blocks if 'num_num_blocks' in locals() else num_blocks)
        ])
        
        self.tail = nn.Sequential(
            nn.Conv2d(base_dim, base_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_dim, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x_v3: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_v3: (B, 1, 256, 256) - AIR-Net v3 restored prediction
        Returns: (refined_output, residual_correction)
        """
        feat = self.head(x_v3)
        for block in self.blocks:
            feat = block(feat)
        residual = self.tail(feat)
        
        refined = x_v3 + self.residual_scale * residual
        return refined, residual

class AIRNetV4(nn.Module):
    """
    AIR-Net v4: High-Fidelity Content-Adaptive Semiconductor Restoration System (128x128 -> 256x256).
    Preserves AIR-Net v3 base + adds Residual High-Fidelity Refinement Module.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        dim: int = 32,
        channels: List[int] = [32, 64, 128, 192],
        heads: List[int] = [1, 2, 4, 6],
        enc_blocks: List[int] = [2, 2, 4],
        latent_blocks: int = 8,
        dec_blocks: List[int] = [4, 2, 2],
        ffn_expansion_factor: float = 2.66,
        norm_params: Optional[Dict] = None,
        use_residual_learning: bool = True,
        refinement_blocks: int = 4,
        residual_scale: float = 0.2
    ):
        super().__init__()
        # Base Content-Adaptive AIR-Net v3
        self.v3_base = AIRNetV3(
            in_channels=in_channels,
            out_channels=out_channels,
            dim=dim,
            channels=channels,
            heads=heads,
            enc_blocks=enc_blocks,
            latent_blocks=latent_blocks,
            dec_blocks=dec_blocks,
            ffn_expansion_factor=ffn_expansion_factor,
            norm_params=norm_params,
            use_residual_learning=use_residual_learning
        )
        
        # AIR-Net v4 High-Fidelity Refinement Module
        self.refinement_module = ResidualRefinementModule(
            in_channels=out_channels,
            base_dim=dim,
            num_blocks=refinement_blocks,
            residual_scale=residual_scale
        )
        
        self.indexer = self.v3_base.indexer

    def freeze_v3_base(self):
        """Freezes AIR-Net v3 base parameters for Stage 1 v4 refinement training."""
        for param in self.v3_base.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        """Unfreezes all parameters for joint fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, lr_img: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        lr_img: (B, 1, 128, 128)
        Returns dictionary containing:
          - restored: (B, 1, 256, 256) Refined AIR-Net v4 output
          - v3_restored: (B, 1, 256, 256) Intermediate v3 prediction
          - residual: (B, 1, 256, 256) Refinement residual correction
          - routing_probs: (B, 5) Soft MoE category routing probabilities
        """
        v3_output_dict = self.v3_base(lr_img)
        v3_pred = v3_output_dict["restored"]
        routing_probs = v3_output_dict["routing_probs"]

        refined_pred, residual_corr = self.refinement_module(v3_pred)

        return {
            "restored": refined_pred,
            "v3_restored": v3_pred,
            "residual": residual_corr,
            "routing_probs": routing_probs
        }
