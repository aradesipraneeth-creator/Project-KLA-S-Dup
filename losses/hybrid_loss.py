import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Union, Any, Tuple

from utils.edge_utils import compute_sobel_edges

try:
    from pytorch_msssim import ssim as ssim_fn
    HAS_PYTORCH_MSSSIM = True
except ImportError:
    HAS_PYTORCH_MSSSIM = False


class HighFrequencyLoss(nn.Module):
    """
    High-Frequency Loss for AIR-Net v2:
        HF(x) = x - GaussianBlur(x, kernel_size=5, sigma=1.0)
        Loss = mean(|HF(pred) - HF(target)|)
    Operates strictly in Float32 to prevent HalfTensor/FloatTensor AMP errors.
    """
    def __init__(self, kernel_size: int = 5, sigma: float = 1.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.register_buffer("kernel", self._create_gaussian_kernel(kernel_size, sigma))

    def _create_gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
        g1d = torch.exp(-coords**2 / (2 * sigma**2))
        g1d = g1d / g1d.sum()
        g2d = g1d.unsqueeze(1) @ g1d.unsqueeze(0)
        return g2d.view(1, 1, kernel_size, kernel_size)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p_f = pred.float()
        t_f = target.float()
        kernel = self.kernel.to(device=p_f.device, dtype=torch.float32)
        
        blur_p = F.conv2d(p_f, kernel, padding=self.kernel_size // 2)
        blur_t = F.conv2d(t_f, kernel, padding=self.kernel_size // 2)
        
        hf_p = p_f - blur_p
        hf_t = t_f - blur_t
        
        return F.l1_loss(hf_p, hf_t)


class AIRNetV2Loss(nn.Module):
    """
    Multi-Objective AIR-Net v2 Loss Function:
        Total Loss = 0.70 * L1 + 0.20 * (1.0 - SSIM) + 0.05 * EdgeLoss + 0.05 * HFLoss
    
    Prioritizes Pixel Fidelity (PSNR >= 25 dB) while enforcing structural & high-frequency detail.
    """
    def __init__(
        self,
        l1_weight: float = 0.70,
        ssim_weight: float = 0.20,
        edge_weight: float = 0.05,
        hf_weight: float = 0.05,
        data_range: float = 1.0
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.edge_weight = edge_weight
        self.hf_weight = hf_weight
        self.data_range = data_range

        self.l1_loss = nn.L1Loss()
        self.hf_loss_fn = HighFrequencyLoss(kernel_size=5, sigma=1.0)

    def forward(
        self,
        pred: Union[torch.Tensor, Dict[str, Any]],
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if isinstance(pred, dict):
            restored_pred = pred["restored"]
            edge_pred = pred.get("edge", None)
        else:
            restored_pred = pred
            edge_pred = None

        p_f = restored_pred.float()
        t_f = target.float()

        # 1. Pixel Fidelity Loss (Float32)
        loss_l1 = self.l1_loss(p_f, t_f)

        # 2. Structural Fidelity Loss (Float32)
        if HAS_PYTORCH_MSSSIM:
            ssim_val = ssim_fn(p_f, t_f, data_range=self.data_range, size_average=True)
        else:
            calc = FallbackSSIM(window_size=11, channel=1, data_range=self.data_range)
            ssim_val = calc(p_f, t_f)
        loss_ssim = 1.0 - ssim_val

        # 3. Edge Reconstruction Loss (Float32)
        if edge_pred is not None:
            gt_edges = compute_sobel_edges(t_f)
            loss_edge = self.l1_loss(edge_pred.float(), gt_edges.float())
        else:
            loss_edge = torch.tensor(0.0, device=target.device)

        # 4. High-Frequency Detail Loss (Float32)
        loss_hf = self.hf_loss_fn(p_f, t_f)

        total_loss = (
            self.l1_weight * loss_l1 +
            self.ssim_weight * loss_ssim +
            self.edge_weight * loss_edge +
            self.hf_weight * loss_hf
        )
        return total_loss, {
            "l1": loss_l1.item(),
            "ssim_loss": loss_ssim.item(),
            "edge": loss_edge.item(),
            "hf": loss_hf.item()
        }


class AIRNetV12Loss(nn.Module):
    """
    Controlled AIR-Net v1.2 Loss Function:
        Total Loss = 0.50 * L1 + 0.20 * (1.0 - SSIM) + 0.15 * EdgeLoss + 0.15 * HFLoss
    """
    def __init__(
        self,
        l1_weight: float = 0.50,
        ssim_weight: float = 0.20,
        edge_weight: float = 0.15,
        hf_weight: float = 0.15,
        data_range: float = 1.0
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.edge_weight = edge_weight
        self.hf_weight = hf_weight
        self.data_range = data_range

        self.l1_loss = nn.L1Loss()
        self.hf_loss_fn = HighFrequencyLoss(kernel_size=5, sigma=1.0)

    def forward(
        self,
        pred: Union[torch.Tensor, Dict[str, Any]],
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if isinstance(pred, dict):
            restored_pred = pred["restored"]
            edge_pred = pred.get("edge", None)
        else:
            restored_pred = pred
            edge_pred = None

        p_f = restored_pred.float()
        t_f = target.float()

        loss_l1 = self.l1_loss(p_f, t_f)

        if HAS_PYTORCH_MSSSIM:
            ssim_val = ssim_fn(p_f, t_f, data_range=self.data_range, size_average=True)
        else:
            calc = FallbackSSIM(window_size=11, channel=1, data_range=self.data_range)
            ssim_val = calc(p_f, t_f)
        loss_ssim = 1.0 - ssim_val

        if edge_pred is not None:
            gt_edges = compute_sobel_edges(t_f)
            loss_edge = self.l1_loss(edge_pred.float(), gt_edges.float())
        else:
            loss_edge = torch.tensor(0.0, device=target.device)

        loss_hf = self.hf_loss_fn(p_f, t_f)

        total_loss = (
            self.l1_weight * loss_l1 +
            self.ssim_weight * loss_ssim +
            self.edge_weight * loss_edge +
            self.hf_weight * loss_hf
        )
        return total_loss, {
            "l1": loss_l1.item(),
            "ssim_loss": loss_ssim.item(),
            "edge": loss_edge.item(),
            "hf": loss_hf.item()
        }


class AIRNetHybridLoss(nn.Module):
    """
    Hybrid Loss function for AIR-Net v1:
        Loss = 0.60 * L1 + 0.25 * (1.0 - SSIM) + 0.15 * EdgeLoss
    """
    def __init__(
        self,
        l1_weight: float = 0.60,
        ssim_weight: float = 0.25,
        edge_weight: float = 0.15,
        data_range: float = 1.0
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.edge_weight = edge_weight
        self.data_range = data_range
        self.l1_loss = nn.L1Loss()

    def forward(
        self,
        pred: Union[torch.Tensor, Dict[str, Any]],
        target: torch.Tensor
    ) -> torch.Tensor:
        if isinstance(pred, dict):
            restored_pred = pred["restored"]
            edge_pred = pred.get("edge", None)
        else:
            restored_pred = pred
            edge_pred = None

        p_f = restored_pred.float()
        t_f = target.float()

        loss_l1 = self.l1_loss(p_f, t_f)

        if HAS_PYTORCH_MSSSIM:
            ssim_val = ssim_fn(p_f, t_f, data_range=self.data_range, size_average=True)
        else:
            calc = FallbackSSIM(window_size=11, channel=1, data_range=self.data_range)
            ssim_val = calc(p_f, t_f)
            
        loss_ssim = 1.0 - ssim_val

        if edge_pred is not None:
            gt_edges = compute_sobel_edges(t_f)
            loss_edge = self.l1_loss(edge_pred.float(), gt_edges.float())
        else:
            loss_edge = torch.tensor(0.0, device=target.device)

        total_loss = (
            self.l1_weight * loss_l1 +
            self.ssim_weight * loss_ssim +
            self.edge_weight * loss_edge
        )
        return total_loss


class FallbackSSIM(nn.Module):
    def __init__(self, window_size: int = 11, channel: int = 1, data_range: float = 1.0):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.data_range = data_range
        self.register_buffer("window", self.create_window(window_size, channel))

    def gaussian(self, window_size: int, sigma: float):
        gauss = torch.exp(torch.tensor([-(x - window_size // 2) ** 2 / float(2 * sigma ** 2) for x in range(window_size)]))
        return gauss / gauss.sum()

    def create_window(self, window_size: int, channel: int):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        return _2D_window.expand(channel, 1, window_size, window_size).contiguous()

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        c1 = (0.01 * self.data_range) ** 2
        c2 = (0.03 * self.data_range) ** 2

        img1_f = img1.float()
        img2_f = img2.float()

        window = self.window.to(device=img1_f.device, dtype=torch.float32)

        mu1 = F.conv2d(img1_f, window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2_f, window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1_f * img1_f, window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2_f * img2_f, window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1_f * img2_f, window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        return ssim_map.mean().to(img1.dtype)


HybridLoss = AIRNetHybridLoss
