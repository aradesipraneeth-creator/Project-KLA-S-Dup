import torch
import torch.nn as nn
import torch.nn.functional as F

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1 smooth variant) for robust pixel reconstruction."""
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))

class SSIMLoss(nn.Module):
    """Structural Similarity Index Measure (SSIM) Loss."""
    def __init__(self, window_size: int = 11, data_range: float = 1.0):
        super().__init__()
        self.window_size = window_size
        self.data_range = data_range
        self.register_buffer("window", self._create_window(window_size))

    def _create_window(self, size: int) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
        g = torch.exp(-coords**2 / (2 * 1.5**2))
        g = g / g.sum()
        g2d = g.unsqueeze(1) @ g.unsqueeze(0)
        return g2d.view(1, 1, size, size)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        C1 = (0.01 * self.data_range) ** 2
        C2 = (0.03 * self.data_range) ** 2
        w = self.window.to(x.device)

        mu_x = F.conv2d(x, w, padding=self.window_size // 2)
        mu_y = F.conv2d(y, w, padding=self.window_size // 2)

        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y

        sigma_x_sq = F.conv2d(x * x, w, padding=self.window_size // 2) - mu_x_sq
        sigma_y_sq = F.conv2d(y * y, w, padding=self.window_size // 2) - mu_y_sq
        sigma_xy = F.conv2d(x * y, w, padding=self.window_size // 2) - mu_xy

        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
        return 1.0 - torch.mean(ssim_map)

class SobelEdgeLoss(nn.Module):
    """Sobel Edge Difference Loss for precise structural boundary fidelity."""
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        gx_x = F.conv2d(x, self.sobel_x, padding=1)
        gy_x = F.conv2d(x, self.sobel_y, padding=1)
        mag_x = torch.sqrt(gx_x**2 + gy_x**2 + 1e-8)

        gx_y = F.conv2d(y, self.sobel_x, padding=1)
        gy_y = F.conv2d(y, self.sobel_y, padding=1)
        mag_y = torch.sqrt(gx_y**2 + gy_y**2 + 1e-8)

        return F.l1_loss(mag_x, mag_y)

class FFTFrequencyLoss(nn.Module):
    """2D Real Fast Fourier Transform (rFFT2) spectral magnitude difference loss."""
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        fft_x = torch.fft.rfft2(x, norm="backward")
        fft_y = torch.fft.rfft2(y, norm="backward")
        mag_x = torch.abs(fft_x)
        mag_y = torch.abs(fft_y)
        return F.l1_loss(mag_x, mag_y)

class IntensityPercentileLoss(nn.Module):
    """
    Global & local intensity distribution alignment loss.
    Penalizes mean mismatch, std mismatch, and quantile mismatch (P5, P25, P50, P75, P95).
    Forces model to learn GT intensity distribution during training.
    """
    def __init__(self, percentiles: list = [0.05, 0.25, 0.50, 0.75, 0.95]):
        super().__init__()
        self.percentiles = percentiles

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Mean and Std matching penalty
        mean_loss = torch.abs(torch.mean(x) - torch.mean(y))
        std_loss = torch.abs(torch.std(x) - torch.std(y))

        # Percentile matching penalty
        q_x = torch.quantile(x.view(x.size(0), -1), torch.tensor(self.percentiles, device=x.device), dim=1)
        q_y = torch.quantile(y.view(y.size(0), -1), torch.tensor(self.percentiles, device=y.device), dim=1)
        quantile_loss = torch.mean(torch.abs(q_x - q_y))

        return mean_loss + std_loss + quantile_loss

class AIRNetV5AdaptiveLoss(nn.Module):
    """
    Content-Adaptive Multi-Objective Loss for AIR-Net v5.
    Combines Pixel, SSIM Structural, Sobel Edge, FFT Frequency, and Intensity Percentile losses.
    Target Weights:
      - Pixel (Charbonnier): 0.45
      - Structural (SSIM): 0.20
      - Edge (Sobel): 0.15
      - Frequency (FFT): 0.10
      - Intensity/Percentile: 0.10
    """
    def __init__(self, data_range: float = 1.0):
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        self.ssim_loss = SSIMLoss(data_range=data_range)
        self.edge_loss = SobelEdgeLoss()
        self.fft_loss = FFTFrequencyLoss()
        self.intensity_loss = IntensityPercentileLoss()

        # Category Loss Weight Matrix (5 Categories x 6 Components)
        base_table = [
            [0.45, 0.20, 0.20, 0.08, 0.05, 0.02],  # EDGE_DOMINANT
            [0.35, 0.20, 0.15, 0.15, 0.13, 0.02],  # TEXTURE_DOMINANT
            [0.55, 0.20, 0.05, 0.05, 0.13, 0.02],  # NOISE_DOMINANT
            [0.50, 0.25, 0.05, 0.05, 0.13, 0.02],  # SMOOTH_LOW_CONTRAST
            [0.45, 0.20, 0.15, 0.10, 0.08, 0.02]   # SPARSE_FEATURE
        ]
        self.register_buffer("weight_matrix", torch.tensor(base_table, dtype=torch.float32))

    def forward(self, output_dict: dict, target: torch.Tensor) -> tuple:
        pred = output_dict["restored"]
        residual = output_dict["residual"]
        routing_probs = output_dict["routing_probs"]

        l_char = self.charbonnier(pred, target)
        l_ssim = self.ssim_loss(pred, target)
        l_edge = self.edge_loss(pred, target)
        l_freq = self.fft_loss(pred, target)
        l_int = self.intensity_loss(pred, target)
        l_res = torch.mean(residual ** 2)

        weights = torch.matmul(routing_probs, self.weight_matrix)  # (B, 6)
        w_mean = torch.mean(weights, dim=0)

        w_char, w_ssim, w_edge, w_freq, w_int, w_res = w_mean[0], w_mean[1], w_mean[2], w_mean[3], w_mean[4], w_mean[5]

        total_loss = (
            w_char * l_char +
            w_ssim * l_ssim +
            w_edge * l_edge +
            w_freq * l_freq +
            w_int * l_int +
            w_res * l_res
        )

        loss_dict = {
            "total_loss": total_loss.item(),
            "charbonnier": l_char.item(),
            "ssim": l_ssim.item(),
            "edge": l_edge.item(),
            "frequency": l_freq.item(),
            "intensity": l_int.item(),
            "residual_reg": l_res.item(),
            "w_char": w_char.item(),
            "w_ssim": w_ssim.item(),
            "w_edge": w_edge.item(),
            "w_freq": w_freq.item(),
            "w_int": w_int.item()
        }

        return total_loss, loss_dict
