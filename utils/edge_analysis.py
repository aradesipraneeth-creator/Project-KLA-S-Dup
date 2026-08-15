import numpy as np
import torch
import torch.nn.functional as F

def compute_sobel_edge_magnitude(img_2d: np.ndarray) -> np.ndarray:
    """
    Computes raw 2D Sobel gradient magnitude map for an image array.
    """
    t = torch.from_numpy(img_2d.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
    gx = F.conv2d(t, sobel_x, padding=1)
    gy = F.conv2d(t, sobel_y, padding=1)
    mag = torch.sqrt(gx**2 + gy**2 + 1e-8).squeeze().numpy()
    return mag

def prepare_edge_map_display(edge_mag: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    """
    Robust percentile normalization for edge map visualization ONLY.
    Scales edge magnitude between 1st and 99th percentiles to eliminate black edge map display artifacts.
    Does NOT alter the underlying prediction array or quantitative metrics.
    """
    if edge_mag is None:
        return None
    mag = np.array(edge_mag, dtype=np.float32)
    max_val = np.max(mag)
    if max_val == 0:
        return np.zeros_like(mag)

    p1 = np.percentile(mag, p_low)
    p99 = np.percentile(mag, p_high)

    if p99 > p1:
        display = np.clip((mag - p1) / (p99 - p1), 0.0, 1.0)
    else:
        display = np.clip(mag / max_val, 0.0, 1.0)

    return display

def compute_edge_statistics(img_2d: np.ndarray) -> dict:
    """
    Calculates quantitative edge metrics: edge_mean, edge_std, edge_density, gradient_energy.
    """
    mag = compute_sobel_edge_magnitude(img_2d)
    return {
        "edge_mean": float(np.mean(mag)),
        "edge_std": float(np.std(mag)),
        "edge_density": float(np.mean(mag > 0.1)),
        "gradient_energy": float(np.mean(mag ** 2))
    }
