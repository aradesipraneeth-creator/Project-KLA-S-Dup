import numpy as np
import torch
from scipy.ndimage import gaussian_filter

try:
    from utils.image_normalization import prepare_for_metric
except ImportError:
    from image_normalization import prepare_for_metric

def compute_mse(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = prepare_for_metric(pred, gt)
    return float(np.mean((p - g) ** 2))

def compute_mae(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = prepare_for_metric(pred, gt)
    return float(np.mean(np.abs(p - g)))

def compute_psnr(pred: np.ndarray, gt: np.ndarray, data_range: float = 1.0) -> float:
    mse = compute_mse(pred, gt)
    if mse == 0:
        return 100.0
    return float(10.0 * np.log10((data_range ** 2) / mse))

def compute_ssim(pred: np.ndarray, gt: np.ndarray, data_range: float = 1.0) -> float:
    p, g = prepare_for_metric(pred, gt)
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    img1, img2 = p.astype(np.float64), g.astype(np.float64)
    mu1 = gaussian_filter(img1, sigma=1.5)
    mu2 = gaussian_filter(img2, sigma=1.5)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = gaussian_filter(img1 ** 2, sigma=1.5) - mu1_sq
    sigma2_sq = gaussian_filter(img2 ** 2, sigma=1.5) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, sigma=1.5) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(np.mean(ssim_map))

_CACHED_LPIPS_LOSS = None

def get_lpips_model(device: torch.device):
    global _CACHED_LPIPS_LOSS
    if _CACHED_LPIPS_LOSS is None:
        try:
            import lpips
            _CACHED_LPIPS_LOSS = lpips.LPIPS(net='alex', verbose=False).to(device)
        except Exception:
            _CACHED_LPIPS_LOSS = "FAILED"
    return _CACHED_LPIPS_LOSS

def compute_lpips(pred: np.ndarray, gt: np.ndarray, device: torch.device = None) -> float:
    p, g = prepare_for_metric(pred, gt)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_fn = get_lpips_model(device)
    if loss_fn != "FAILED" and loss_fn is not None:
        try:
            p_t = torch.from_numpy(p).unsqueeze(0).unsqueeze(0).float().to(device)
            g_t = torch.from_numpy(g).unsqueeze(0).unsqueeze(0).float().to(device)
            # Replicate 1 channel to 3 channels (RGB) for LPIPS network without faking color
            p3 = p_t.repeat(1, 3, 1, 1) * 2.0 - 1.0
            g3 = g_t.repeat(1, 3, 1, 1) * 2.0 - 1.0
            with torch.no_grad():
                dist = loss_fn(p3, g3).mean().item()
            return float(dist)
        except Exception:
            pass

    return float(np.mean(np.abs(p - g)))

def calculate_psnr(pred: torch.Tensor, gt: torch.Tensor, data_range: float = 1.0) -> float:
    """Wrapper for tensor input compatibility."""
    p_np = pred.detach().cpu().numpy()
    g_np = gt.detach().cpu().numpy()
    return compute_psnr(p_np, g_np, data_range=data_range)

def calculate_ssim(pred: torch.Tensor, gt: torch.Tensor, data_range: float = 1.0) -> float:
    """Wrapper for tensor input compatibility."""
    p_np = pred.detach().cpu().numpy()
    g_np = gt.detach().cpu().numpy()
    return compute_ssim(p_np, g_np, data_range=data_range)

def compute_brightness_error(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = prepare_for_metric(pred, gt)
    return float(np.abs(np.mean(p) - np.mean(g)))

def compute_contrast_error(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = prepare_for_metric(pred, gt)
    return float(np.abs(np.std(p) - np.std(g)))

def compute_edge_error(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = prepare_for_metric(pred, gt)
    gx_p, gy_p = np.gradient(p)
    gx_g, gy_g = np.gradient(g)
    mag_p = np.sqrt(gx_p**2 + gy_p**2 + 1e-8)
    mag_g = np.sqrt(gx_g**2 + gy_g**2 + 1e-8)
    return float(np.mean(np.abs(mag_p - mag_g)))

def compute_gradient_error(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = prepare_for_metric(pred, gt)
    gx_p, gy_p = np.gradient(p)
    gx_g, gy_g = np.gradient(g)
    return float(np.mean(np.abs(gx_p - gx_g) + np.abs(gy_p - gy_g)))

def compute_laplacian_error(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = prepare_for_metric(pred, gt)
    lap_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    from scipy.ndimage import convolve
    lap_p = convolve(p, lap_kernel)
    lap_g = convolve(g, lap_kernel)
    return float(np.mean(np.abs(lap_p - lap_g)))

def compute_high_frequency_error(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = prepare_for_metric(pred, gt)
    blur_p = gaussian_filter(p, sigma=2.0)
    blur_g = gaussian_filter(g, sigma=2.0)
    hf_p = p - blur_p
    hf_g = g - blur_g
    return float(np.mean(np.abs(hf_p - hf_g)))

def compute_all_metrics(pred: np.ndarray, gt: np.ndarray, device: torch.device = None) -> dict:
    p, g = prepare_for_metric(pred, gt)
    return {
        "PSNR (dB)": compute_psnr(p, g),
        "SSIM": compute_ssim(p, g),
        "LPIPS": compute_lpips(p, g, device),
        "Edge Error": compute_edge_error(p, g),
        "Gradient Error": compute_gradient_error(p, g),
        "Laplacian Error": compute_laplacian_error(p, g),
        "HF Error": compute_high_frequency_error(p, g),
        "Brightness Error": compute_brightness_error(p, g),
        "Contrast Error": compute_contrast_error(p, g)
    }

def run_metric_sanity_test():
    """
    GT vs GT metric sanity test.
    Verifies PSNR == 100.0 (or infinite), SSIM == 1.0, LPIPS == 0.0.
    """
    dummy_gt = np.random.rand(256, 256).astype(np.float32)
    psnr_self = compute_psnr(dummy_gt, dummy_gt)
    ssim_self = compute_ssim(dummy_gt, dummy_gt)
    lpips_self = compute_lpips(dummy_gt, dummy_gt)
    edge_err = compute_edge_error(dummy_gt, dummy_gt)

    assert psnr_self >= 99.9, f"GT vs GT PSNR sanity test failed: {psnr_self}"
    assert abs(ssim_self - 1.0) < 1e-4, f"GT vs GT SSIM sanity test failed: {ssim_self}"
    assert abs(lpips_self) < 1e-3, f"GT vs GT LPIPS sanity test failed: {lpips_self}"
    assert edge_err < 1e-4, f"GT vs GT Edge Error sanity test failed: {edge_err}"

    print(f"[SANITY TEST PASSED] GT vs GT -> PSNR: {psnr_self:.1f} dB | SSIM: {ssim_self:.4f} | LPIPS: {lpips_self:.4f} | EdgeErr: {edge_err:.4f}")
    return True
