import numpy as np
import torch
from PIL import Image
from typing import Tuple, Union

def normalize_for_display_and_metrics(img_input: Union[np.ndarray, torch.Tensor, Image.Image]) -> np.ndarray:
    """
    Central authoritative image normalization function for display and metrics.
    Accepts: numpy arrays, torch tensors, PIL images, uint8, uint16, float32, float64,
             ranges [0, 1], [0, 255], [-1, 1], [0, 65535].
    Returns: float32 2D numpy array (H, W) in range [0.0, 1.0].
    """
    if torch.is_tensor(img_input):
        arr = img_input.detach().cpu().numpy()
    elif hasattr(img_input, "read") or isinstance(img_input, Image.Image):
        if not isinstance(img_input, Image.Image):
            img_input = Image.open(img_input)
        img = img_input.convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0
    else:
        arr = np.array(img_input)

    arr = np.squeeze(arr)
    if arr.ndim > 2:
        if arr.shape[0] in [1, 3, 4]:
            arr = arr[0]
        elif arr.shape[2] in [1, 3, 4]:
            arr = arr[:, :, 0]

    arr = arr.astype(np.float32)

    if np.isnan(arr).any() or np.isinf(arr).any():
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)

    min_val, max_val = float(np.min(arr)), float(np.max(arr))

    if max_val <= 1.0 and min_val >= 0.0:
        norm_arr = arr
    elif max_val > 1.0 and max_val <= 255.0 and min_val >= 0.0:
        norm_arr = arr / 255.0
    elif min_val >= -1.0 and max_val <= 1.0 and min_val < 0.0:
        norm_arr = (arr + 1.0) / 2.0
    elif max_val > 1.0 and max_val <= 65535.0 and min_val >= 0.0:
        norm_arr = arr / 65535.0
    else:
        if max_val > min_val:
            norm_arr = (arr - min_val) / (max_val - min_val)
        else:
            norm_arr = np.zeros_like(arr)

    return np.clip(norm_arr, 0.0, 1.0).astype(np.float32)

def normalize_input(img_input: Union[np.ndarray, Image.Image, torch.Tensor]) -> np.ndarray:
    """
    Authoritative input image normalization utility.
    Converts NPY, PIL, or Tensor input to float32 2D array (128, 128) in range [0.0, 1.0].
    """
    arr = normalize_for_display_and_metrics(img_input)

    if arr.shape != (128, 128):
        raise ValueError(f"Input image shape mismatch. Expected: (128, 128), Received: {arr.shape}")

    return arr

def normalize_target(target_img: Union[np.ndarray, Image.Image, torch.Tensor]) -> np.ndarray:
    """
    Authoritative target image normalization utility.
    Converts NPY, PIL, or Tensor target to float32 2D array (256, 256) in range [0.0, 1.0].
    """
    arr = normalize_for_display_and_metrics(target_img)

    if arr.shape != (256, 256):
        raise ValueError(f"Target Ground Truth shape mismatch. Expected: (256, 256), Received: {arr.shape}")

    return arr

def denormalize_output(output: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Converts model raw output to float32 2D array (256, 256) clamped safely to [0.0, 1.0].
    """
    return normalize_for_display_and_metrics(output)

def prepare_for_metric(pred: Union[np.ndarray, torch.Tensor], gt: Union[np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepares prediction and Ground Truth arrays for quantitative evaluation.
    Verifies float32 dtype, matching 2D shapes (256, 256), and intensity range [0.0, 1.0].
    Raises explicit ValueError if shapes mismatch.
    """
    p = normalize_for_display_and_metrics(pred)
    g = normalize_for_display_and_metrics(gt)

    if p.shape != g.shape:
        raise ValueError(f"Prediction and GT resolution mismatch: pred={p.shape}, gt={g.shape}")

    if p.shape != (256, 256):
        raise ValueError(f"Quantitative metrics require 256x256 resolution, received {p.shape}")

    return p, g

def validate_metric_inputs(pred: Union[np.ndarray, torch.Tensor], gt: Union[np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Strict validation for prediction and Ground Truth arrays before metric computation.
    Checks matching shape (256, 256), float32 dtype, finite values, and [0.0, 1.0] range.
    Fails loudly with descriptive error messages instead of broadcast errors.
    """
    p = normalize_for_display_and_metrics(pred)
    g = normalize_for_display_and_metrics(gt)

    if p.shape != (256, 256):
        raise ValueError(f"Resolution mismatch: prediction shape {p.shape} != required (256, 256)")

    if g.shape != (256, 256):
        raise ValueError(f"Resolution mismatch: Ground Truth shape {g.shape} != required (256, 256)")

    if p.shape != g.shape:
        raise ValueError(f"Shape mismatch: prediction {p.shape} != Ground Truth {g.shape}")

    if np.isnan(p).any() or np.isinf(p).any():
        raise ValueError("Prediction array contains NaN or Inf values")

    if np.isnan(g).any() or np.isinf(g).any():
        raise ValueError("Ground Truth array contains NaN or Inf values")

    return p, g

def compute_array_stats(name: str, arr: Union[np.ndarray, torch.Tensor]) -> dict:
    """
    Computes min, max, mean, std, shape, dtype, NaN/Inf counts for diagnostic logging.
    """
    if arr is None:
        return {"name": name, "status": "N/A", "shape": "N/A", "min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}

    if torch.is_tensor(arr):
        arr_np = arr.detach().cpu().numpy()
    else:
        arr_np = np.array(arr)

    arr_np = np.squeeze(arr_np)
    has_nan = bool(np.isnan(arr_np).any())
    has_inf = bool(np.isinf(arr_np).any())

    return {
        "name": name,
        "status": "VALID" if not (has_nan or has_inf) else "INVALID",
        "shape": str(arr_np.shape),
        "dtype": str(arr_np.dtype),
        "min": float(np.min(arr_np)),
        "max": float(np.max(arr_np)),
        "mean": float(np.mean(arr_np)),
        "std": float(np.std(arr_np)),
        "has_nan": has_nan,
        "has_inf": has_inf
    }

def prepare_for_display(img_2d: np.ndarray) -> np.ndarray:
    """
    Prepares 2D float32 array for Streamlit / Matplotlib visualization in range [0.0, 1.0].
    """
    return normalize_for_display_and_metrics(img_2d)
