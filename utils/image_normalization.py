import numpy as np
import torch
from PIL import Image
from typing import Tuple, Union

def normalize_input(img_input: Union[np.ndarray, Image.Image, torch.Tensor]) -> np.ndarray:
    """
    Authoritative input image normalization utility.
    Converts NPY, PIL, or Tensor input to float32 2D array (128, 128) in range [0.0, 1.0].
    """
    if torch.is_tensor(img_input):
        arr = img_input.detach().cpu().numpy().astype(np.float32)
    elif isinstance(img_input, np.ndarray):
        arr = img_input.astype(np.float32)
    elif hasattr(img_input, "read") or isinstance(img_input, Image.Image):
        if not isinstance(img_input, Image.Image):
            img_input = Image.open(img_input)
        img = img_input.convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0
    else:
        raise ValueError("Unsupported input format for normalize_input")

    arr = np.squeeze(arr)
    if arr.ndim > 2:
        arr = arr[0]
    if arr.max() > 1.0:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)

    if arr.shape != (128, 128):
        raise ValueError(f"Input image shape mismatch. Expected: (128, 128), Received: {arr.shape}")

    return arr

def normalize_target(target_img: Union[np.ndarray, Image.Image, torch.Tensor]) -> np.ndarray:
    """
    Authoritative target image normalization utility.
    Converts NPY, PIL, or Tensor target to float32 2D array (256, 256) in range [0.0, 1.0].
    """
    if torch.is_tensor(target_img):
        arr = target_img.detach().cpu().numpy().astype(np.float32)
    elif isinstance(target_img, np.ndarray):
        arr = target_img.astype(np.float32)
    elif hasattr(target_img, "read") or isinstance(target_img, Image.Image):
        if not isinstance(target_img, Image.Image):
            target_img = Image.open(target_img)
        img = target_img.convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0
    else:
        raise ValueError("Unsupported target format for normalize_target")

    arr = np.squeeze(arr)
    if arr.ndim > 2:
        arr = arr[0]
    if arr.max() > 1.0:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)

    if arr.shape != (256, 256):
        raise ValueError(f"Target Ground Truth shape mismatch. Expected: (256, 256), Received: {arr.shape}")

    return arr

def denormalize_output(output: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Converts model raw output to float32 2D array (256, 256) clamped safely to [0.0, 1.0].
    """
    if torch.is_tensor(output):
        arr = output.detach().cpu().numpy().astype(np.float32)
    else:
        arr = np.array(output, dtype=np.float32)

    arr = np.squeeze(arr)
    if arr.ndim > 2:
        arr = arr[0]

    return np.clip(arr, 0.0, 1.0)

def prepare_for_metric(pred: Union[np.ndarray, torch.Tensor], gt: Union[np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepares prediction and Ground Truth arrays for quantitative evaluation.
    Verifies float32 dtype, matching 2D shapes (256, 256), and intensity range [0.0, 1.0].
    Raises explicit ValueError if shapes mismatch.
    """
    p = denormalize_output(pred)
    g = denormalize_output(gt)

    if p.shape != g.shape:
        raise ValueError(f"Prediction and GT resolution mismatch: pred={p.shape}, gt={g.shape}")

    if p.shape != (256, 256):
        raise ValueError(f"Quantitative metrics require 256x256 resolution, received {p.shape}")

    return p, g

def prepare_for_display(img_2d: np.ndarray) -> np.ndarray:
    """
    Prepares 2D float32 array for Streamlit / Matplotlib visualization in range [0.0, 1.0].
    """
    arr = np.array(img_2d, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)
