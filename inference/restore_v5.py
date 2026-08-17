import os
import sys
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.airnet_v5 import AIRNetV5
from utils.checkpoint_manager import CheckpointManager
from utils.image_normalization import normalize_input, denormalize_output

def restore_image_v5(
    lr_input: np.ndarray,
    v5_checkpoint_path: str = "",
    v4_checkpoint_path: str = "",
    v3_checkpoint_path: str = "",
    device: torch.device = None
) -> dict:
    """
    Standalone AIR-Net v5 Inference API (128x128 -> 256x256).
    Pure inference pipeline. Does NOT require Ground Truth or per-image post-processing.
    Priority: v5 system checkpoint > v4 foundation > v3 foundation > RuntimeError.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lr_norm = normalize_input(lr_input)
    assert lr_norm.shape == (128, 128), f"Input resolution mismatch: {lr_norm.shape}"

    foundation_mode = "v4"
    if v3_checkpoint_path and not v4_checkpoint_path:
        foundation_mode = "v3"

    v5_model = AIRNetV5(foundation_mode=foundation_mode).to(device)

    # 1. Load V5 System Checkpoint if available
    if v5_checkpoint_path and os.path.exists(v5_checkpoint_path):
        state_dict, meta = CheckpointManager.load_checkpoint_state_dict(v5_checkpoint_path, map_location="cpu")
        v5_model.load_state_dict(state_dict, strict=False)
    # 2. Or load V4/V3 foundation checkpoint into base
    elif v4_checkpoint_path and os.path.exists(v4_checkpoint_path):
        state_dict, meta = CheckpointManager.load_checkpoint_state_dict(v4_checkpoint_path, map_location="cpu")
        v5_model.foundation_base.load_state_dict(state_dict, strict=True)
    elif v3_checkpoint_path and os.path.exists(v3_checkpoint_path):
        state_dict, meta = CheckpointManager.load_checkpoint_state_dict(v3_checkpoint_path, map_location="cpu")
        v5_model.foundation_base.load_state_dict(state_dict, strict=False)
    else:
        raise RuntimeError("No valid checkpoint found for AIR-Net v5 inference. Random weights prohibited.")

    v5_model.eval()

    lr_t = torch.from_numpy(lr_norm).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad(), torch.inference_mode():
        out_dict = v5_model(lr_t)

    v5_restored = denormalize_output(out_dict["restored"])
    foundation_restored = denormalize_output(out_dict["foundation_restored"])

    assert v5_restored.shape == (256, 256), f"V5 output resolution mismatch: {v5_restored.shape}"

    return {
        "restored": v5_restored,
        "foundation_restored": foundation_restored,
        "residual": out_dict["residual"].squeeze().cpu().numpy(),
        "alpha": float(out_dict["alpha"].item()),
        "routing_probs": out_dict["routing_probs"].squeeze().cpu().numpy()
    }
