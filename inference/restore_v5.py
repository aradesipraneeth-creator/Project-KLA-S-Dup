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
    v3_checkpoint_path: str = "",
    v5_checkpoint_path: str = "",
    device: torch.device = None
) -> dict:
    """
    Standalone AIR-Net v5 Inference API (128x128 -> 256x256).
    Pure inference pipeline. Does NOT require Ground Truth or per-image post-processing.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lr_norm = normalize_input(lr_input)
    assert lr_norm.shape == (128, 128), f"Input resolution mismatch: {lr_norm.shape}"

    # Model Initialization
    v5_model = AIRNetV5().to(device)

    # Load V5 System Checkpoint (which includes V3 foundation)
    if v5_checkpoint_path and os.path.exists(v5_checkpoint_path):
        v5_model = CheckpointManager.load_strictly_verified_model(
            v5_model, v5_checkpoint_path, architecture_name="AIR-Net v5 System", device=device
        )
    elif v3_checkpoint_path and os.path.exists(v3_checkpoint_path):
        # Fallback to V3 base loading if V5 refinement checkpoint not yet trained
        v5_model.v3_base = CheckpointManager.load_strictly_verified_model(
            v5_model.v3_base, v3_checkpoint_path, architecture_name="AIR-Net v3 Foundation", device=device
        )

    v5_model.eval()

    lr_t = torch.from_numpy(lr_norm).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad(), torch.inference_mode():
        out_dict = v5_model(lr_t)

    v5_restored = denormalize_output(out_dict["restored"])
    v3_restored = denormalize_output(out_dict["v3_restored"])

    assert v5_restored.shape == (256, 256), f"V5 output resolution mismatch: {v5_restored.shape}"

    return {
        "restored": v5_restored,
        "v3_restored": v3_restored,
        "residual": out_dict["residual"].squeeze().cpu().numpy(),
        "alpha": float(out_dict["alpha"].item()),
        "routing_probs": out_dict["routing_probs"].squeeze().cpu().numpy()
    }
