import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.airnet_v4 import AIRNetV4
from utils.checkpoint_manager import CheckpointManager, compute_file_sha256
from utils.image_normalization import normalize_input, denormalize_output
from utils.edge_analysis import compute_sobel_edge_magnitude, prepare_edge_map_display
from utils.device import get_device

_CACHED_V4_MODEL = None
_CACHED_V4_PATH = None

def get_v4_model(v3_checkpoint_path: str = None, v4_checkpoint_path: str = None, device: torch.device = None) -> AIRNetV4:
    global _CACHED_V4_MODEL, _CACHED_V4_PATH
    if device is None:
        device = get_device()

    if v4_checkpoint_path is None:
        cand_paths = [
            os.path.join(PROJECT_ROOT, "outputs", "v4", "checkpoints", "best_v4_model.pth"),
            os.path.join(PROJECT_ROOT, "outputs", "v4", "checkpoints", "latest_v4_model.pth")
        ]
        for p in cand_paths:
            if os.path.exists(p):
                v4_checkpoint_path = p
                break

    if _CACHED_V4_MODEL is not None and _CACHED_V4_PATH == v4_checkpoint_path:
        return _CACHED_V4_MODEL

    model = AIRNetV4().to(device)

    # 1. Load V3 Foundation
    if v3_checkpoint_path is None:
        v3_cand = [
            os.path.join(PROJECT_ROOT, "outputs", "v3", "checkpoints", "airnet_v3_ema_best_model.pth"),
            os.path.join(PROJECT_ROOT, "outputs", "v3", "checkpoints", "airnet_v3_best_model.pth")
        ]
        for c in v3_cand:
            if os.path.exists(c):
                v3_checkpoint_path = c
                break

    if v3_checkpoint_path and os.path.exists(v3_checkpoint_path):
        v3_ver = CheckpointManager.verify_checkpoint(model.v3_base, v3_checkpoint_path, architecture_name="AIR-Net v3 Base", device=device)
        if not v3_ver.is_verified:
            raise ValueError(f"AIR-Net v3 foundation verification failed: {v3_ver.status_summary}")
        print(f"[OK] AIR-Net v3 foundation loaded and verified from: '{v3_checkpoint_path}'")

    # 2. Load V4 Checkpoint
    if v4_checkpoint_path and os.path.exists(v4_checkpoint_path):
        v4_ver = CheckpointManager.verify_checkpoint(model, v4_checkpoint_path, architecture_name="AIR-Net v4 System", device=device)
        if not v4_ver.is_verified:
            raise ValueError(f"AIR-Net v4 checkpoint verification failed: {v4_ver.status_summary}")
        print(f"[OK] AIR-Net v4 checkpoint loaded and verified from: '{v4_checkpoint_path}'")
    else:
        raise FileNotFoundError(f"AIR-Net v4 checkpoint not found. Expected: '{v4_checkpoint_path}'. Inference disabled.")

    model.eval()
    _CACHED_V4_MODEL = model
    _CACHED_V4_PATH = v4_checkpoint_path
    return model

def restore_v4_image(
    lr_input,
    v3_checkpoint_path: str = None,
    v4_checkpoint_path: str = None,
    device: torch.device = None
) -> dict:
    if device is None:
        device = get_device()

    arr = normalize_input(lr_input)
    model = get_v4_model(v3_checkpoint_path=v3_checkpoint_path, v4_checkpoint_path=v4_checkpoint_path, device=device)

    raw_indices = model.indexer.compute_indices(arr)
    lr_tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad(), torch.inference_mode():
        out = model(lr_tensor)
        v4_raw = denormalize_output(out["restored"])
        v3_raw = denormalize_output(out["v3_restored"])
        residual_raw = out["residual"].squeeze().cpu().numpy()
        r_probs = out["routing_probs"].squeeze().cpu().numpy()

    categories = ["EDGE_DOMINANT", "TEXTURE_DOMINANT", "NOISE_DOMINANT", "SMOOTH_LOW_CONTRAST", "SPARSE_FEATURE"]
    dom_cat = categories[int(np.argmax(r_probs))]
    r_dict = {cat: float(p) for cat, p in zip(categories, r_probs)}

    return {
        "restored": v4_raw,
        "v3_restored": v3_raw,
        "residual": residual_raw,
        "dominant_category": dom_cat,
        "routing_probs": r_dict,
        "indices": raw_indices
    }

def main():
    parser = argparse.ArgumentParser(description="AIR-Net v4 Standalone Single-Image Inference API")
    parser.add_argument("--input", type=str, required=True, help="Path to 128x128 input image (.npy, .png, .jpg)")
    parser.add_argument("--v3-checkpoint", type=str, default="", help="Path to AIR-Net v3 foundation checkpoint")
    parser.add_argument("--v4-checkpoint", type=str, default="", help="Path to AIR-Net v4 checkpoint")
    parser.add_argument("--output", type=str, default="restored_v4.png", help="Path for output PNG file")
    args = parser.parse_args()

    print("==============================================================================")
    print("AIR-Net v4 STANDALONE INFERENCE (128x128 -> 256x256)")
    print("==============================================================================")

    res = restore_v4_image(
        args.input,
        v3_checkpoint_path=args.v3_checkpoint if args.v3_checkpoint else None,
        v4_checkpoint_path=args.v4_checkpoint if args.v4_checkpoint else None
    )

    out_img = Image.fromarray((res["restored"] * 255.0).round().astype(np.uint8))
    out_img.save(args.output)
    print(f"[OK] Saved restored 256x256 image to: '{args.output}'")

    npy_out = args.output.replace(".png", ".npy")
    np.save(npy_out, res["restored"])
    print(f"[OK] Saved restored 256x256 NPY array to: '{npy_out}'")

    edge_mag = compute_sobel_edge_magnitude(res["restored"])
    edge_vis = prepare_edge_map_display(edge_mag)
    edge_img = Image.fromarray((edge_vis * 255.0).round().astype(np.uint8))
    edge_out = args.output.replace(".png", "_edge.png")
    edge_img.save(edge_out)
    print(f"[OK] Saved 256x256 edge map to: '{edge_out}'")

    diag_out = args.output.replace(".png", "_diagnostics.json")
    diag = {
        "input_shape": [128, 128],
        "output_shape": list(res["restored"].shape),
        "dominant_category": res["dominant_category"],
        "routing_probs": res["routing_probs"],
        "output_min": float(np.min(res["restored"])),
        "output_max": float(np.max(res["restored"])),
        "output_mean": float(np.mean(res["restored"])),
        "output_std": float(np.std(res["restored"]))
    }
    with open(diag_out, "w") as f:
        json.dump(diag, f, indent=4)
    print(f"[OK] Saved diagnostics to: '{diag_out}'")
    print("==============================================================================")

if __name__ == "__main__":
    main()
