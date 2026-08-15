import os
import sys
import json
import time
import argparse
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.airnet_v3 import AIRNetV3
from models.airnet_v4 import AIRNetV4
from utils.checkpoint_manager import CheckpointManager, compute_file_sha256

def main():
    parser = argparse.ArgumentParser(description="AIR-Net v3 / v4 Checkpoint Verification Script")
    parser.add_argument("--v3-checkpoint", type=str, default="", help="Path to AIR-Net v3 checkpoint")
    parser.add_argument("--v4-checkpoint", type=str, default="", help="Path to AIR-Net v4 checkpoint")
    args = parser.parse_args()

    print("==============================================================================")
    print("AIR-NET V3 / V4 PROGRAMMATIC CHECKPOINT VERIFICATION")
    print("==============================================================================")

    v3_path = args.v3_checkpoint
    if not v3_path:
        cand_v3 = [
            os.path.join(PROJECT_ROOT, "outputs", "v3", "checkpoints", "airnet_v3_ema_best_model.pth"),
            os.path.join(PROJECT_ROOT, "outputs", "v3", "checkpoints", "airnet_v3_best_model.pth")
        ]
        for c in cand_v3:
            if os.path.exists(c):
                v3_path = c
                break

    v4_path = args.v4_checkpoint
    if not v4_path:
        cand_v4 = [
            os.path.join(PROJECT_ROOT, "outputs", "v4", "checkpoints", "best_v4_model.pth"),
            os.path.join(PROJECT_ROOT, "outputs", "v4", "checkpoints", "latest_v4_model.pth")
        ]
        for c in cand_v4:
            if os.path.exists(c):
                v4_path = c
                break

    v3_verified = False
    v4_verified = False
    v3_sha = "N/A"
    v4_sha = "N/A"

    # 1. Verify V3 Checkpoint
    if v3_path and os.path.exists(v3_path):
        print(f"\n--- Verifying AIR-Net v3 Foundation Checkpoint: '{v3_path}' ---")
        v3_model = AIRNetV3()
        res_v3 = CheckpointManager.verify_checkpoint(v3_model, v3_path, architecture_name="AIR-Net v3 Foundation")
        v3_sha = res_v3.sha256
        v3_verified = res_v3.is_verified

        print(f"  File Size:     {res_v3.file_size_mb:.2f} MB")
        print(f"  SHA256:        {res_v3.sha256}")
        print(f"  Format:        PyTorch Binary Checkpoint")
        print(f"  Parameters:    {res_v3.num_parameters:,}")
        print(f"  State Key:     '{res_v3.state_key_used}'")
        print(f"  Missing Keys:  {len(res_v3.missing_keys)}")
        print(f"  Unexpected:    {len(res_v3.unexpected_keys)}")
        print(f"  Output Range:  [{res_v3.output_min:.4f}, {res_v3.output_max:.4f}] (Mean: {res_v3.output_mean:.4f}, Std: {res_v3.output_std:.4f})")
        print(f"  NaN: {res_v3.has_nan} | Inf: {res_v3.has_inf}")
        print(f"  Status:        {res_v3.status_summary}")
    else:
        print(f"\n[NOTICE] AIR-Net v3 checkpoint file not found at '{v3_path}'")

    # 2. Verify V4 Checkpoint
    if v4_path and os.path.exists(v4_path):
        print(f"\n--- Verifying AIR-Net v4 Checkpoint: '{v4_path}' ---")
        v4_model = AIRNetV4()
        res_v4 = CheckpointManager.verify_checkpoint(v4_model, v4_path, architecture_name="AIR-Net v4 System")
        v4_sha = res_v4.sha256
        v4_verified = res_v4.is_verified

        print(f"  File Size:     {res_v4.file_size_mb:.2f} MB")
        print(f"  SHA256:        {res_v4.sha256}")
        print(f"  Format:        PyTorch Binary Checkpoint")
        print(f"  Parameters:    {res_v4.num_parameters:,}")
        print(f"  State Key:     '{res_v4.state_key_used}'")
        print(f"  Missing Keys:  {len(res_v4.missing_keys)}")
        print(f"  Unexpected:    {len(res_v4.unexpected_keys)}")
        print(f"  V3 Shape:      {res_v4.v3_output_shape}")
        print(f"  V4 Shape:      {res_v4.v4_output_shape}")
        print(f"  Output Range:  [{res_v4.output_min:.4f}, {res_v4.output_max:.4f}] (Mean: {res_v4.output_mean:.4f}, Std: {res_v4.output_std:.4f})")
        print(f"  NaN: {res_v4.has_nan} | Inf: {res_v4.has_inf}")
        print(f"  Status:        {res_v4.status_summary}")
    else:
        print(f"\n[NOTICE] AIR-Net v4 checkpoint file not found at '{v4_path}'")

    # Create Checkpoint Manifest
    manifest_dir = os.path.join(PROJECT_ROOT, "outputs", "v4", "checkpoints")
    os.makedirs(manifest_dir, exist_ok=True)
    manifest_path = os.path.join(manifest_dir, "checkpoint_manifest.json")

    manifest = {
        "project": "KLA Semiconductor Image Restoration (Project S)",
        "verification_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_resolution": "128x128",
        "output_resolution": "256x256",
        "normalization": "[0.0, 1.0] float32",
        "v3_checkpoint": {
            "path": v3_path,
            "sha256": v3_sha,
            "verified": v3_verified
        },
        "v4_checkpoint": {
            "path": v4_path,
            "sha256": v4_sha,
            "verified": v4_verified
        }
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)
    print(f"\n[OK] Checkpoint Manifest created at '{manifest_path}'")

    print("\n==============================================================================")
    if (v3_path and not v3_verified) or (v4_path and not v4_verified):
        print("VERIFICATION RESULT: FAIL (One or more checkpoints failed verification)")
        sys.exit(1)
    else:
        print("VERIFICATION RESULT: PASS (All available checkpoints verified)")
        sys.exit(0)

if __name__ == "__main__":
    main()
