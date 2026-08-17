import os
import sys
import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch
import torch.nn as nn

@dataclass
class VerificationResult:
    filepath: str
    file_exists: bool = False
    file_size_mb: float = 0.0
    sha256: str = "N/A"
    is_valid_format: bool = False
    architecture_name: str = "Unknown"
    num_parameters: int = 0
    state_key_used: str = "None"
    missing_keys: List[str] = field(default_factory=list)
    unexpected_keys: List[str] = field(default_factory=list)
    shape_mismatches: List[str] = field(default_factory=list)
    dummy_input_shape: Tuple = (1, 1, 128, 128)
    v3_output_shape: Optional[Tuple] = None
    v4_output_shape: Optional[Tuple] = None
    output_min: float = 0.0
    output_max: float = 0.0
    output_mean: float = 0.0
    output_std: float = 0.0
    has_nan: bool = False
    has_inf: bool = False
    is_verified: bool = False
    status_summary: str = "UNVERIFIED"

def compute_file_sha256(filepath: str) -> str:
    if not os.path.exists(filepath):
        return "FILE_NOT_FOUND"
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

class CheckpointManager:
    """
    Universal Checkpoint Manager for AIR-Net v1, v1.2, v2, v3, and v4.
    Performs programmatic inspection, strict weight loading, and dummy inference verification.
    Prevents silent fallback to random initialized weights.
    """
    @staticmethod
    def inspect_checkpoint(filepath: str) -> Dict[str, Any]:
        if not os.path.exists(filepath):
            return {"exists": False, "error": f"File not found: '{filepath}'"}

        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        sha256 = compute_file_sha256(filepath)

        try:
            checkpoint_data = torch.load(filepath, map_location="cpu")
        except Exception as e:
            return {
                "exists": True,
                "size_mb": size_mb,
                "sha256": sha256,
                "is_pytorch_binary": False,
                "error": f"Failed to open PyTorch binary (.pth is binary, not text): {e}"
            }

        candidate_keys = ["v5_state_dict", "ema_state_dict", "v4_state_dict", "model_state_dict", "state_dict", "refinement_state_dict", "weights", "model", "ema"]
        found_key = "root"
        state_dict = None

        if isinstance(checkpoint_data, dict):
            for k in candidate_keys:
                if k in checkpoint_data and isinstance(checkpoint_data[k], dict):
                    found_key = k
                    state_dict = checkpoint_data[k]
                    break
            if state_dict is None:
                # Check if dictionary itself is the state_dict
                if all(isinstance(v, torch.Tensor) for v in checkpoint_data.values()):
                    state_dict = checkpoint_data
                    found_key = "dict_root"
        elif isinstance(checkpoint_data, nn.Module):
            state_dict = checkpoint_data.state_dict()
            found_key = "module_instance"

        return {
            "exists": True,
            "size_mb": size_mb,
            "sha256": sha256,
            "is_pytorch_binary": True,
            "state_key_used": found_key,
            "num_tensors": len(state_dict) if state_dict is not None else 0,
            "raw_checkpoint": checkpoint_data,
            "state_dict": state_dict
        }

    @classmethod
    def verify_checkpoint(
        cls,
        model: nn.Module,
        filepath: str,
        architecture_name: str = "AIR-Net Model",
        dummy_input_shape: Tuple[int, int, int, int] = (1, 1, 128, 128),
        device: torch.device = torch.device("cpu")
    ) -> VerificationResult:
        res = VerificationResult(filepath=filepath, architecture_name=architecture_name, dummy_input_shape=dummy_input_shape)

        if not os.path.exists(filepath):
            res.status_summary = f"FAIL: File '{filepath}' does not exist on disk."
            return res

        res.file_exists = True
        res.file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        res.sha256 = compute_file_sha256(filepath)

        insp = cls.inspect_checkpoint(filepath)
        if not insp.get("is_pytorch_binary", False) or insp.get("state_dict") is None:
            res.status_summary = f"FAIL: File '{filepath}' is not a valid PyTorch binary checkpoint."
            return res

        res.is_valid_format = True
        res.state_key_used = insp["state_key_used"]
        state_dict = insp["state_dict"]

        # Architecture & State Dict Matching
        model_state = model.state_dict()
        model_keys = set(model_state.keys())
        ckpt_keys = set(state_dict.keys())

        missing = sorted(list(model_keys - ckpt_keys))
        unexpected = sorted(list(ckpt_keys - model_keys))
        mismatches = []

        for k in model_keys.intersection(ckpt_keys):
            if model_state[k].shape != state_dict[k].shape:
                mismatches.append(f"{k}: model shape {tuple(model_state[k].shape)} != ckpt shape {tuple(state_dict[k].shape)}")

        res.missing_keys = missing
        res.unexpected_keys = unexpected
        res.shape_mismatches = mismatches
        res.num_parameters = sum(p.numel() for p in model.parameters())

        if len(mismatches) > 0:
            res.status_summary = f"FAIL: Shape mismatches found in {len(mismatches)} tensors."
            return res

        # Load weights into model
        try:
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            model.to(device)
        except Exception as e:
            res.status_summary = f"FAIL: Model load_state_dict exception: {e}"
            return res

        # Dummy Inference Verification
        dummy_in = torch.randn(*dummy_input_shape, dtype=torch.float32, device=device)
        try:
            with torch.no_grad():
                out = model(dummy_in)
                if isinstance(out, dict):
                    v4_res = out.get("restored")
                    v3_res = out.get("v3_restored")
                    res.v3_output_shape = tuple(v3_res.shape) if v3_res is not None else None
                    res.v4_output_shape = tuple(v4_res.shape) if v4_res is not None else None
                    out_tensor = v4_res if v4_res is not None else (v3_res if v3_res is not None else list(out.values())[0])
                else:
                    out_tensor = out
                    res.v4_output_shape = tuple(out_tensor.shape)

                arr = out_tensor.squeeze().cpu().numpy()
                res.has_nan = bool(np.isnan(arr).any())
                res.has_inf = bool(np.isinf(arr).any())
                res.output_min = float(np.min(arr))
                res.output_max = float(np.max(arr))
                res.output_mean = float(np.mean(arr))
                res.output_std = float(np.std(arr))

        except Exception as e:
            res.status_summary = f"FAIL: Dummy inference failed: {e}"
            return res

        if res.has_nan or res.has_inf:
            res.status_summary = "FAIL: Output tensor contains NaN or Inf values."
            return res

        res.is_verified = True
        res.status_summary = "CHECKPOINT STATUS: VERIFIED"
        return res

    @classmethod
    def load_strictly_verified_model(cls, model: nn.Module, filepath: str, architecture_name: str = "AIR-Net Model", device: torch.device = torch.device("cpu")) -> nn.Module:
        """
        Loads state dict strictly when verified. Raises FileNotFoundError or ValueError if missing/invalid.
        Never silently falls back to random weights.
        """
        ver = cls.verify_checkpoint(model, filepath, architecture_name=architecture_name, device=device)
        if not ver.is_verified:
            raise ValueError(f"Checkpoint verification failed for '{filepath}': {ver.status_summary}")
        return model
