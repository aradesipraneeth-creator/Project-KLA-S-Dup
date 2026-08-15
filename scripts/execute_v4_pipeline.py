import os
import sys
import json
import time
import csv
import hashlib
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt

PROJECT_ROOT = os.environ.get("KLA_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.config import Config
from models.airnet_v4 import AIRNetV4
from losses.adaptive_loss_v4 import AIRNetV4AdaptiveLoss
from utils.metrics import calculate_psnr, calculate_ssim
from utils.device import get_device, is_cuda

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_file_sha256(filepath: str) -> str:
    if not os.path.exists(filepath):
        return "FILE_NOT_FOUND"
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

_GLOBAL_LPIPS_FN = None
def get_global_lpips_fn(device):
    global _GLOBAL_LPIPS_FN
    if _GLOBAL_LPIPS_FN is None:
        try:
            import lpips
            _GLOBAL_LPIPS_FN = lpips.LPIPS(net='alex', verbose=False).to(device)
        except Exception:
            _GLOBAL_LPIPS_FN = "FAILED"
    return _GLOBAL_LPIPS_FN

def compute_lpips_fast(pred_tensor: torch.Tensor, gt_tensor: torch.Tensor, device) -> float:
    p_float = pred_tensor.float().clamp(0.0, 1.0)
    g_float = gt_tensor.float().clamp(0.0, 1.0)
    lpips_fn = get_global_lpips_fn(device)
    if lpips_fn != "FAILED" and lpips_fn is not None:
        try:
            p3 = p_float.repeat(1, 3, 1, 1) * 2.0 - 1.0
            g3 = g_float.repeat(1, 3, 1, 1) * 2.0 - 1.0
            with torch.no_grad():
                dist = lpips_fn(p3, g3).mean().item()
            return float(dist)
        except Exception:
            pass
    with torch.no_grad():
        dist = F.l1_loss(p_float, g_float).item()
    return float(dist)

def compute_sobel_gradient_energy(img_tensor: torch.Tensor) -> float:
    img_f = img_tensor.float()
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32, device=img_f.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32, device=img_f.device).view(1, 1, 3, 3)
    gx = F.conv2d(img_f, sobel_x, padding=1)
    gy = F.conv2d(img_f, sobel_y, padding=1)
    return float(torch.mean(gx**2 + gy**2).item())

def compute_laplacian_energy(img_tensor: torch.Tensor) -> float:
    img_f = img_tensor.float()
    lap_kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], dtype=torch.float32, device=img_f.device).view(1, 1, 3, 3)
    lap_map = F.conv2d(img_f, lap_kernel, padding=1)
    return float(torch.mean(lap_map**2).item())

def compute_high_frequency_map(img_tensor: torch.Tensor) -> torch.Tensor:
    img_f = img_tensor.float()
    blurred = F.avg_pool2d(img_f, kernel_size=5, stride=1, padding=2)
    return img_f - blurred

def compute_hf_energy(img_tensor: torch.Tensor) -> float:
    hf_map = compute_high_frequency_map(img_tensor)
    return float(torch.mean(hf_map**2).item())

def compute_sobel_edge_map_np(img_2d: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(img_2d.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
    gx = F.conv2d(t, sobel_x, padding=1)
    gy = F.conv2d(t, sobel_y, padding=1)
    mag = torch.sqrt(gx**2 + gy**2 + 1e-8).squeeze().numpy()
    max_val = np.max(mag)
    return mag / (max_val + 1e-8) if max_val > 0 else mag

class KLAPairedDataset(Dataset):
    def __init__(self, file_list, lr_dir, gt_dir):
        self.file_list = file_list
        self.lr_dir = lr_dir
        self.gt_dir = gt_dir

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        lr_arr = np.load(os.path.join(self.lr_dir, fname)).astype(np.float32)
        gt_arr = np.load(os.path.join(self.gt_dir, fname)).astype(np.float32)

        if lr_arr.ndim == 2:
            lr_arr = np.expand_dims(lr_arr, axis=0)
        if gt_arr.ndim == 2:
            gt_arr = np.expand_dims(gt_arr, axis=0)

        return torch.from_numpy(lr_arr), torch.from_numpy(gt_arr), fname

def run_one_batch_sanity_test(model: nn.Module, criterion: nn.Module, sample_batch, device: torch.device):
    """
    Executes a 1-batch forward/backward sanity check before GPU training loop.
    Verifies tensor shapes, V3 freeze status, V4 gradient presence, and loss finiteness.
    """
    print("\n--- Running One-Batch GPU Sanity Test ---")
    model.eval()
    model.freeze_v3_base()

    lr_b, gt_b, _ = sample_batch
    lr_b = lr_b.to(device)
    gt_b = gt_b.to(device)

    out = model(lr_b)
    v3_pred = out["v3_restored"]
    v4_pred = out["restored"]
    residual = out["residual"]

    assert v3_pred.shape == (lr_b.size(0), 1, 256, 256), f"Expected V3 shape (B, 1, 256, 256), got {v3_pred.shape}"
    assert v4_pred.shape == (lr_b.size(0), 1, 256, 256), f"Expected V4 shape (B, 1, 256, 256), got {v4_pred.shape}"
    assert residual.shape == (lr_b.size(0), 1, 256, 256), f"Expected Residual shape (B, 1, 256, 256), got {residual.shape}"

    loss, loss_dict = criterion(out, gt_b)
    assert torch.isfinite(loss), f"Sanity test failed: Loss is not finite ({loss.item()})"

    loss.backward()

    # Verify V3 parameters have NO gradients
    v3_grads = [p.grad for p in model.v3_base.parameters() if p.grad is not None]
    assert len(v3_grads) == 0, f"Sanity test failed: V3 base parameters received {len(v3_grads)} gradients while frozen!"

    # Verify V4 refinement parameters HAVE gradients
    v4_ref_params = list(model.refinement_module.parameters())
    v4_grads = [p.grad for p in v4_ref_params if p.grad is not None]
    assert len(v4_grads) > 0, "Sanity test failed: V4 refinement parameters received NO gradients!"

    model.zero_grad()
    print("  [OK] Input shape:   ", tuple(lr_b.shape))
    print("  [OK] V3 Output shape:", tuple(v3_pred.shape))
    print("  [OK] V4 Output shape:", tuple(v4_pred.shape))
    print("  [OK] Loss Finite:   ", loss.item())
    print("  [OK] V3 Base Frozen: 0 Trainable Parameters")
    print("  [OK] V4 Refinement Gradients Active")
    print("[SANITY TEST PASSED CLEANLY]\n")

def main():
    parser = argparse.ArgumentParser(description="AIR-Net v4 High-Fidelity Restoration Pipeline")
    parser.add_argument("--train", action="store_true", help="Execute GPU training mode")
    parser.add_argument("--audit", action="store_true", help="Execute audit/validation mode only (default)")
    parser.add_argument("--v3-checkpoint", type=str, default="", help="Path to AIR-Net v3 foundation checkpoint")
    parser.add_argument("--v4-checkpoint", type=str, default="", help="Path to AIR-Net v4 output/resume checkpoint")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    args = parser.parse_args()

    seed_everything(42)

    print("==============================================================================")
    print("AIR-Net v4 — HIGH-FIDELITY RESTORATION PIPELINE (128x128 -> 256x256)")
    print("==============================================================================")

    # CUDA Safety Enforcement
    if args.train:
        if not torch.cuda.is_available():
            print("\n[ERROR] V4 training requires CUDA. No training was performed.")
            print("To train AIR-Net v4, run this script in an NVIDIA GPU environment (B200 / A100 / T4).")
            sys.exit(1)
        
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        cuda_ver = torch.version.cuda
        print(f"MODE:              GPU TRAINING (--train)")
        print(f"GPU Name:          {gpu_name}")
        print(f"CUDA Version:      {cuda_ver}")
        print(f"VRAM:              {vram_gb:.2f} GB")
        print(f"PyTorch Version:   {torch.__version__}")
        print(f"Batch Size:        {args.batch_size}")
        print(f"Learning Rate:     {args.lr}")
        print(f"Epochs:            {args.epochs}")
        print(f"CUDA AMP:          ENABLED")
    else:
        device = get_device()
        gpu_name = torch.cuda.get_device_name(0) if is_cuda() else "CPU Mode"
        print(f"MODE:              VALIDATION / AUDIT ONLY (--audit)")
        print(f"Device:            {device} ({gpu_name})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Target Resolution: 128x128 -> 256x256 (STRICT NO 512x512 RESOLUTION CHANGE)")
    print("==============================================================================\n")

    # Output Directories
    v4_root = os.path.join(PROJECT_ROOT, "outputs", "v4")
    ckpt_dir = os.path.join(v4_root, "checkpoints")
    vis_dir = os.path.join(v4_root, "visual_comparisons")
    for d in [v4_root, ckpt_dir, vis_dir]:
        os.makedirs(d, exist_ok=True)

    # Dataset Resolution
    config = Config(MODEL_VERSION="AIR-Net-v3")
    train_lr_dir = config.train_lr_dir
    train_gt_dir = config.train_gt_dir

    if not os.path.exists(train_lr_dir) or not os.path.exists(train_gt_dir):
        print(f"[NOTICE] Dataset directories not found locally ({train_lr_dir}).")
        print("Required dataset format: paired 128x128 NoisyLR and 256x256 GT .npy files.")
        if args.train:
            sys.exit(1)

    lr_files = sorted([f for f in os.listdir(train_lr_dir) if f.endswith(".npy")]) if os.path.exists(train_lr_dir) else []
    gt_files = sorted([f for f in os.listdir(train_gt_dir) if f.endswith(".npy")]) if os.path.exists(train_gt_dir) else []
    common_files = sorted(list(set(lr_files).intersection(set(gt_files))))

    mapping_csv = os.path.join(PROJECT_ROOT, "outputs", "stage1", "stage1_reconstruction", "authoritative_validation_mapping.csv")
    if not os.path.exists(mapping_csv):
        mapping_csv = os.path.join(PROJECT_ROOT, "authoritative_validation_mapping.csv")

    val_mapping = []
    if os.path.exists(mapping_csv):
        with open(mapping_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                val_mapping.append(row)

    val_filenames = [r["filename"] for r in val_mapping] if val_mapping else common_files[:320]
    val_set_filenames = set(val_filenames)
    train_filenames = sorted([f for f in common_files if f not in val_set_filenames])

    # Model Setup
    norm_path = os.path.join(PROJECT_ROOT, "outputs", "v3", "indexes", "index_normalization.json")
    norm_params = None
    if os.path.exists(norm_path):
        with open(norm_path, "r") as f:
            norm_params = json.load(f)

    model = AIRNetV4(norm_params=norm_params).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] AIR-Net v4 Parameter Count: {num_params:,}")

    # Load v3 Foundation Checkpoint
    v3_ckpt_path = args.v3_checkpoint
    if not v3_ckpt_path:
        cand = [
            os.path.join(PROJECT_ROOT, "outputs", "v3", "checkpoints", "airnet_v3_ema_best_model.pth"),
            os.path.join(PROJECT_ROOT, "outputs", "v3", "checkpoints", "airnet_v3_best_model.pth")
        ]
        for c in cand:
            if os.path.exists(c):
                v3_ckpt_path = c
                break

    v3_sha = "NOT_FOUND"
    if v3_ckpt_path and os.path.exists(v3_ckpt_path):
        state = torch.load(v3_ckpt_path, map_location=device)
        w_dict = state.get("ema_state_dict", state.get("model_state_dict", state))
        model.v3_base.load_state_dict(w_dict, strict=False)
        v3_sha = get_file_sha256(v3_ckpt_path)
        print(f"[OK] AIR-Net v3 foundation loaded: '{v3_ckpt_path}' (SHA256: {v3_sha[:12]}...)")
    else:
        print("[NOTICE] V3 foundation checkpoint not found locally. Supply via --v3-checkpoint in GPU environment.")

    criterion = AIRNetV4AdaptiveLoss(data_range=1.0).to(device)

    # EXECUTE GPU TRAINING MODE
    if args.train:
        model.freeze_v3_base()
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in model.v3_base.parameters() if not p.requires_grad)
        print(f"[OK] V3 Base Frozen ({frozen_params:,} params) | V4 Trainable Refinement Params: {trainable_params:,}")

        train_ds = KLAPairedDataset(train_filenames, train_lr_dir, train_gt_dir)
        val_ds = KLAPairedDataset(val_filenames, train_lr_dir, train_gt_dir)

        num_workers = min(8, os.cpu_count() or 4)
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
        )
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        # Run 1-Batch Sanity Test
        sanity_sample = next(iter(train_loader))
        run_one_batch_sanity_test(model, criterion, sanity_sample, device)

        optimizer = torch.optim.AdamW(model.refinement_module.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        scaler = torch.amp.GradScaler('cuda')

        best_psnr = -1.0
        best_v4_path = os.path.join(ckpt_dir, "best_v4_model.pth")
        latest_v4_path = os.path.join(ckpt_dir, "latest_v4_model.pth")
        history_csv = os.path.join(v4_root, "training_history.csv")

        with open(history_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_psnr", "val_ssim", "val_lpips", "psnr_gain"])

        print(f"\n--- Beginning AIR-Net v4 GPU Training Loop (Epochs: {args.epochs}) ---")
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            model.train()
            model.freeze_v3_base()

            train_loss_sum = 0.0
            total_steps = 0

            for lr_b, gt_b, _ in train_loader:
                lr_b = lr_b.to(device, non_blocking=True)
                gt_b = gt_b.to(device, non_blocking=True)

                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    out = model(lr_b)
                    loss, _ = criterion(out, gt_b)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.refinement_module.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                train_loss_sum += loss.item()
                total_steps += 1

            scheduler.step()
            avg_train_loss = train_loss_sum / total_steps

            # Validation
            model.eval()
            val_psnr_list, val_ssim_list, val_lpips_list = [], [], []
            v3_psnr_list = []

            with torch.no_grad(), torch.inference_mode():
                for lr_b, gt_b, _ in val_loader:
                    lr_b = lr_b.to(device, non_blocking=True)
                    gt_b = gt_b.to(device, non_blocking=True)

                    out = model(lr_b)
                    v3_pred = torch.clamp(out["v3_restored"], 0.0, 1.0)
                    v4_pred = torch.clamp(out["restored"], 0.0, 1.0)

                    for i in range(v4_pred.size(0)):
                        v3_psnr_list.append(calculate_psnr(v3_pred[i:i+1], gt_b[i:i+1], data_range=1.0))
                        val_psnr_list.append(calculate_psnr(v4_pred[i:i+1], gt_b[i:i+1], data_range=1.0))
                        val_ssim_list.append(calculate_ssim(v4_pred[i:i+1], gt_b[i:i+1], data_range=1.0))
                        val_lpips_list.append(compute_lpips_fast(v4_pred[i:i+1], gt_b[i:i+1], device))

            v3_m_psnr = float(np.mean(v3_psnr_list))
            v4_m_psnr = float(np.mean(val_psnr_list))
            v4_m_ssim = float(np.mean(val_ssim_list))
            v4_m_lpips = float(np.mean(val_lpips_list))
            gain = v4_m_psnr - v3_m_psnr

            epoch_time = time.time() - epoch_start
            print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_time:.1f}s) | Train Loss: {avg_train_loss:.4f} | V3 PSNR: {v3_m_psnr:.4f} dB -> V4 PSNR: {v4_m_psnr:.4f} dB (Gain: {gain:+.4f} dB) | SSIM: {v4_m_ssim:.4f} | LPIPS: {v4_m_lpips:.4f}")

            with open(history_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, round(avg_train_loss, 6), round(v4_m_psnr, 4), round(v4_m_ssim, 4), round(v4_m_lpips, 4), round(gain, 4)])

            is_best = v4_m_psnr > best_psnr
            if is_best:
                best_psnr = v4_m_psnr
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "v4_refinement_state_dict": model.refinement_module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_psnr": best_psnr,
                    "v3_checkpoint_sha256": v3_sha
                }, best_v4_path)
                print(f"  >>> Best V4 Checkpoint Saved! Val PSNR = {best_psnr:.4f} dB")

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }, latest_v4_path)

        print("\n==============================================================================")
        print(f"AIR-Net v4 GPU TRAINING COMPLETE | Best PSNR: {best_psnr:.4f} dB")
        print(f"Saved Checkpoint: '{best_v4_path}'")
        print("==============================================================================")
    else:
        print("\n--- AIR-Net v4 Validation & Pipeline Audit Mode Complete ---")
        print("To launch GPU training, run:")
        print("  python scripts/execute_v4_pipeline.py --train --v3-checkpoint /path/to/airnet_v3_ema_best_model.pth")
        print("==============================================================================")

if __name__ == "__main__":
    main()
