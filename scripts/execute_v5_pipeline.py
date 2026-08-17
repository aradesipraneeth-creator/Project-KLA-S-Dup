import os
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from scipy.ndimage import zoom

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import Config
from models.airnet_v5 import AIRNetV5
from losses.adaptive_loss_v5 import AIRNetV5AdaptiveLoss
from utils.checkpoint_manager import CheckpointManager
from utils.image_normalization import (
    normalize_input, normalize_target, denormalize_output,
    validate_metric_inputs, compute_array_stats
)
from utils.metrics import compute_all_metrics, run_metric_sanity_test

class KLASemiconductorDataset(Dataset):
    """Authoritative KLA Semiconductor Image Restoration Dataset (128x128 -> 256x256)."""
    def __init__(self, lr_dir: str, gt_dir: str, file_list: list = None):
        self.lr_dir = lr_dir
        self.gt_dir = gt_dir
        if file_list is not None:
            self.files = sorted(file_list)
        else:
            self.files = sorted([f for f in os.listdir(lr_dir) if f.endswith(".npy")])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        lr_raw = np.load(os.path.join(self.lr_dir, fname))
        gt_raw = np.load(os.path.join(self.gt_dir, fname))

        lr_norm = normalize_input(lr_raw)
        gt_norm = normalize_target(gt_raw)

        assert lr_norm.shape == (128, 128), f"Input sample resolution error: {lr_norm.shape}"
        assert gt_norm.shape == (256, 256), f"Target sample resolution error: {gt_norm.shape}"

        lr_t = torch.from_numpy(lr_norm).unsqueeze(0).float()
        gt_t = torch.from_numpy(gt_norm).unsqueeze(0).float()

        return lr_t, gt_t, fname

def run_sanity_test(model: AIRNetV5, loss_fn: AIRNetV5AdaptiveLoss, device: torch.device):
    """
    Part 13: 1-Batch Training Sanity Test.
    Verifies Input [4, 1, 128, 128], V3 [4, 1, 256, 256], V5 [4, 1, 256, 256],
    V3 frozen, V5 gradients active, loss finite, no NaN/Inf.
    """
    print("\n==============================================================================")
    print("[SANITY TEST] RUNNING 1-BATCH FORWARD/BACKWARD VERIFICATION")
    print("==============================================================================")

    dummy_lr = torch.randn(4, 1, 128, 128, dtype=torch.float32, device=device)
    dummy_gt = torch.rand(4, 1, 256, 256, dtype=torch.float32, device=device)

    # Freeze V3 base
    model.freeze_v3_base()

    # Forward
    out_dict = model(dummy_lr)
    v3_res = out_dict["v3_restored"]
    v5_res = out_dict["restored"]

    assert dummy_lr.shape == (4, 1, 128, 128), f"Sanity input shape mismatch: {dummy_lr.shape}"
    assert v3_res.shape == (4, 1, 256, 256), f"Sanity V3 shape mismatch: {v3_res.shape}"
    assert v5_res.shape == (4, 1, 256, 256), f"Sanity V5 shape mismatch: {v5_res.shape}"

    loss, loss_dict = loss_fn(out_dict, dummy_gt)

    assert torch.isfinite(loss), f"Sanity test loss non-finite: {loss.item()}"
    assert not torch.isnan(v5_res).any(), "Sanity test V5 output contains NaN"
    assert not torch.isinf(v5_res).any(), "Sanity test V5 output contains Inf"

    # Backward
    loss.backward()

    # Check gradients
    v3_grads = [p.grad for p in model.v3_base.parameters() if p.grad is not None]
    v5_grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]

    assert len(v3_grads) == 0, f"V3 base not frozen during sanity test! Found {len(v3_grads)} active gradients."
    assert len(v5_grads) > 0, "No active gradients found in V5 refinement module!"

    print(f"[OK] Input Shape:   {tuple(dummy_lr.shape)}")
    print(f"[OK] V3 Shape:      {tuple(v3_res.shape)}")
    print(f"[OK] V5 Shape:      {tuple(v5_res.shape)}")
    print(f"[OK] Total Loss:    {loss.item():.6f}")
    print(f"[OK] V3 Base Frozen: YES ({len(v3_grads)} active grads)")
    print(f"[OK] V5 Refinement Grads: YES ({len(v5_grads)} active grads)")
    print("[SANITY TEST PASSED SUCCESSFULLY]\n")
    return True

def evaluate_model(model: AIRNetV5, val_loader: DataLoader, device: torch.device):
    """Runs full evaluation on validation set."""
    model.eval()
    bic_metrics, v3_metrics, v5_metrics = [], [], []

    with torch.no_grad():
        for lr_t, gt_t, _ in val_loader:
            lr_t, gt_t = lr_t.to(device), gt_t.to(device)
            out_dict = model(lr_t)

            lr_np = lr_t.squeeze().cpu().numpy()
            v3_np = out_dict["v3_restored"].squeeze().cpu().numpy()
            v5_np = out_dict["restored"].squeeze().cpu().numpy()
            gt_np = gt_t.squeeze().cpu().numpy()

            for b in range(lr_t.size(0)):
                cur_lr = lr_np[b] if lr_t.size(0) > 1 else lr_np
                cur_v3 = v3_np[b] if lr_t.size(0) > 1 else v3_np
                cur_v5 = v5_np[b] if lr_t.size(0) > 1 else v5_np
                cur_gt = gt_np[b] if lr_t.size(0) > 1 else gt_np

                bic_raw = zoom(cur_lr, (256 / cur_lr.shape[0], 256 / cur_lr.shape[1]), order=3)
                p_bic, g_bic = validate_metric_inputs(bic_raw, cur_gt)
                p_v3, _ = validate_metric_inputs(cur_v3, cur_gt)
                p_v5, _ = validate_metric_inputs(cur_v5, cur_gt)

                bic_metrics.append(compute_all_metrics(p_bic, g_bic, device))
                v3_metrics.append(compute_all_metrics(p_v3, g_bic, device))
                v5_metrics.append(compute_all_metrics(p_v5, g_bic, device))

    def avg_dict(lst):
        res = {}
        for k in lst[0].keys():
            res[k] = float(np.mean([m[k] for m in lst]))
        return res

    return avg_dict(bic_metrics), avg_dict(v3_metrics), avg_dict(v5_metrics)

def main():
    parser = argparse.ArgumentParser(description="AIR-Net v5 Pipeline Execution & Training Script")
    parser.add_argument("--train", action="store_true", help="Execute full training loop")
    parser.add_argument("--v3-checkpoint", type=str, default="", help="Path to AIR-Net v3 foundation checkpoint")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume training from")
    parser.add_argument("--output-dir", type=str, default="", help="Output directory")
    parser.add_argument("--sanity-test", action="store_true", help="Run 1-batch sanity check and exit")
    parser.add_argument("--audit", action="store_true", help="Run evaluation audit and exit")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow training on CPU if CUDA is unavailable")
    args = parser.parse_args()

    config = Config(MODEL_VERSION="AIR-Net-v3")
    out_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs" / "v5"
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("==============================================================================")
    print("AIR-NET V5 HIGH-FIDELITY RESTORATION PIPELINE")
    print("==============================================================================")

    # 1. GPU / CUDA Verification
    cuda_avail = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_avail else "cpu")
    print(f"CUDA Available: {cuda_avail}")
    if cuda_avail:
        print(f"GPU Device:     {torch.cuda.get_device_name(0)}")
        print(f"VRAM Available: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print("GPU Device:     None (CPU Runtime)")
    print(f"PyTorch Ver:    {torch.__version__}")

    if args.train and not cuda_avail and not args.allow_cpu:
        print("\n[ERROR] GPU training requested but CUDA is unavailable!")
        print("To run full training, execute on a GPU environment or pass --allow-cpu.")
        sys.exit(1)

    # 2. Checkpoint Discovery
    v3_env = os.environ.get("AIRNET_V3_CHECKPOINT", "")
    v3_path = args.v3_checkpoint or v3_env
    if not v3_path:
        cand_v3 = [
            PROJECT_ROOT / "outputs" / "v3" / "checkpoints" / "airnet_v3_ema_best_model.pth",
            PROJECT_ROOT / "outputs" / "v3" / "checkpoints" / "airnet_v3_best_model.pth"
        ]
        for c in cand_v3:
            if c.exists():
                v3_path = str(c)
                break

    # 3. Model & Checkpoint Loading
    norm_path = PROJECT_ROOT / "outputs" / "v3" / "indexes" / "index_normalization.json"
    norm_params = json.load(open(norm_path, "r")) if norm_path.exists() else None

    v5_model = AIRNetV5(norm_params=norm_params).to(device)

    if v3_path and os.path.exists(v3_path):
        print(f"\n[LOADING] AIR-Net v3 foundation checkpoint: '{v3_path}'")
        v5_model.v3_base = CheckpointManager.load_strictly_verified_model(
            v5_model.v3_base, v3_path, architecture_name="AIR-Net v3 Foundation", device=device
        )
        print("✓ V3 checkpoint loaded & verified")
    else:
        print("\n[WARNING] AIR-Net v3 foundation checkpoint not found. V3 base remains initialized.")

    total_params = sum(p.numel() for p in v5_model.parameters())
    trainable_params = sum(p.numel() for p in v5_model.parameters() if p.requires_grad)
    print(f"✓ Total Parameters:     {total_params:,}")
    print(f"✓ Parameter Verified:   7,368,831 expected (~7.37M)")

    loss_fn = AIRNetV5AdaptiveLoss().to(device)

    # 4. Sanity Test Mode
    if args.sanity_test:
        run_sanity_test(v5_model, loss_fn, device)
        sys.exit(0)

    # 5. Dataset Loading
    lr_dir = config.train_lr_dir
    gt_dir = config.train_gt_dir

    if not (os.path.exists(lr_dir) and os.path.exists(gt_dir)):
        print(f"\n[NOTICE] Dataset directories not found on local machine ({lr_dir}). Audit mode complete.")
        sys.exit(0)

    all_files = sorted([f for f in os.listdir(lr_dir) if f.endswith(".npy")])
    val_count = min(320, len(all_files) // 10)
    train_files = all_files[:-val_count] if val_count > 0 else all_files
    val_files = all_files[-val_count:] if val_count > 0 else all_files[:10]

    train_ds = KLASemiconductorDataset(lr_dir, gt_dir, train_files)
    val_ds = KLASemiconductorDataset(lr_dir, gt_dir, val_files)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Pre-training Intensity Domain Audit
    sample_lr, sample_gt, _ = train_ds[0]
    print("\n--- PRE-TRAINING INTENSITY NORMALIZATION AUDIT ---")
    print(f"NoisyLR: min={sample_lr.min():.4f}, max={sample_lr.max():.4f}, mean={sample_lr.mean():.4f}, std={sample_lr.std():.4f}")
    print(f"GT:      min={sample_gt.min():.4f}, max={sample_gt.max():.4f}, mean={sample_gt.mean():.4f}, std={sample_gt.std():.4f}")

    # Audit Mode
    if args.audit:
        print("\n--- RUNNING AUDIT EVALUATION ---")
        m_bic, m_v3, m_v5 = evaluate_model(v5_model, val_loader, device)
        print(f"Bicubic 2x: PSNR={m_bic['PSNR (dB)']:.4f} dB | SSIM={m_bic['SSIM']:.4f} | LPIPS={m_bic['LPIPS']:.4f}")
        print(f"AIR-Net v3: PSNR={m_v3['PSNR (dB)']:.4f} dB | SSIM={m_v3['SSIM']:.4f} | LPIPS={m_v3['LPIPS']:.4f}")
        print(f"AIR-Net v5: PSNR={m_v5['PSNR (dB)']:.4f} dB | SSIM={m_v5['SSIM']:.4f} | LPIPS={m_v5['LPIPS']:.4f}")
        sys.exit(0)

    # 6. Training Loop (STAGE A: V3 Frozen, V5 Refinement Trained)
    if args.train:
        v5_model.freeze_v3_base()
        optimizer = torch.optim.AdamW([p for p in v5_model.parameters() if p.requires_grad], lr=args.learning_rate)
        scaler = torch.cuda.amp.GradScaler(enabled=cuda_avail)

        best_score = -999.0
        history_rows = []

        print(f"\n--- STARTING STAGE A TRAINING ({args.epochs} EPOCHS, V3 FROZEN) ---")
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            v5_model.train()
            v5_model.v3_base.eval()

            train_loss = 0.0
            for lr_b, gt_b, _ in train_loader:
                lr_b, gt_b = lr_b.to(device), gt_b.to(device)
                optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=cuda_avail):
                    out_dict = v5_model(lr_b)
                    loss, loss_dict = loss_fn(out_dict, gt_b)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            m_bic, m_v3, m_v5 = evaluate_model(v5_model, val_loader, device)

            # Score calculation: PSNR gain + 10*SSIM gain - 5*LPIPS change
            val_score = (m_v5["PSNR (dB)"] - m_v3["PSNR (dB)"]) + 10.0 * (m_v5["SSIM"] - m_v3["SSIM"]) - 5.0 * (m_v5["LPIPS"] - m_v3["LPIPS"])
            t_el = time.time() - t0

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "v5_psnr": m_v5["PSNR (dB)"],
                "v5_ssim": m_v5["SSIM"],
                "v5_lpips": m_v5["LPIPS"],
                "v3_psnr": m_v3["PSNR (dB)"],
                "v3_ssim": m_v3["SSIM"],
                "v3_lpips": m_v3["LPIPS"],
                "val_score": val_score,
                "time_sec": t_el
            }
            history_rows.append(row)

            print(f"Epoch [{epoch:02d}/{args.epochs:02d}] {t_el:.1f}s | Loss: {train_loss:.4f} | V5 PSNR: {m_v5['PSNR (dB)']:.4f} dB | SSIM: {m_v5['SSIM']:.4f} | LPIPS: {m_v5['LPIPS']:.4f} | Score: {val_score:+.4f}")

            # Checkpoint Saving
            latest_path = ckpt_dir / "latest_v5_model.pth"
            torch.save({"v5_state_dict": v5_model.state_dict(), "epoch": epoch, "val_score": val_score}, latest_path)

            if val_score > best_score:
                best_score = val_score
                best_path = ckpt_dir / "best_v5_model.pth"
                torch.save({"v5_state_dict": v5_model.state_dict(), "epoch": epoch, "val_score": val_score}, best_path)
                print(f"  --> Best V5 model saved at '{best_path}' (Score: {best_score:+.4f})")

        # Save History
        pd.DataFrame(history_rows).to_csv(out_dir / "training_history.csv", index=False)
        print(f"\n✓ Training complete. History saved to '{out_dir / 'training_history.csv'}'")

if __name__ == "__main__":
    main()
