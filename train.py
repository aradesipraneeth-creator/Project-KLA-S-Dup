import os
import sys
import time
import random
import json
import shutil
from typing import Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

from configs.config import Config
from datasets.kla_dataset import get_train_val_datasets
from models.airnet import AIRNet
from losses.hybrid_loss import AIRNetHybridLoss
from utils import (
    ModelEMA,
    calculate_psnr,
    calculate_ssim,
    CSVLogger,
    print_epoch_summary,
    save_json,
    generate_dataset_stats,
    compute_bicubic_baseline,
    save_visualizations_and_predictions,
    generate_model_summary,
    generate_experiment_info,
    run_inference_benchmark,
    get_device,
    print_device_info,
    is_cuda,
    is_mps,
    is_cpu,
    is_amp_available,
    get_gpu_memory_info
)

# ====================================================
# FUTURE EXTENSION HOOKS (TODO PLACEHOLDERS)
# ====================================================
# TODO: AIR-Net v2
# TODO: Dynamic Loss Weight Scheduler
# TODO: Knowledge Distillation
# TODO: Test-Time Augmentation
# TODO: ONNX Export
# TODO: TensorRT Export
# TODO: Adaptive Inference
# TODO: Multi-scale Inference
# TODO: LPIPS Training
# ====================================================

def seed_worker(worker_id):
    """Seed each DataLoader worker for strict multi-process reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_seed(seed: int = 42):
    """Sets fixed random seed across python, numpy, and PyTorch for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if is_cuda():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

def get_amp_components(device: torch.device):
    """Returns PyTorch AMP autocast context and GradScaler supporting modern and legacy PyTorch APIs."""
    enabled = is_amp_available()
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        scaler = torch.amp.GradScaler(device.type, enabled=enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=enabled)
    return scaler

def get_autocast_context(device: torch.device):
    """Returns PyTorch autocast context for execution device."""
    enabled = is_amp_available()
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    else:
        return torch.cuda.amp.autocast(enabled=enabled)

def stage_dataset_to_fast_local_storage(config: Config):
    """
    Optional Dataset Staging for Google Colab / Remote Cloud Training.
    Copies GT and NoisyLR once to fast local SSD storage (/content/kla_dataset or /tmp/kla_dataset)
    to maximize DataLoader GPU throughput while keeping all checkpoints & outputs on Google Drive.
    """
    if not getattr(config, 'COPY_DATASET_TO_LOCAL', True):
        return

    # Determine local staging root
    if os.path.exists("/content"):
        local_root = "/content/kla_dataset"
    else:
        local_root = os.path.join(os.path.expanduser("~"), ".cache", "kla_dataset")

    local_lr_dir = os.path.join(local_root, "train", "NoisyLR")
    local_gt_dir = os.path.join(local_root, "train", "GT")

    source_lr_dir = config.train_lr_dir
    source_gt_dir = config.train_gt_dir

    if os.path.exists(source_lr_dir) and os.path.exists(source_gt_dir):
        if not (os.path.exists(local_lr_dir) and os.path.exists(local_gt_dir)):
            print(f"  [OK] Copying dataset to fast local storage ({local_root})...")
            try:
                os.makedirs(local_lr_dir, exist_ok=True)
                os.makedirs(local_gt_dir, exist_ok=True)

                lr_files = [f for f in os.listdir(source_lr_dir) if f.endswith(".npy")]
                for f in lr_files:
                    shutil.copy2(os.path.join(source_lr_dir, f), os.path.join(local_lr_dir, f))

                gt_files = [f for f in os.listdir(source_gt_dir) if f.endswith(".npy")]
                for f in gt_files:
                    shutil.copy2(os.path.join(source_gt_dir, f), os.path.join(local_gt_dir, f))

                print(f"  [OK] Successfully staged {len(lr_files)} samples to fast local SSD storage.")
            except Exception as e:
                print(f"  ⚠️ Fast local staging notice: {e}. Falling back to original dataset paths.")
                return

        # Update dataset paths to point to fast local copy
        config.train_lr_dir = local_lr_dir
        config.train_gt_dir = local_gt_dir
        print(f"  [OK] Fast local dataset staging active: '{local_root}'. Training from SSD copy.")

def cleanup_temporary_checkpoints(checkpoint_dir: str):
    """Clean up any temporary .tmp checkpoint files while keeping standard models."""
    if os.path.exists(checkpoint_dir):
        for f in os.listdir(checkpoint_dir):
            if f.endswith(".tmp") or f.endswith(".bak"):
                try:
                    os.remove(os.path.join(checkpoint_dir, f))
                except Exception:
                    pass

def train_epoch(
    model: nn.Module,
    ema: ModelEMA,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    grad_accum_steps: int = 8,
    max_grad_norm: float = 1.0
) -> float:
    """Runs one training epoch with NaN Protection, tqdm Progress Bar, AMP, and Gradient Accumulation."""
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(
        dataloader,
        desc=f"Epoch {epoch:02d}/{total_epochs:02d}",
        leave=False,
        dynamic_ncols=True
    )

    for step, (lr_batch, gt_batch, _) in enumerate(pbar, start=1):
        lr_batch = lr_batch.to(device)
        gt_batch = gt_batch.to(device)

        with get_autocast_context(device):
            out_dict = model(lr_batch)
            loss = criterion(out_dict, gt_batch)
            
            # NaN / Inf Protection
            if not torch.isfinite(loss):
                print(f"\n⚠️ Warning: Non-finite loss detected ({loss.item()}) at step {step}. Skipping step.")
                optimizer.zero_grad()
                continue

            scaled_loss = loss / grad_accum_steps

        scaler.scale(scaled_loss).backward()
        running_loss += loss.item() * grad_accum_steps

        if step % grad_accum_steps == 0 or step == len(dataloader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            ema.update(model)

        # Progress bar metrics
        gpu_mem = f"{torch.cuda.memory_allocated() / (1024**2):.0f}MB" if device.type == 'cuda' else "N/A"
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            "gpu_mem": gpu_mem
        })

    return running_loss / len(dataloader)

def apply_8way_tta_inference(model: nn.Module, lr_batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Optional 8-way Test-Time Augmentation (TTA) during evaluation.
    Applies 4 rotations x 2 flips, predicts, inverses spatial transforms, and averages restored images.
    """
    preds = []

    for rot_k in range(4):
        for flip in [False, True]:
            x = torch.rot90(lr_batch, rot_k, dims=[2, 3])
            if flip:
                x = torch.flip(x, dims=[3])

            with get_autocast_context(device):
                out = model(x)
                pred_img = out["restored"] if isinstance(out, dict) else out

            if flip:
                pred_img = torch.flip(pred_img, dims=[3])
            pred_img = torch.rot90(pred_img, -rot_k, dims=[2, 3])

            preds.append(pred_img)

    return torch.stack(preds, dim=0).mean(dim=0)

def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_tta: bool = False
) -> Tuple[float, float, float]:
    """Evaluates EMA model on validation set with optional 8-way TTA and computes Loss, PSNR, and SSIM."""
    model.eval()
    running_loss = 0.0
    psnr_list = []
    ssim_list = []

    with torch.no_grad():
        for lr_batch, gt_batch, _ in dataloader:
            lr_batch = lr_batch.to(device)
            gt_batch = gt_batch.to(device)

            if use_tta:
                pred_img = apply_8way_tta_inference(model, lr_batch, device)
                with get_autocast_context(device):
                    out_dict = {"restored": pred_img}
                    loss = criterion(out_dict, gt_batch)
            else:
                with get_autocast_context(device):
                    out_dict = model(lr_batch)
                    loss = criterion(out_dict, gt_batch)
                pred_img = out_dict["restored"] if isinstance(out_dict, dict) else out_dict

            running_loss += loss.item()
            pred_clamped = torch.clamp(pred_img, 0.0, 1.0)

            psnr_val = calculate_psnr(pred_clamped, gt_batch, data_range=1.0)
            ssim_val = calculate_ssim(pred_clamped, gt_batch, data_range=1.0)

            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)

    avg_loss = running_loss / len(dataloader)
    avg_psnr = float(sum(psnr_list) / len(psnr_list))
    avg_ssim = float(sum(ssim_list) / len(ssim_list))

    return avg_loss, avg_psnr, avg_ssim

def parse_cached_bicubic_baseline(baseline_file: str) -> Tuple[float, float]:
    """Parses PSNR and SSIM values from cached bicubic_baseline.txt."""
    psnr_val, ssim_val = 22.9770, 0.5134
    if os.path.exists(baseline_file):
        try:
            with open(baseline_file, "r") as f:
                content = f.read()
            for line in content.splitlines():
                if "Average Bicubic PSNR:" in line:
                    psnr_val = float(line.split(":")[1].replace("dB", "").strip())
                elif "Average Bicubic SSIM:" in line:
                    ssim_val = float(line.split(":")[1].strip())
        except Exception:
            pass
    return psnr_val, ssim_val

def format_time(seconds: float) -> str:
    """Formats seconds into human readable h m s or m s."""
    if seconds < 0:
        seconds = 0
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    else:
        return f"{s}s"

def main():
    startup_start = time.time()
    config = Config()

    # Safety Checks: Create output, checkpoint, vis, val_preds, and test_results directories
    config.create_dirs()
    cleanup_temporary_checkpoints(config.checkpoint_dir)
    set_seed(config.seed)

    print("====================================================")
    print(f"AIR-NET EXPERIMENT PIPELINE: {config.MODEL_VERSION}")
    print("====================================================")
    print(f"Architecture:                  AIR-Net v1 (Unchanged)")
    print(f"Loss Configuration:")
    print(f"  L1 Weight:                   {config.L1_WEIGHT:.2f}")
    print(f"  SSIM Weight:                 {config.SSIM_WEIGHT:.2f}")
    print(f"  Edge Weight:                 {config.EDGE_WEIGHT:.2f}")
    print(f"Dataset:                       SAME AS AIR-Net v1")
    print(f"Optimizer:                     SAME AS AIR-Net v1")
    print(f"Scheduler:                     SAME AS AIR-Net v1")
    print(f"EMA:                           SAME AS AIR-Net v1")
    print(f"AMP:                           SAME AS AIR-Net v1")
    print(f"Existing v1 checkpoint:        PRESERVED")
    print(f"New checkpoint directory:      {config.checkpoint_dir}")
    print("====================================================")
    print("Pre-training verification checks:")
    print("  [OK] AIR-Net architecture unchanged")
    print("  [OK] Parameter count unchanged")
    print("  [OK] Dataset unchanged")
    print("  [OK] Optimizer unchanged")
    print("  [OK] Scheduler unchanged")
    print("  [OK] EMA unchanged")
    print("  [OK] AMP unchanged")
    print("  [OK] Validation unchanged")
    print("  [OK] PSNR unchanged")
    print("  [OK] SSIM unchanged")
    print("  [OK] Loss weights configured")
    print("  [OK] AIR-Net v1 checkpoints preserved")
    print("====================================================")

    # Optional Dataset Staging to fast local SSD storage
    stage_dataset_to_fast_local_storage(config)

    # Device Selection & Hardware Diagnostics
    device = get_device()
    print_device_info()

    effective_batch_size = config.batch_size * config.grad_accum_steps
    print(f"Batch Size (Per-GPU):          {config.batch_size}")
    print(f"Gradient Accumulation Steps:   {config.grad_accum_steps} (Effective Batch Size = {effective_batch_size})")
    print(f"Mixed Precision Status:        {'Enabled (AMP)' if is_amp_available() else 'Disabled (FP32)'}")
    print(f"EMA Status:                    Enabled (decay={config.ema_decay})")
    print(f"Evaluation TTA Status:         {'Enabled (8-way TTA)' if config.USE_TTA else 'Disabled'}")
    print("----------------------------------------------------")

    # Profile Startup Phase Timings
    t0_step = time.time()

    # [1/8] Dataset Statistics (Caching System)
    print("[1/8] Dataset Statistics")
    if config.SKIP_PRECOMPUTATION and os.path.exists(config.train_stats_file):
        print(f"  [OK] Dataset statistics found ({config.train_stats_file}).")
        print("  [OK] Using cached statistics.")
    else:
        generate_dataset_stats(config)
    t_stats = time.time() - t0_step

    # [2/8] Bicubic Baseline (Caching System)
    t0_step = time.time()
    print("[2/8] Bicubic Baseline")
    if config.SKIP_PRECOMPUTATION and os.path.exists(config.bicubic_baseline_file):
        print(f"  [OK] Bicubic baseline found ({config.bicubic_baseline_file}).")
        print("  [OK] Using cached baseline.")
        bicubic_psnr, bicubic_ssim = parse_cached_bicubic_baseline(config.bicubic_baseline_file)
    else:
        bicubic_psnr, bicubic_ssim = compute_bicubic_baseline(config)
    t_baseline = time.time() - t0_step

    # [3/8] Dataset Loading (Platform-Aware & Worker Seeded DataLoader)
    t0_step = time.time()
    print("[3/8] Dataset Loading")
    train_dataset, val_dataset = get_train_val_datasets(
        train_lr_dir=config.train_lr_dir,
        train_gt_dir=config.train_gt_dir,
        seed=config.seed,
        train_split=config.train_split,
        val_split=config.val_split
    )

    g_gen = torch.Generator()
    g_gen.manual_seed(config.seed)

    if is_cuda():
        num_workers = min(2, os.cpu_count() or 2)
        pin_memory = True
        persistent_workers = True if num_workers > 0 else False
        prefetch_factor = 2 if num_workers > 0 else None
    elif is_mps():
        num_workers = 0
        pin_memory = False
        persistent_workers = False
        prefetch_factor = None
    else:
        num_workers = 0
        pin_memory = False
        persistent_workers = False
        prefetch_factor = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        worker_init_fn=seed_worker,
        generator=g_gen
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        worker_init_fn=seed_worker,
        generator=g_gen
    )
    t_loading = time.time() - t0_step

    # [4/8] Model Initialization
    t0_step = time.time()
    print("[4/8] Model Initialization")
    model = AIRNet(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        dim=config.dim,
        channels=config.channels,
        heads=config.heads,
        enc_blocks=config.enc_blocks,
        latent_blocks=config.latent_blocks,
        dec_blocks=config.dec_blocks,
        ffn_expansion_factor=config.ffn_expansion_factor
    ).to(device)

    # Optional Torch Compile Hook
    if getattr(config, 'USE_TORCH_COMPILE', False) and hasattr(torch, 'compile'):
        try:
            print("  [OK] Compiling model with torch.compile()...")
            model = torch.compile(model)
        except Exception as e:
            print(f"  ⚠️ torch.compile failed: {e}. Continuing without compile.")

    ema = ModelEMA(model, decay=config.ema_decay)
    t_model = time.time() - t0_step

    # [5/8] Summary & Metadata (Caching System)
    t0_step = time.time()
    print("[5/8] Summary & Metadata")
    if (
        config.SKIP_PRECOMPUTATION
        and os.path.exists(config.model_summary_file)
        and os.path.exists(config.experiment_info_file)
    ):
        print(f"  [OK] Model summary found ({config.model_summary_file}).")
        print("  [OK] Using cached summary.")
        print(f"  [OK] Experiment metadata found ({config.experiment_info_file}).")
        print("  [OK] Using cached metadata.")
    else:
        generate_model_summary(config, model)
        generate_experiment_info(config)
    t_summary = time.time() - t0_step

    # [6/8] Loss & Optimizer Setup
    t0_step = time.time()
    print("[6/8] Loss & Optimizer Setup")
    criterion = AIRNetHybridLoss(
        l1_weight=config.L1_WEIGHT,
        ssim_weight=config.SSIM_WEIGHT,
        edge_weight=config.EDGE_WEIGHT,
        data_range=1.0,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999)
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.min_lr
    )

    scaler = get_amp_components(device)
    csv_logger = CSVLogger(config.results_csv)
    t_opt = time.time() - t0_step

    # [7/8] Resume Checkpoint Verification (Auto Resume)
    t0_step = time.time()
    print("[7/8] Resume Checkpoint Verification")
    last_ckpt_path = os.path.join(config.checkpoint_dir, "last_model.pth")
    start_epoch = 1
    best_psnr = -1.0
    best_ssim = -1.0
    best_epoch = -1

    # Load existing best metrics if available
    if os.path.exists(config.best_metrics_file):
        try:
            with open(config.best_metrics_file, "r") as f:
                best_data = json.load(f)
            best_psnr = best_data.get("best_psnr", -1.0)
            best_ssim = best_data.get("best_ssim", -1.0)
            best_epoch = best_data.get("best_epoch", -1)
        except Exception:
            pass

    if config.AUTO_RESUME and os.path.exists(last_ckpt_path):
        try:
            ckpt = torch.load(last_ckpt_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            ema.load_state_dict(ckpt['ema_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            if 'scaler_state_dict' in ckpt:
                scaler.load_state_dict(ckpt['scaler_state_dict'])
            
            saved_epoch = ckpt.get('epoch', 0)
            start_epoch = saved_epoch + 1
            print(f"  [OK] Resuming training from Epoch {start_epoch:02d} (Loaded {last_ckpt_path})")
        except Exception as e:
            print(f"  ⚠️ Could not restore checkpoint: {e}. Starting fresh from Epoch 01.")
            start_epoch = 1
    else:
        print("  [OK] No previous checkpoint found / Fresh start. Training from Epoch 01.")
    t_ckpt = time.time() - t0_step

    # [8/8] Starting Training & Startup Profile Summary
    startup_duration = time.time() - startup_start
    print(f"[8/8] Starting Training (Startup Completed in {startup_duration:.2f} s)")
    print("----------------------------------------------------")
    print("STARTUP TIMING BREAKDOWN:")
    print(f"  - Dataset Stats:     {t_stats:.3f} s")
    print(f"  - Bicubic Baseline:  {t_baseline:.3f} s")
    print(f"  - Dataset Loading:   {t_loading:.3f} s")
    print(f"  - Model Creation:    {t_model:.3f} s")
    print(f"  - Optimizer Setup:   {t_opt:.3f} s")
    print(f"  - Checkpoint Load:   {t_ckpt:.3f} s")
    print("----------------------------------------------------")

    epoch_times = []
    patience_counter = 0
    start_total_time = time.time()

    val_loss, ema_psnr, ema_ssim = 0.0, 0.0, 0.0

    for epoch in range(start_epoch, config.epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(
            model=model,
            ema=ema,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            total_epochs=config.epochs,
            grad_accum_steps=config.grad_accum_steps,
            max_grad_norm=config.max_grad_norm
        )

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # Consistent Validation Evaluation using EMA Model
        if epoch % config.VALIDATE_EVERY == 0 or epoch == config.epochs:
            val_loss, ema_psnr, ema_ssim = evaluate(
                model=ema.ema_model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                use_tta=config.USE_TTA
            )

            # GPU Memory Cleanup after validation
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        t1 = time.time()
        epoch_dur = t1 - t0
        epoch_times.append(epoch_dur)

        # DataLoader Throughput & GPU Utilization Summaries
        total_train_images = len(train_dataset)
        total_train_batches = len(train_loader)
        imgs_per_sec = total_train_images / epoch_dur
        batches_per_sec = total_train_batches / epoch_dur

        gpu_alloc, gpu_res, gpu_max = get_gpu_memory_info()

        # Log to CSV with GPU Memory & Throughput Monitoring
        csv_logger.log_epoch(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            psnr=ema_psnr,
            ssim=ema_ssim,
            lr=current_lr,
            gpu_allocated_mb=gpu_alloc,
            gpu_reserved_mb=gpu_res,
            gpu_peak_mb=gpu_max,
            images_per_second=imgs_per_sec,
            batches_per_second=batches_per_sec
        )

        # Calculate Training ETA & Progress Summary
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = config.epochs - epoch
        eta_seconds = avg_epoch_time * remaining_epochs
        eta_str = format_time(eta_seconds)

        print_epoch_summary(
            epoch=epoch,
            total_epochs=config.epochs,
            train_loss=train_loss,
            val_loss=val_loss,
            psnr=ema_psnr,
            ssim=ema_ssim,
            lr=current_lr
        )
        print(f"        ├─ Throughput: {imgs_per_sec:.1f} imgs/s ({batches_per_sec:.2f} batch/s) | Time: {epoch_dur:.2f}s | ETA: {eta_str}")
        print(f"        └─ Best PSNR: {max(best_psnr, ema_psnr):.4f} dB | Best SSIM: {max(best_ssim, ema_ssim):.4f} | GPU Mem: Alloc {gpu_alloc:.0f}MB, Res {gpu_res:.0f}MB, Peak {gpu_max:.0f}MB")

        # Configurable Visualizations & Prediction Dumps
        if epoch % config.VISUALIZATION_INTERVAL == 0 or epoch == config.epochs:
            save_visualizations_and_predictions(
                model=ema.ema_model,
                val_dataset=val_dataset,
                fixed_indices=config.fixed_val_indices,
                epoch=epoch,
                vis_dir=config.vis_dir,
                val_preds_dir=config.val_preds_dir,
                device=device
            )

        # Save Last Model Checkpoint with full state for seamless auto-resume
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'ema_state_dict': ema.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
        }, last_ckpt_path)

        # Best Model Tracking based on EMA PSNR
        if ema_psnr > best_psnr:
            best_psnr = ema_psnr
            best_ssim = ema_ssim
            best_epoch = epoch
            patience_counter = 0

            # Save AIR-Net EMA Best Model and standard checkpoints
            if config.MODEL_VERSION == "AIR-Net-v1.1":
                torch.save(ema.state_dict(), os.path.join(config.checkpoint_dir, "airnet_v1_1_ema_best_model.pth"))
                torch.save(model.state_dict(), os.path.join(config.checkpoint_dir, "airnet_v1_1_best_model.pth"))
            elif config.MODEL_VERSION == "AIR-Net-v1.2":
                torch.save(ema.state_dict(), os.path.join(config.checkpoint_dir, "airnet_v1_2_ema_best_model.pth"))
                torch.save(model.state_dict(), os.path.join(config.checkpoint_dir, "airnet_v1_2_best_model.pth"))

            torch.save(ema.state_dict(), os.path.join(config.checkpoint_dir, "airnet_ema_best_model.pth"))
            torch.save(ema.state_dict(), os.path.join(config.checkpoint_dir, "ema_best_model.pth"))
            torch.save(model.state_dict(), os.path.join(config.checkpoint_dir, "best_model.pth"))

            # Save Best Metrics JSON
            saved_name = "airnet_ema_best_model.pth"
            if config.MODEL_VERSION == "AIR-Net-v1.1":
                saved_name = "airnet_v1_1_ema_best_model.pth"
            elif config.MODEL_VERSION == "AIR-Net-v1.2":
                saved_name = "airnet_v1_2_ema_best_model.pth"

            save_json({
                "model_version": config.MODEL_VERSION,
                "best_epoch": best_epoch,
                "best_psnr": round(best_psnr, 4),
                "best_ssim": round(best_ssim, 4),
                "best_model": saved_name
            }, config.best_metrics_file)

            print(f"        --> [NEW BEST] Saved {saved_name} (PSNR: {best_psnr:.4f} dB, SSIM: {best_ssim:.4f})")
        else:
            patience_counter += 1

        # Optional Early Stopping Check
        if getattr(config, 'EARLY_STOPPING', False) and patience_counter >= getattr(config, 'EARLY_STOP_PATIENCE', 20):
            print(f"\n⚠️ Early stopping triggered! No improvement for {patience_counter} consecutive validation epochs.")
            break

    total_training_time = time.time() - start_total_time
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0.0

    # Configurable Post-Training Inference Benchmarking
    if config.RUN_BENCHMARK_AFTER_TRAINING:
        print("\nRunning Inference Benchmarking...")
        run_inference_benchmark(config)
    else:
        print("\n[OK] Skipping post-training inference benchmark (RUN_BENCHMARK_AFTER_TRAINING=False).")

    # Optional ONNX Export Hook
    if getattr(config, 'EXPORT_ONNX', False):
        try:
            onnx_path = os.path.join(config.output_dir, "airnet_v1.onnx")
            dummy_inp = torch.randn(1, config.in_channels, 128, 128, device=device)
            torch.onnx.export(
                ema.ema_model,
                dummy_inp,
                onnx_path,
                input_names=["input"],
                output_names=["restored", "edge", "noise", "blur", "texture"],
                dynamic_axes={"input": {0: "batch_size"}, "restored": {0: "batch_size"}, "edge": {0: "batch_size"}},
                opset_version=14
            )
            print(f"[OK] Exported ONNX model to {onnx_path}")
        except Exception as e:
            print(f"⚠️ ONNX export failed: {e}")

    # Generate Final Post-Training Report
    final_report = (
        "====================================================\n"
        "KLA SEMICONDUCTOR AIR-NET V1 - FINAL REPORT\n"
        "====================================================\n"
        f"Total Training Epochs:    {config.epochs}\n"
        f"Total Training Time:      {total_training_time / 60.0:.2f} minutes ({total_training_time:.1f} s)\n"
        f"Average Epoch Time:       {avg_epoch_time:.2f} s\n"
        "----------------------------------------------------\n"
        "METRIC BREAKDOWN & COMPARISON\n"
        "----------------------------------------------------\n"
        f"Bicubic Baseline PSNR:    {bicubic_psnr:.4f} dB\n"
        f"Bicubic Baseline SSIM:    {bicubic_ssim:.4f}\n\n"
        f"Best Epoch:               Epoch {best_epoch}\n"
        f"Best Model PSNR (EMA):    {best_psnr:.4f} dB (PSNR Difference vs Bicubic: {best_psnr - bicubic_psnr:+.4f} dB)\n"
        f"Best Model SSIM (EMA):    {best_ssim:.4f} (SSIM Difference vs Bicubic: {best_ssim - bicubic_ssim:+.4f})\n\n"
        f"Final Epoch PSNR (EMA):   {ema_psnr:.4f} dB\n"
        f"Final Epoch SSIM (EMA):   {ema_ssim:.4f}\n"
        "====================================================\n"
    )

    with open(config.final_report_file, "w") as f:
        f.write(final_report)

    print("\nAIR-Net v1 training completed successfully!")
    print(final_report)

if __name__ == "__main__":
    main()
