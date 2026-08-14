import os
from typing import Tuple, List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets.augmentations import PairedTransform

class KLADataset(Dataset):
    """
    PyTorch Dataset for paired KLA Semiconductor Image Restoration.
    Reads paired .npy files from NoisyLR (128x128 float32) and GT (256x256 float32).
    Preserves exact original float values without clipping.
    """
    def __init__(
        self,
        lr_dir: str,
        gt_dir: str,
        filenames: List[str],
        transform: Optional[PairedTransform] = None
    ):
        """
        Args:
            lr_dir: Path to directory containing NoisyLR .npy files.
            gt_dir: Path to directory containing GT .npy files.
            filenames: List of matched file basenames (e.g. ['0001.npy', '0002.npy', ...]).
            transform: Optional PairedTransform instance.
        """
        self.lr_dir = lr_dir
        self.gt_dir = gt_dir
        self.filenames = sorted(filenames)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        fname = self.filenames[idx]
        lr_path = os.path.join(self.lr_dir, fname)
        gt_path = os.path.join(self.gt_dir, fname)

        lr_arr = np.load(lr_path).astype(np.float32)
        gt_arr = np.load(gt_path).astype(np.float32)

        # Ensure channel dimension: (H, W) -> (1, H, W)
        if lr_arr.ndim == 2:
            lr_arr = np.expand_dims(lr_arr, axis=0)
        if gt_arr.ndim == 2:
            gt_arr = np.expand_dims(gt_arr, axis=0)

        lr_tensor = torch.from_numpy(lr_arr)
        gt_tensor = torch.from_numpy(gt_arr)

        if self.transform is not None:
            lr_tensor, gt_tensor = self.transform(lr_tensor, gt_tensor)

        return lr_tensor, gt_tensor, fname

def get_train_val_datasets(
    train_lr_dir: str,
    train_gt_dir: str,
    seed: int = 42,
    train_split: int = 2880,
    val_split: int = 320
) -> Tuple[KLADataset, KLADataset]:
    """
    Scans the training directories, verifies paired filenames,
    and returns reproducible (train_dataset, val_dataset) split using fixed seed 42.
    """
    if not os.path.exists(train_lr_dir):
        raise FileNotFoundError(
            f"Dataset directory not found: '{train_lr_dir}'\n"
            f"Current working directory: '{os.getcwd()}'\n"
            f"Please verify that your dataset folder exists at 'Train\train\NoisyLR' or 'Train\train\NoisyLR'."
        )
    if not os.path.exists(train_gt_dir):
        raise FileNotFoundError(
            f"Dataset directory not found: '{train_gt_dir}'\n"
            f"Current working directory: '{os.getcwd()}'\n"
            f"Please verify that your dataset folder exists at 'Train\train\GT' or 'Train\train\GT'."
        )

    lr_files = set(f for f in os.listdir(train_lr_dir) if f.endswith(".npy"))
    gt_files = set(f for f in os.listdir(train_gt_dir) if f.endswith(".npy"))

    common_files = sorted(list(lr_files.intersection(gt_files)))
    if len(common_files) < train_split + val_split:
        print(f"Warning: Expected {train_split + val_split} paired files, but found {len(common_files)}. Using all available {len(common_files)} files for training and a fraction for validation.")
        train_split = int(len(common_files) * 0.9)
        val_split = len(common_files) - train_split
    assert len(common_files) >= train_split + val_split, (
        f"Expected {train_split + val_split} paired files, but found {len(common_files)}."
    )

    # Fixed seed shuffling
    rng = np.random.RandomState(seed)
    shuffled_files = common_files.copy()
    rng.shuffle(shuffled_files)

    train_files = sorted(shuffled_files[:train_split])
    val_files = sorted(shuffled_files[train_split:train_split + val_split])

    train_dataset = KLADataset(
        lr_dir=train_lr_dir,
        gt_dir=train_gt_dir,
        filenames=train_files,
        transform=PairedTransform(is_train=True)
    )

    val_dataset = KLADataset(
        lr_dir=train_lr_dir,
        gt_dir=train_gt_dir,
        filenames=val_files,
        transform=PairedTransform(is_train=False)
    )

    return train_dataset, val_dataset
