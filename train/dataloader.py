from __future__ import annotations

from pathlib import Path

import h5py
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


def build_dataloaders_from_h5(
    h5_path: str | Path,
    batch_size: int,
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, object]]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-12:
        raise ValueError(f"train_ratio + val_ratio + test_ratio must equal 1.0, got {ratio_sum}")

    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as handle:
        keys = set(handle.keys())
        if {"X", "Y"} - keys:
            raise KeyError(f"HDF5 file must contain keys 'X' and 'Y', found {sorted(keys)}")
        x_np = handle["X"][:]
        y_np = handle["Y"][:]

    features = torch.tensor(x_np, dtype=torch.float32)
    targets = torch.tensor(y_np, dtype=torch.float32)
    if targets.ndim == 1:
        targets = targets.unsqueeze(1)

    if features.shape[0] != targets.shape[0]:
        raise ValueError(
            f"Number of samples in X and Y are inconsistent: {features.shape[0]} vs {targets.shape[0]}"
        )

    sample_count = features.shape[0]
    if sample_count < 10:
        raise ValueError(f"Too few samples for stable training: N={sample_count}")

    full_dataset = TensorDataset(features, targets)
    train_size = int(sample_count * train_ratio)
    val_size = int(sample_count * val_ratio)
    test_size = sample_count - train_size - val_size
    if min(train_size, val_size, test_size) <= 0:
        raise ValueError(
            f"Invalid split sizes: train={train_size}, val={val_size}, test={test_size}"
        )

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    train_indices = train_set.indices
    train_samples = features[train_indices]
    mu = train_samples.mean(dim=0)
    sigma = train_samples.std(dim=0, unbiased=False)

    sigma = sigma.clone()
    zero_sigma_mask = sigma == 0
    if torch.any(zero_sigma_mask):
        sigma[zero_sigma_mask] = 1.0

    features.sub_(mu).div_(sigma)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    meta: dict[str, object] = {
        "N_total": sample_count,
        "N_train": train_size,
        "N_val": val_size,
        "N_test": test_size,
        "X_shape": tuple(features.shape),
        "Y_shape": tuple(targets.shape),
        "batch_size": batch_size,
        "seed": seed,
        "mu": mu.detach().clone(),
        "sigma": sigma.detach().clone(),
        "zero_sigma_features": int(zero_sigma_mask.sum().item()),
    }
    return train_loader, val_loader, test_loader, meta
