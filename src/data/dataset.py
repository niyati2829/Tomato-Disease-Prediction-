"""PyTorch Dataset & DataLoader Module for PlantVillage Tomato Leaves.

This module provides:
1. Standard vision transformations:
   - Training: Resize(224, 224), RandomHorizontalFlip, RandomRotation(15), ToTensor, Normalize(ImageNet).
   - Validation & Test: Resize(224, 224), ToTensor, Normalize(ImageNet) [Deterministic].
2. Custom PyTorch `Dataset` (`PlantVillageTomatoDataset`) that loads images via Pillow and indexes manifests.
3. `create_dataloaders()` helper function with configurable batch size, num_workers, and shuffling.

Author: Crop Disease Detection Project Team
"""

from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Standard ImageNet normalization parameters (mean and standard deviation per RGB channel)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(
    image_size: Tuple[int, int] = (224, 224),
    is_training: bool = True,
) -> transforms.Compose:
    """Build torchvision transform pipelines for training and evaluation.

    Why this matters for your viva:
    1. Training transforms include stochastic data augmentation (flips, rotations)
       to synthetically expand data diversity and prevent the CNN from overfitting.
    2. Validation/Test transforms must be strictly deterministic (no randomness)
       to ensure fair, reproducible, and unbiased model benchmarking.

    Args:
        image_size: Tuple of (height, width), default (224, 224).
        is_training: If True, applies mild data augmentation. If False, deterministic only.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class PlantVillageTomatoDataset(Dataset):
    """Custom PyTorch Dataset for loading PlantVillage tomato leaf images from CSV manifests."""

    def __init__(
        self,
        split_csv: Path | str,
        root_dir: Path | str | None = None,
        transform: Optional[Callable] = None,
    ) -> None:
        """Initialize dataset from split manifest CSV.

        Args:
            split_csv: Path to CSV file containing ['image_path', 'class_name', 'class_index'].
            root_dir: Root directory prepended to relative image paths in CSV. Defaults to project root.
            transform: Optional torchvision transform pipeline to apply to images.
        """
        self.split_csv = Path(split_csv)
        if not self.split_csv.exists():
            raise FileNotFoundError(f"Split CSV manifest not found: {self.split_csv}")

        self.df = pd.read_csv(self.split_csv)
        required_cols = {"image_path", "class_name", "class_index"}
        if not required_cols.issubset(self.df.columns):
            raise ValueError(f"CSV manifest must contain columns {required_cols}, found {list(self.df.columns)}")

        if root_dir is None:
            # Default to repo root (2 levels up from src/data)
            self.root_dir = Path(__file__).resolve().parent.parent.parent
        else:
            self.root_dir = Path(root_dir)

        self.transform = transform

        # Cache class index mappings for convenient inspection
        unique_classes = self.df.sort_values(by="class_index")[["class_index", "class_name"]].drop_duplicates()
        self.classes = list(unique_classes["class_name"])
        self.class_to_idx = dict(zip(unique_classes["class_name"], unique_classes["class_index"]))
        self.idx_to_class = dict(zip(unique_classes["class_index"], unique_classes["class_name"]))

    def __len__(self) -> int:
        """Return total number of samples in this split."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Fetch, decode, transform, and return a single sample.

        Args:
            idx: Integer index of the sample.

        Returns:
            Tuple of (image_tensor, class_index) where image_tensor has shape (3, H, W).
        """
        row = self.df.iloc[idx]
        image_rel_path = row["image_path"]
        class_idx = int(row["class_index"])

        image_full_path = self.root_dir / image_rel_path
        if not image_full_path.exists():
            raise FileNotFoundError(f"Image not found at path: {image_full_path}")

        # Open and ensure 3-channel RGB
        with Image.open(image_full_path) as img:
            image = img.convert("RGB")

        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            # Default fallback: ToTensor without normalization
            image_tensor = transforms.ToTensor()(image)

        return image_tensor, class_idx


def get_datasets(
    splits_dir: Path | str = "data/splits",
    root_dir: Path | str | None = None,
    image_size: Tuple[int, int] = (224, 224),
) -> Dict[str, PlantVillageTomatoDataset]:
    """Instantiate Train, Validation, and Test Dataset objects.

    Args:
        splits_dir: Directory containing train.csv, val.csv, and test.csv.
        root_dir: Repository root directory.
        image_size: Target image dimensions (height, width).

    Returns:
        Dictionary mapping 'train', 'val', 'test' to their respective Dataset instances.
    """
    splits_path = Path(splits_dir)

    train_dataset = PlantVillageTomatoDataset(
        split_csv=splits_path / "train.csv",
        root_dir=root_dir,
        transform=get_transforms(image_size=image_size, is_training=True),
    )

    val_dataset = PlantVillageTomatoDataset(
        split_csv=splits_path / "val.csv",
        root_dir=root_dir,
        transform=get_transforms(image_size=image_size, is_training=False),
    )

    test_dataset = PlantVillageTomatoDataset(
        split_csv=splits_path / "test.csv",
        root_dir=root_dir,
        transform=get_transforms(image_size=image_size, is_training=False),
    )

    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }


def create_dataloaders(
    splits_dir: Path | str = "data/splits",
    batch_size: int = 32,
    num_workers: int = 0,
    image_size: Tuple[int, int] = (224, 224),
    root_dir: Path | str | None = None,
) -> Dict[str, DataLoader]:
    """Create PyTorch DataLoaders for Train, Validation, and Test sets.

    Why this matters for your viva:
    - DataLoaders handle mini-batch packaging, memory pinning, shuffling, and multi-threaded loading.
    - Training data is shuffled (shuffle=True) so gradients in each optimization step are i.i.d.
    - Validation and test data are NOT shuffled (shuffle=False) for deterministic evaluation.

    Args:
        splits_dir: Directory containing split CSVs.
        batch_size: Number of images per mini-batch (default 32).
        num_workers: Background worker subprocesses (default 0 for macOS M2 stability).
        image_size: Target image size (height, width).
        root_dir: Repository root directory.

    Returns:
        Dictionary mapping 'train', 'val', 'test' to their respective DataLoader instances.
    """
    datasets = get_datasets(splits_dir=splits_dir, root_dir=root_dir, image_size=image_size)

    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )

    val_loader = DataLoader(
        datasets["val"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    test_loader = DataLoader(
        datasets["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }
