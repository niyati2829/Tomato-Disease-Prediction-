"""PlantSeg Real-World Tomato Lesion Segmentation Dataset & DataLoader (Phase J).

Provides PyTorch Dataset and DataLoader abstractions for the cleaned, deduplicated,
and leakage-safe PlantSeg tomato lesion segmentation benchmark.

Target task: Class-agnostic semantic localization of tomato foliar disease lesions
(Binary segmentation: 0 = Healthy lamina / backdrop, 1 = Diseased lesion spot).

Author: Crop Disease Detection Project Team
Phase: Phase J
"""

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class PlantSegTomatoLesionDataset(Dataset):
    """PyTorch Dataset for real-world tomato leaf lesion segmentation.

    Loads cleaned RGB images and binary lesion masks from the processed PlantSeg
    tomato dataset. Ensures nearest-neighbor interpolation for masks to maintain
    crisp lesion boundaries.
    """

    def __init__(
        self,
        split: str = "train",
        dataset_dir: Union[Path, str] = "data/processed/tomato_lesions_plantseg",
        image_size: Tuple[int, int] = (256, 256),
        transform: Optional[transforms.Compose] = None,
        as_float_mask: bool = True,
        preserve_aspect_ratio: bool = True,
        augment: bool = False,
    ) -> None:
        """Initialize PlantSeg tomato lesion dataset.

        Args:
            split: Target split ("train", "val", "test", or "all").
            dataset_dir: Root directory of processed tomato lesion data.
            image_size: Target spatial resolution (H, W). Default (256, 256).
            transform: Optional torchvision transforms for the RGB image.
            as_float_mask: If True, returns mask as (1, H, W) float in [0.0, 1.0].
                           If False, returns mask as (H, W) torch.long in {0, 1}.
            preserve_aspect_ratio: If True, preserves aspect ratio with centered
                                   padding (letterbox). If False, resizes directly.
            augment: If True, applies random synchronized spatial flips/rotations
                     and photometric jitter for robust training regularization.
        """
        self.dataset_dir = Path(dataset_dir)
        self.split = split.lower()
        self.image_size = image_size
        self.as_float_mask = as_float_mask
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.augment = augment
        self.color_jitter = transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15)

        self.images_dir = self.dataset_dir / "images"
        self.masks_dir = self.dataset_dir / "masks"
        self.splits_dir = self.dataset_dir / "splits"

        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset root directory not found: {self.dataset_dir}")

        # Select manifest file
        if self.split in ("train", "val", "test"):
            manifest_file = self.splits_dir / f"{self.split}.csv"
        elif self.split == "all":
            manifest_file = self.splits_dir / "manifest_full.csv"
        else:
            raise ValueError(f"Unknown split: {split}. Expected 'train', 'val', 'test', or 'all'.")

        if not manifest_file.exists():
            raise FileNotFoundError(f"Manifest file not found: {manifest_file}")

        self.manifest_df = pd.read_csv(manifest_file)
        self.records = self.manifest_df.to_dict(orient="records")

        # Default ImageNet normalization
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        record = self.records[idx]
        img_p = self.images_dir / record["image_name"]
        mask_p = self.masks_dir / record["mask_name"]

        # Load RGB leaf image
        pil_img = Image.open(img_p).convert("RGB")
        # Load single-channel binary mask
        pil_mask = Image.open(mask_p).convert("L")

        # Synchronized Data Augmentation for training split
        if self.augment:
            if random.random() < 0.5:
                pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
                pil_mask = pil_mask.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                pil_img = pil_img.transpose(Image.FLIP_TOP_BOTTOM)
                pil_mask = pil_mask.transpose(Image.FLIP_TOP_BOTTOM)
            if random.random() < 0.5:
                rot = random.choice([Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270])
                pil_img = pil_img.transpose(rot)
                pil_mask = pil_mask.transpose(rot)
            # Photometric perturbation on RGB channels only
            pil_img = self.color_jitter(pil_img)

        target_h, target_w = self.image_size

        if self.preserve_aspect_ratio:
            orig_w, orig_h = pil_img.size

            # Scale preserving aspect ratio
            scale = min(target_w / orig_w, target_h / orig_h)
            new_w = max(1, int(round(orig_w * scale)))
            new_h = max(1, int(round(orig_h * scale)))

            # Image resized with bilinear interpolation
            resized_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
            # Mask resized with NEAREST NEIGHBOR to preserve discrete binary boundary
            resized_mask = pil_mask.resize((new_w, new_h), Image.NEAREST)

            # Centered padding (letterbox) to maintain spatial coordinate correspondence
            pad_x = (target_w - new_w) // 2
            pad_y = (target_h - new_h) // 2

            padded_img = Image.new("RGB", (target_w, target_h), (0, 0, 0))
            padded_img.paste(resized_img, (pad_x, pad_y))

            padded_mask = Image.new("L", (target_w, target_h), 0)
            padded_mask.paste(resized_mask, (pad_x, pad_y))

            pil_img = padded_img
            pil_mask = padded_mask
        else:
            # Direct resize (bilinear for image, nearest for mask)
            pil_img = pil_img.resize((target_w, target_h), Image.BILINEAR)
            pil_mask = pil_mask.resize((target_w, target_h), Image.NEAREST)

        mask_np = np.array(pil_mask, dtype=np.uint8)
        binary_np = (mask_np > 0).astype(np.float32 if self.as_float_mask else np.int64)

        img_tensor = self.transform(pil_img)

        if self.as_float_mask:
            # Shape (1, H, W) float in [0.0, 1.0]
            mask_tensor = torch.from_numpy(binary_np).unsqueeze(0).float()
        else:
            # Shape (H, W) long in {0, 1}
            mask_tensor = torch.from_numpy(binary_np).long()

        return img_tensor, mask_tensor

    def get_metadata(self, idx: int) -> Dict:
        """Return metadata dictionary for a given index."""
        return dict(self.records[idx])


def get_plantseg_dataloaders(
    dataset_dir: Union[Path, str] = "data/processed/tomato_lesions_plantseg",
    batch_size: int = 8,
    image_size: Tuple[int, int] = (256, 256),
    num_workers: int = 0,
    as_float_mask: bool = True,
    preserve_aspect_ratio: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test PyTorch DataLoaders.

    Args:
        dataset_dir: Root directory of processed dataset.
        batch_size: Batch size for mini-batching.
        image_size: Target (H, W) resolution.
        num_workers: Worker processes for parallel loading.
        as_float_mask: Whether masks are returned as (1, H, W) float or (H, W) long.
        preserve_aspect_ratio: Whether to use letterboxing padding to preserve aspect ratio.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_dataset = PlantSegTomatoLesionDataset(
        split="train",
        dataset_dir=dataset_dir,
        image_size=image_size,
        as_float_mask=as_float_mask,
        preserve_aspect_ratio=preserve_aspect_ratio,
    )
    val_dataset = PlantSegTomatoLesionDataset(
        split="val",
        dataset_dir=dataset_dir,
        image_size=image_size,
        as_float_mask=as_float_mask,
        preserve_aspect_ratio=preserve_aspect_ratio,
    )
    test_dataset = PlantSegTomatoLesionDataset(
        split="test",
        dataset_dir=dataset_dir,
        image_size=image_size,
        as_float_mask=as_float_mask,
        preserve_aspect_ratio=preserve_aspect_ratio,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader, test_loader
