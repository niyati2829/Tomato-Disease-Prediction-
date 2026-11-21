"""Semantic Segmentation Dataset & Synthetic Lesion Benchmark Generator (REQ-08).

Design & Academic Integrity Note:
PlantVillage provides image-level classification labels and whole-leaf foreground masks,
but lacks pixel-level disease lesion ground truth.
To enable complete, reproducible training, validation, and testing of U-Net on a MacBook M2
without fabricating annotations or misrepresenting PlantVillage labels, this module provides:
1. `LeafSegmentationDataset`: PyTorch Dataset handling image-mask pairs with nearest-neighbor
   mask interpolation to prevent discrete class corruption.
2. `generate_synthetic_lesion_benchmark`: Transparent educational benchmark generator
   synthesizing realistic leaf contours (Class 1: Lamina) with necrotic spots (Class 2: Lesion)
   on a black/neutral backdrop (Class 0: Background) with exact pixel-level ground truth.

Author: Crop Disease Detection Project Team
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class LeafSegmentationDataset(Dataset):
    """PyTorch Dataset for paired leaf RGB images and segmentation masks."""

    def __init__(
        self,
        image_paths: List[Path | str],
        mask_paths: List[Path | str],
        image_size: Tuple[int, int] = (128, 128),
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        """Initialize dataset.

        Args:
            image_paths: List of file paths to RGB leaf images.
            mask_paths: List of file paths to single-channel integer class masks.
            image_size: Target resolution (H, W).
            transform: Optional torchvision transforms for image.
        """
        if len(image_paths) != len(mask_paths):
            raise ValueError(f"Image count ({len(image_paths)}) must match mask count ({len(mask_paths)})")

        self.image_paths = [Path(p) for p in image_paths]
        self.mask_paths = [Path(p) for p in mask_paths]
        self.image_size = image_size
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_p = self.image_paths[idx]
        mask_p = self.mask_paths[idx]

        # Load RGB image
        pil_img = Image.open(img_p).convert("RGB")
        # Resize image with bilinear interpolation
        pil_img = pil_img.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)

        # Load single-channel integer categorical mask
        pil_mask = Image.open(mask_p)
        # CRITICAL: Resize categorical mask with NEAREST NEIGHBOR to preserve discrete class IDs
        pil_mask = pil_mask.resize((self.image_size[1], self.image_size[0]), Image.NEAREST)

        mask_np = np.array(pil_mask, dtype=np.int64)

        # Handle potential 3-channel masks by taking first channel
        if mask_np.ndim == 3:
            mask_np = mask_np[:, :, 0]

        img_tensor = self.transform(pil_img)
        mask_tensor = torch.from_numpy(mask_np).long()

        return img_tensor, mask_tensor


def generate_synthetic_lesion_benchmark(
    n_samples: int = 60,
    image_size: int = 128,
    out_dir: Path | str = "data/processed/synthetic_segmentation",
    seed: int = 42,
) -> Dict[str, Tuple[List[Path], List[Path]]]:
    """Generate a clean, verifiable synthetic leaf lesion segmentation dataset.

    Classes:
        0: Background (Black/Gray)
        1: Healthy Leaf Lamina (Green foliage)
        2: Diseased / Necrotic Lesion (Brown / Yellow concentric patches)

    Splits:
        - train: 40 images
        - val: 10 images
        - test: 10 images

    Returns:
        Dictionary with keys 'train', 'val', 'test', each containing (image_paths, mask_paths).
    """
    out_p = Path(out_dir)
    img_dir = out_p / "images"
    mask_dir = out_p / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)
    all_images: List[Path] = []
    all_masks: List[Path] = []

    center = (image_size // 2, image_size // 2)

    for i in range(n_samples):
        # 1. Canvas initialization
        img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        mask = np.zeros((image_size, image_size), dtype=np.uint8)

        # Background subtle noise
        bg_noise = rng.randint(15, 30, size=(image_size, image_size, 3), dtype=np.uint8)
        img[:] = bg_noise

        # 2. Draw Leaf Lamina (Class 1) - Organic Elliptical / Polygonal shape
        leaf_axes = (
            rng.randint(int(image_size * 0.32), int(image_size * 0.44)),
            rng.randint(int(image_size * 0.22), int(image_size * 0.34)),
        )
        angle = rng.randint(-35, 35)

        # Foliage color variation (lush greens)
        base_green = np.array([
            rng.randint(30, 60),    # R
            rng.randint(120, 180),  # G
            rng.randint(25, 55),    # B
        ], dtype=np.uint8)

        cv2.ellipse(img, center, leaf_axes, angle, 0, 360, base_green.tolist(), -1)
        cv2.ellipse(mask, center, leaf_axes, angle, 0, 360, 1, -1)

        # Add leaf veins
        n_veins = rng.randint(3, 7)
        for _ in range(n_veins):
            v_start = center
            v_end = (
                center[0] + rng.randint(-leaf_axes[0] // 2, leaf_axes[0] // 2),
                center[1] + rng.randint(-leaf_axes[1] // 2, leaf_axes[1] // 2),
            )
            cv2.line(img, v_start, v_end, (80, 200, 70), 1)

        # 3. Draw Necrotic / Chlorotic Lesions (Class 2)
        n_lesions = rng.randint(2, 6)
        for _ in range(n_lesions):
            # Place lesion inside leaf area
            offset_x = rng.randint(-leaf_axes[0] // 3, leaf_axes[0] // 3)
            offset_y = rng.randint(-leaf_axes[1] // 3, leaf_axes[1] // 3)
            l_center = (center[0] + offset_x, center[1] + offset_y)
            l_radius = rng.randint(4, 12)

            # Outer chlorotic yellow halo
            halo_color = (rng.randint(180, 220), rng.randint(160, 200), rng.randint(30, 60))
            cv2.circle(img, l_center, l_radius + 2, halo_color, -1)
            cv2.circle(mask, l_center, l_radius + 2, 2, -1)

            # Inner necrotic brown center
            necrotic_color = (rng.randint(90, 130), rng.randint(50, 80), rng.randint(20, 40))
            cv2.circle(img, l_center, max(l_radius - 2, 2), necrotic_color, -1)
            cv2.circle(mask, l_center, max(l_radius - 2, 2), 2, -1)

        # Save files
        img_file = img_dir / f"leaf_sample_{i:04d}.png"
        mask_file = mask_dir / f"leaf_sample_{i:04d}_mask.png"

        Image.fromarray(img).save(img_file)
        Image.fromarray(mask).save(mask_file)

        all_images.append(img_file)
        all_masks.append(mask_file)

    # Partition: 40 train, 10 val, 10 test
    splits = {
        "train": (all_images[:40], all_masks[:40]),
        "val": (all_images[40:50], all_masks[40:50]),
        "test": (all_images[50:], all_masks[50:]),
    }

    return splits

