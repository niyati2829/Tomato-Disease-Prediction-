"""Generative Subset Dataset Loader (REQ-09).

Safety & Isolation Rules:
- Reads ONLY `data/splits/train.csv` for generative fitting and `data/splits/val.csv` for checkpoint selection.
- NEVER accesses `data/splits/test.csv`.
- Uses a manageable balanced subset suitable for rapid training on Apple Silicon M2.

Author: Crop Disease Detection Project Team
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


TARGET_CLASSES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Bacterial_spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
]


class LeafGenerativeDataset(Dataset):
    """Dataset returning (image_tensor, is_diseased_flag, class_name)."""

    def __init__(
        self,
        manifest_csv: Path | str,
        image_size: int = 64,
        max_per_class: int = 200,
        range_neg_one: bool = False,
        seed: int = 42,
    ) -> None:
        manifest_p = Path(manifest_csv)
        if "test" in manifest_p.name.lower():
            raise ValueError("SECURITY VIOLATION: Generative training/analysis must NEVER access test.csv!")
        if not manifest_p.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_p}")

        self.image_size = image_size
        self.range_neg_one = range_neg_one

        # Read rows grouped by target class
        rows_by_class: Dict[str, List[Tuple[str, int]]] = {c: [] for c in TARGET_CLASSES}
        with open(manifest_p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                c_name = row["class_name"]
                if c_name in rows_by_class:
                    rows_by_class[c_name].append((row["image_path"], int(row["class_index"])))

        rng = np.random.RandomState(seed)
        self.samples: List[Tuple[Path, int, str]] = []
        for c_name in TARGET_CLASSES:
            items = rows_by_class[c_name]
            rng.shuffle(items)
            selected = items[:max_per_class]
            is_diseased = 0 if c_name == "Tomato___healthy" else 1
            for img_p, _ in selected:
                self.samples.append((Path(img_p), is_diseased, c_name))

        # Final shuffle across classes
        rng.shuffle(self.samples)

        tfms = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # [0.0, 1.0]
        ]
        if range_neg_one:
            tfms.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))  # [-1.0, 1.0]

        self.transform = transforms.Compose(tfms)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        img_p, is_diseased, c_name = self.samples[idx]
        pil_img = Image.open(img_p).convert("RGB")
        tensor = self.transform(pil_img)
        return tensor, is_diseased, c_name

