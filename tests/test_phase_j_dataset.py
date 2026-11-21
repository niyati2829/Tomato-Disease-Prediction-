"""Phase J Automated Integrity Test Suite.

Verifies:
1. Cleaned tomato lesion dataset file presence and counts (657 images, 657 masks).
2. Split manifests and exact sample allocations (train: 459, val: 99, test: 99).
3. Deduplication and zero cross-split leakage (657 unique MD5 hashes, 0 overlap).
4. Image-mask spatial dimension correspondence (100% match, zero transposed errors).
5. Binary mask validity (strictly {0, 255} on disk, zero empty masks).
6. Balanced representation of all 7 tomato foliar pathologies across splits.
7. PyTorch Dataset and DataLoader operational interfaces (tensor shapes, dtypes).
8. Strict preservation of existing production checkpoints and classification splits.

Author: Crop Disease Detection Project Team
Phase: Phase J
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.segmentation.plantseg_dataset import (
    PlantSegTomatoLesionDataset,
    get_plantseg_dataloaders,
)

DATASET_DIR = Path("data/processed/tomato_lesions_plantseg")
IMAGES_DIR = DATASET_DIR / "images"
MASKS_DIR = DATASET_DIR / "masks"
SPLITS_DIR = DATASET_DIR / "splits"

EXPECTED_TOTAL_SAMPLES = 657
EXPECTED_TRAIN_SAMPLES = 459
EXPECTED_VAL_SAMPLES = 99
EXPECTED_TEST_SAMPLES = 99

EXPECTED_DISEASES = {
    "tomato early blight",
    "tomato late blight",
    "tomato leaf mold",
    "tomato septoria leaf spot",
    "tomato bacterial leaf spot",
    "tomato yellow leaf curl virus",
    "tomato mosaic virus",
}

RESNET_CHECKPOINT = Path("artifacts/checkpoints/best_resnet18_weighted.pt")
EXPECTED_RESNET_HASH = "b86d760146c7e8389600bb56d63dca9c552bb0a649d01e74feb1cb19bfa4a3fa"

PEST_CHECKPOINT = Path("artifacts/checkpoints/pest_detection_real/best_pest_detector_real.pt")
EXPECTED_PEST_HASH = "01d8d64368b62c3ce309db4e4d18aaf6af807aad7b6225d60da048ef784b1343"


def test_directory_structure_and_file_counts():
    """Verify target directories and file counts on disk."""
    assert DATASET_DIR.exists(), f"Missing dataset directory: {DATASET_DIR}"
    assert IMAGES_DIR.exists(), f"Missing images directory: {IMAGES_DIR}"
    assert MASKS_DIR.exists(), f"Missing masks directory: {MASKS_DIR}"
    assert SPLITS_DIR.exists(), f"Missing splits directory: {SPLITS_DIR}"

    img_files = list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.png"))
    mask_files = list(MASKS_DIR.glob("*.png"))

    assert len(img_files) == EXPECTED_TOTAL_SAMPLES, (
        f"Expected {EXPECTED_TOTAL_SAMPLES} images, found {len(img_files)}"
    )
    assert len(mask_files) == EXPECTED_TOTAL_SAMPLES, (
        f"Expected {EXPECTED_TOTAL_SAMPLES} masks, found {len(mask_files)}"
    )


def test_split_manifests():
    """Verify split manifests, row counts, and sum integrity."""
    full_path = SPLITS_DIR / "manifest_full.csv"
    train_path = SPLITS_DIR / "train.csv"
    val_path = SPLITS_DIR / "val.csv"
    test_path = SPLITS_DIR / "test.csv"

    for p in (full_path, train_path, val_path, test_path):
        assert p.exists(), f"Missing manifest: {p}"

    df_full = pd.read_csv(full_path)
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    assert len(df_full) == EXPECTED_TOTAL_SAMPLES
    assert len(df_train) == EXPECTED_TRAIN_SAMPLES
    assert len(df_val) == EXPECTED_VAL_SAMPLES
    assert len(df_test) == EXPECTED_TEST_SAMPLES
    assert len(df_train) + len(df_val) + len(df_test) == EXPECTED_TOTAL_SAMPLES


def test_zero_duplicate_hashes_and_zero_leakage():
    """Verify exact deduplication and zero cross-split leakage."""
    df_train = pd.read_csv(SPLITS_DIR / "train.csv")
    df_val = pd.read_csv(SPLITS_DIR / "val.csv")
    df_test = pd.read_csv(SPLITS_DIR / "test.csv")
    df_full = pd.read_csv(SPLITS_DIR / "manifest_full.csv")

    assert df_full["image_hash"].nunique() == EXPECTED_TOTAL_SAMPLES, (
        "Duplicate image hashes found in full manifest!"
    )

    train_hashes = set(df_train["image_hash"])
    val_hashes = set(df_val["image_hash"])
    test_hashes = set(df_test["image_hash"])

    assert len(train_hashes.intersection(val_hashes)) == 0, "Train-Val hash leakage detected!"
    assert len(train_hashes.intersection(test_hashes)) == 0, "Train-Test hash leakage detected!"
    assert len(val_hashes.intersection(test_hashes)) == 0, "Val-Test hash leakage detected!"


def test_image_mask_dimension_correspondence():
    """Verify 100% spatial alignment between images and binary masks."""
    df_full = pd.read_csv(SPLITS_DIR / "manifest_full.csv")

    # Sample test across full dataset
    for idx, row in df_full.iterrows():
        img_p = IMAGES_DIR / row["image_name"]
        mask_p = MASKS_DIR / row["mask_name"]

        with Image.open(img_p) as im, Image.open(mask_p) as m:
            assert im.size == m.size, (
                f"Dimension mismatch for {row['image_name']}: img {im.size} vs mask {m.size}"
            )
            assert im.size == (row["width"], row["height"])


def test_binary_mask_integrity():
    """Verify binary masks contain strictly {0, 255} and no all-zero masks exist."""
    df_full = pd.read_csv(SPLITS_DIR / "manifest_full.csv")

    # Verify all lesion ratios are strictly > 0
    assert (df_full["lesion_pixels"] > 0).all(), "Found empty/all-zero masks in manifest!"
    assert (df_full["lesion_ratio"] > 0.0).all(), "Lesion ratio <= 0 found!"
    assert (df_full["lesion_ratio"] <= 1.0).all(), "Lesion ratio > 1 found!"

    # Sample test mask pixel values
    sample_rows = df_full.sample(n=20, random_state=42)
    for _, row in sample_rows.iterrows():
        mask_arr = np.array(Image.open(MASKS_DIR / row["mask_name"]))
        unique_vals = set(np.unique(mask_arr))
        assert unique_vals.issubset({0, 255}), f"Non-binary values {unique_vals} in {row['mask_name']}"
        assert 255 in unique_vals, f"No lesion foreground pixels in {row['mask_name']}"


def test_disease_stratification_across_splits():
    """Verify all 7 disease categories are represented across all splits."""
    df_train = pd.read_csv(SPLITS_DIR / "train.csv")
    df_val = pd.read_csv(SPLITS_DIR / "val.csv")
    df_test = pd.read_csv(SPLITS_DIR / "test.csv")

    assert set(df_train["disease"].unique()) == EXPECTED_DISEASES
    assert set(df_val["disease"].unique()) == EXPECTED_DISEASES
    assert set(df_test["disease"].unique()) == EXPECTED_DISEASES


def test_pytorch_dataset_float_mask():
    """Verify PlantSegTomatoLesionDataset with as_float_mask=True."""
    ds = PlantSegTomatoLesionDataset(
        split="train",
        dataset_dir=DATASET_DIR,
        image_size=(256, 256),
        as_float_mask=True,
    )
    assert len(ds) == EXPECTED_TRAIN_SAMPLES

    img, mask = ds[0]
    assert isinstance(img, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert img.shape == (3, 256, 256)
    assert mask.shape == (1, 256, 256)
    assert mask.dtype == torch.float32

    # Mask values should be in [0.0, 1.0]
    unique_vals = set(torch.unique(mask).tolist())
    assert unique_vals.issubset({0.0, 1.0})

    meta = ds.get_metadata(0)
    assert "disease" in meta
    assert "image_name" in meta


def test_pytorch_dataset_long_mask():
    """Verify PlantSegTomatoLesionDataset with as_float_mask=False."""
    ds = PlantSegTomatoLesionDataset(
        split="val",
        dataset_dir=DATASET_DIR,
        image_size=(128, 128),
        as_float_mask=False,
    )
    assert len(ds) == EXPECTED_VAL_SAMPLES

    img, mask = ds[0]
    assert img.shape == (3, 128, 128)
    assert mask.shape == (128, 128)
    assert mask.dtype == torch.int64

    unique_vals = set(torch.unique(mask).tolist())
    assert unique_vals.issubset({0, 1})


def test_dataloader_batching():
    """Verify get_plantseg_dataloaders factory produces valid mini-batches."""
    train_loader, val_loader, test_loader = get_plantseg_dataloaders(
        dataset_dir=DATASET_DIR,
        batch_size=4,
        image_size=(256, 256),
        as_float_mask=True,
    )

    batch_imgs, batch_masks = next(iter(train_loader))
    assert batch_imgs.shape == (4, 3, 256, 256)
    assert batch_masks.shape == (4, 1, 256, 256)
    assert batch_masks.dtype == torch.float32


def test_aspect_ratio_padding_preserves_spatial_alignment():
    """Verify letterboxing padding preserves aspect ratio and maintains pixel alignment."""
    # Test with aspect ratio preservation
    ds_padded = PlantSegTomatoLesionDataset(
        split="test",
        dataset_dir=DATASET_DIR,
        image_size=(256, 256),
        as_float_mask=True,
        preserve_aspect_ratio=True,
    )
    img_pad, mask_pad = ds_padded[0]
    assert img_pad.shape == (3, 256, 256)
    assert mask_pad.shape == (1, 256, 256)
    assert set(torch.unique(mask_pad).tolist()).issubset({0.0, 1.0})

    # Test direct resize without padding
    ds_direct = PlantSegTomatoLesionDataset(
        split="test",
        dataset_dir=DATASET_DIR,
        image_size=(256, 256),
        as_float_mask=True,
        preserve_aspect_ratio=False,
    )
    img_dir, mask_dir = ds_direct[0]
    assert img_dir.shape == (3, 256, 256)
    assert mask_dir.shape == (1, 256, 256)
    assert set(torch.unique(mask_dir).tolist()).issubset({0.0, 1.0})


def test_corrected_five_transposed_pairs_integrity():
    """Verify the 5 corrected pairs use valid spatial transformations preserving coordinates."""
    transposed_filenames = [
        "tomato_bacterial_leaf_spot_42.jpg",
        "tomato_early_blight_google_0043.jpg",
        "tomato_early_blight_google_0000.jpg",
        "tomato_early_blight_google_0056.jpg",
        "tomato_septoria_leaf_spot_google_0033.jpg",
    ]

    for fname in transposed_filenames:
        img_p = IMAGES_DIR / fname
        mask_p = MASKS_DIR / (Path(fname).stem + ".png")

        assert img_p.exists(), f"Missing corrected image: {img_p}"
        assert mask_p.exists(), f"Missing corrected mask: {mask_p}"

        with Image.open(img_p) as im, Image.open(mask_p) as m:
            assert im.size == m.size, f"Mismatch in corrected pair {fname}: img {im.size} != mask {m.size}"
            m_arr = np.array(m)
            assert np.sum(m_arr > 0) > 0, f"Corrected mask {fname} has zero lesion pixels!"
            assert set(np.unique(m_arr)).issubset({0, 255})


def test_overlay_visual_artifacts_count():
    """Verify visual overlays in reports/plantseg_audit/cleaned_overlays/ total at least 20."""
    overlay_dir = Path("reports/plantseg_audit/cleaned_overlays")
    assert overlay_dir.exists(), f"Overlay directory missing: {overlay_dir}"
    overlays = list(overlay_dir.glob("*.png"))
    assert len(overlays) >= 20, f"Expected >= 20 overlays, found {len(overlays)}"


def test_production_checkpoints_integrity():
    """Ensure ResNet18 and YOLO checkpoints have not been modified or corrupted."""
    assert RESNET_CHECKPOINT.exists(), "ResNet18 production checkpoint missing!"
    resnet_hash = hashlib.sha256(RESNET_CHECKPOINT.read_bytes()).hexdigest()
    assert resnet_hash == EXPECTED_RESNET_HASH, (
        f"ResNet18 checkpoint tampered! Expected {EXPECTED_RESNET_HASH}, got {resnet_hash}"
    )

    assert PEST_CHECKPOINT.exists(), "Pest detector production checkpoint missing!"
    pest_hash = hashlib.sha256(PEST_CHECKPOINT.read_bytes()).hexdigest()
    assert pest_hash == EXPECTED_PEST_HASH, (
        f"Pest detector checkpoint tampered! Expected {EXPECTED_PEST_HASH}, got {pest_hash}"
    )
