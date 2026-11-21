"""Unit Tests for Dataset, Splitting, and DataLoaders.

Verifies:
1. All 10 classes are present across all split manifests.
2. Split sizes are correct (70% Train, 15% Val, 15% Test).
3. Zero data leakage between splits (pairwise set intersection is empty).
4. Tensor dimensions match expectations: single sample (3, 224, 224), batch (32, 3, 224, 224).
5. Tensor pixel values are finite (no NaNs or Infs).
6. Labels are valid class indices in range [0, 9].
7. Validation and test transformations are strictly deterministic.

Author: Crop Disease Detection Project Team
"""

from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

from src.data.dataset import (
    PlantVillageTomatoDataset,
    create_dataloaders,
    get_datasets,
    get_transforms,
)


@pytest.fixture(scope="module")
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def splits_dir(project_root) -> Path:
    return project_root / "data" / "splits"


def test_split_files_exist(splits_dir):
    """Verify that train.csv, val.csv, and test.csv exist."""
    assert (splits_dir / "train.csv").exists(), "train.csv missing!"
    assert (splits_dir / "val.csv").exists(), "val.csv missing!"
    assert (splits_dir / "test.csv").exists(), "test.csv missing!"


def test_split_sizes_and_proportions(splits_dir):
    """Verify exact split sizes and preservation of 70% / 15% / 15% proportions."""
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")
    test_df = pd.read_csv(splits_dir / "test.csv")

    total_images = len(train_df) + len(val_df) + len(test_df)
    assert total_images == 18160, f"Expected 18,160 total images, got {total_images}"

    # Verify counts match exact 70/15/15 calculation
    assert len(train_df) == 12712, f"Expected 12,712 train images, got {len(train_df)}"
    assert len(val_df) == 2724, f"Expected 2,724 val images, got {len(val_df)}"
    assert len(test_df) == 2724, f"Expected 2,724 test images, got {len(test_df)}"


def test_all_ten_classes_present_in_each_split(splits_dir):
    """Verify all 10 tomato classes are represented in train, val, and test splits."""
    for split_name in ["train.csv", "val.csv", "test.csv"]:
        df = pd.read_csv(splits_dir / split_name)
        unique_indices = sorted(df["class_index"].unique().tolist())
        assert unique_indices == list(range(10)), (
            f"{split_name} does not have all 10 classes! Found: {unique_indices}"
        )


def test_no_data_leakage(splits_dir):
    """Verify zero overlap between train, val, and test image paths."""
    train_paths = set(pd.read_csv(splits_dir / "train.csv")["image_path"])
    val_paths = set(pd.read_csv(splits_dir / "val.csv")["image_path"])
    test_paths = set(pd.read_csv(splits_dir / "test.csv")["image_path"])

    assert len(train_paths & val_paths) == 0, "Data leakage between Train and Val!"
    assert len(train_paths & test_paths) == 0, "Data leakage between Train and Test!"
    assert len(val_paths & test_paths) == 0, "Data leakage between Val and Test!"


def test_dataset_item_shape_and_label(project_root, splits_dir):
    """Verify single dataset item shape (3, 224, 224) and valid integer label."""
    datasets = get_datasets(splits_dir=splits_dir, root_dir=project_root)
    train_dataset = datasets["train"]

    assert len(train_dataset) == 12712
    image, label = train_dataset[0]

    # Verify shape
    assert isinstance(image, torch.Tensor), "Image is not a torch.Tensor!"
    assert image.shape == (3, 224, 224), f"Expected shape (3, 224, 224), got {image.shape}"
    assert image.dtype == torch.float32, f"Expected dtype float32, got {image.dtype}"

    # Verify finite values (no NaNs or Infs)
    assert torch.isfinite(image).all(), "Tensor contains NaN or Inf values!"

    # Verify label
    assert isinstance(label, int), f"Label should be integer, got {type(label)}"
    assert 0 <= label < 10, f"Label {label} outside valid class range [0, 9]"


def test_dataloader_batch_dimensions(project_root, splits_dir):
    """Verify DataLoader yields a batch of shape [32, 3, 224, 224] with finite values."""
    loaders = create_dataloaders(
        splits_dir=splits_dir,
        root_dir=project_root,
        batch_size=32,
        num_workers=0,
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    # Check a single batch from train loader
    images, labels = next(iter(train_loader))
    assert images.shape == (32, 3, 224, 224), f"Expected batch shape (32, 3, 224, 224), got {images.shape}"
    assert labels.shape == (32,), f"Expected labels shape (32,), got {labels.shape}"
    assert torch.isfinite(images).all(), "Train batch contains non-finite values!"
    assert (labels >= 0).all() and (labels < 10).all(), "Batch labels outside [0, 9] range!"

    # Check a single batch from val loader
    val_images, val_labels = next(iter(val_loader))
    assert val_images.shape == (32, 3, 224, 224)
    assert val_labels.shape == (32,)
    assert torch.isfinite(val_images).all()


def test_validation_transform_is_deterministic(project_root, splits_dir):
    """Verify that applying validation transforms to the same image twice produces identical tensors."""
    val_transform = get_transforms(image_size=(224, 224), is_training=False)

    val_df = pd.read_csv(splits_dir / "val.csv")
    sample_path = project_root / val_df.iloc[0]["image_path"]

    with Image.open(sample_path) as img:
        img_rgb = img.convert("RGB")

    t1 = val_transform(img_rgb)
    t2 = val_transform(img_rgb)

    assert torch.equal(t1, t2), "Validation transforms must be strictly deterministic!"
