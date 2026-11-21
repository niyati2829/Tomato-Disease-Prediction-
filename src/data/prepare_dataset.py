"""Dataset Splitting, Verification, and Preparation Module.

This module provides reproducible, beginner-friendly functions to:
1. Fully decode every tomato image in the PlantVillage dataset using PIL Image.load().
2. Construct deterministic, stratified Train/Validation/Test splits (70% / 15% / 15%).
3. Enforce an alphabetical class-to-index mapping across all splits.
4. Guarantee zero data leakage between train, validation, and test partitions.
5. Export relative-path split manifests to `data/splits/train.csv`, `val.csv`, `test.csv`.

Author: Crop Disease Detection Project Team
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure repository root is in sys.path so 'src' can be imported when running script directly
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split

from src.data.explore_dataset import find_tomato_classes


def get_class_mappings(tomato_dirs: Dict[str, Path]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Generate consistent alphabetical class-to-index mappings.

    Why this matters for your viva:
    Consistency in class indexing is paramount in multi-class classification.
    Alphabetical sorting ensures class index 0 consistently maps to the same disease
    across all splits, data loaders, confusion matrices, and the final demo app.

    Args:
        tomato_dirs: Dictionary mapping class name to folder path.

    Returns:
        Tuple of (class_to_idx, idx_to_class) dictionaries.
    """
    sorted_classes = sorted(list(tomato_dirs.keys()))
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted_classes)}
    idx_to_class = {idx: cls_name for idx, cls_name in enumerate(sorted_classes)}
    return class_to_idx, idx_to_class


def verify_and_decode_all_images(
    tomato_dirs: Dict[str, Path],
    class_to_idx: Dict[str, int],
    project_root: Path,
) -> Tuple[pd.DataFrame, List[str]]:
    """Perform a full image decoding audit using PIL Image.load().

    Why this matters for your viva:
    Header verification (Image.verify) only tests that file headers are intact.
    A full decode (Image.load) actually parses the compressed image payload into
    raw pixel memory. This guarantees that PyTorch's DataLoader will not encounter
    a decompression bomb, truncated scanline, or broken block during training.

    Args:
        tomato_dirs: Dictionary of {class_name: class_path}.
        class_to_idx: Mapping of class name to integer index.
        project_root: Root path of the repository for relative path construction.

    Returns:
        Tuple of:
          - DataFrame with ['image_path', 'class_name', 'class_index']
          - List of unreadable / corrupted file errors (empty if all images decode successfully).
    """
    records = []
    unreadable_files = []

    for class_name in sorted(tomato_dirs.keys()):
        class_path = tomato_dirs[class_name]
        class_idx = class_to_idx[class_name]

        image_files = sorted([
            f for f in class_path.iterdir()
            if f.is_file() and not f.name.startswith(".")
        ])

        for img_file in image_files:
            try:
                # Fully decode pixel array into RAM
                with Image.open(img_file) as img:
                    img.load()

                # Use relative path from project root for portability
                rel_path = img_file.relative_to(project_root).as_posix()
                records.append({
                    "image_path": rel_path,
                    "class_name": class_name,
                    "class_index": class_idx,
                })
            except Exception as e:
                unreadable_files.append(f"{img_file} (Error: {e})")

    df = pd.DataFrame(records)
    return df, unreadable_files


def create_stratified_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into Train, Val, and Test subsets while preserving class proportions.

    Why this matters for your viva:
    In an imbalanced dataset (e.g. 14.36x ratio), random splitting could accidentally
    allocate very few or zero minority-class samples to the validation or test sets.
    Stratified splitting enforces that each class is represented in the exact same
    proportions (70% / 15% / 15%) across all three subsets.

    Args:
        df: Master DataFrame with all valid images.
        train_ratio: Fraction for training (default 0.70).
        val_ratio: Fraction for validation (default 0.15).
        test_ratio: Fraction for testing (default 0.15).
        seed: Random seed for exact reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    if not abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6:
        raise ValueError("Ratios must sum to 1.0")

    # Step 1: Split into Train (70%) and Temp (30%)
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        random_state=seed,
        stratify=df["class_index"],
    )

    # Step 2: Split Temp (30%) equally into Validation (15%) and Test (15%)
    val_share_of_temp = val_ratio / temp_ratio  # 0.15 / 0.30 = 0.50
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_share_of_temp),
        random_state=seed,
        stratify=temp_df["class_index"],
    )

    # Reset indices for clean CSV manifests
    train_df = train_df.sort_values(by=["class_index", "image_path"]).reset_index(drop=True)
    val_df = val_df.sort_values(by=["class_index", "image_path"]).reset_index(drop=True)
    test_df = test_df.sort_values(by=["class_index", "image_path"]).reset_index(drop=True)

    return train_df, val_df, test_df


def validate_no_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    expected_total: int,
    expected_num_classes: int = 10,
) -> bool:
    """Verify zero data overlap and valid class coverage across all splits.

    Why this matters for your viva:
    Data leakage occurs when samples from the validation or test set leak into the
    training set, leading to overly optimistic and invalid evaluation metrics.
    Rigorous set intersection checks mathematically guarantee zero leakage.

    Args:
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        test_df: Testing DataFrame.
        expected_total: Total number of verified images.
        expected_num_classes: Total number of unique classes expected.

    Returns:
        True if all checks pass. Raises AssertionError otherwise.
    """
    train_paths = set(train_df["image_path"])
    val_paths = set(val_df["image_path"])
    test_paths = set(test_df["image_path"])

    # 1. Zero Intersection Checks
    train_val_overlap = train_paths.intersection(val_paths)
    train_test_overlap = train_paths.intersection(test_paths)
    val_test_overlap = val_paths.intersection(test_paths)

    assert len(train_val_overlap) == 0, f"Data leakage between Train and Val: {len(train_val_overlap)} images"
    assert len(train_test_overlap) == 0, f"Data leakage between Train and Test: {len(train_test_overlap)} images"
    assert len(val_test_overlap) == 0, f"Data leakage between Val and Test: {len(val_test_overlap)} images"

    # 2. Total Count Conservation
    actual_total = len(train_df) + len(val_df) + len(test_df)
    assert actual_total == expected_total, f"Total count mismatch: expected {expected_total}, got {actual_total}"

    # 3. Class Representation in Every Split
    assert train_df["class_index"].nunique() == expected_num_classes, "Train set missing classes!"
    assert val_df["class_index"].nunique() == expected_num_classes, "Val set missing classes!"
    assert test_df["class_index"].nunique() == expected_num_classes, "Test set missing classes!"

    return True


def save_split_manifests(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    class_to_idx: Dict[str, int],
    splits_dir: Path,
) -> Dict[str, Path]:
    """Save split CSV manifests and class mapping JSON to disk.

    Args:
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        test_df: Testing DataFrame.
        class_to_idx: Mapping of class names to integer indices.
        splits_dir: Target directory (`data/splits`).

    Returns:
        Dictionary of created file paths.
    """
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_path = splits_dir / "train.csv"
    val_path = splits_dir / "val.csv"
    test_path = splits_dir / "test.csv"
    mapping_path = splits_dir / "class_mapping.json"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2)

    return {
        "train": train_path,
        "val": val_path,
        "test": test_path,
        "mapping": mapping_path,
    }


def compute_split_distribution_summary(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    idx_to_class: Dict[int, str],
) -> pd.DataFrame:
    """Compute per-class counts and percentages across Train, Val, and Test splits.

    Args:
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        test_df: Testing DataFrame.
        idx_to_class: Mapping of integer index to class name.

    Returns:
        Summary DataFrame showing counts and percentages for each split.
    """
    total_train = len(train_df)
    total_val = len(val_df)
    total_test = len(test_df)

    records = []
    for idx in sorted(idx_to_class.keys()):
        cls_name = idx_to_class[idx]
        train_count = int((train_df["class_index"] == idx).sum())
        val_count = int((val_df["class_index"] == idx).sum())
        test_count = int((test_df["class_index"] == idx).sum())
        total_class = train_count + val_count + test_count

        records.append({
            "class_index": idx,
            "class_name": cls_name,
            "train_count": train_count,
            "train_pct": (train_count / total_train) * 100.0,
            "val_count": val_count,
            "val_pct": (val_count / total_val) * 100.0,
            "test_count": test_count,
            "test_pct": (test_count / total_test) * 100.0,
            "total_count": total_class,
        })

    summary_df = pd.DataFrame(records)
    return summary_df


def prepare_dataset(
    project_root: Path | None = None,
    config_path: Path | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute end-to-end dataset preparation and split creation.

    Args:
        project_root: Root path of the repository.
        config_path: Path to configuration YAML.

    Returns:
        Tuple of (train_df, val_df, test_df, summary_df).
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent.parent

    if config_path is None:
        config_path = project_root / "configs" / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    color_dir = project_root / "data" / "raw" / "PlantVillage-Dataset" / "raw" / "color"
    splits_dir = project_root / "data" / "splits"

    train_ratio = config["dataset"].get("train_split", 0.70)
    val_ratio = config["dataset"].get("val_split", 0.15)
    test_ratio = config["dataset"].get("test_split", 0.15)
    seed = config.get("seed", 42)

    print("=" * 70)
    print("STAGE 2: DATA PREPARATION & STRATIFIED SPLITTING")
    print("=" * 70)
    print(f"Project Root: {project_root}")
    print(f"Dataset Path: {color_dir}")
    print(f"Split Ratios: {train_ratio*100:.0f}% Train / {val_ratio*100:.0f}% Val / {test_ratio*100:.0f}% Test (Seed: {seed})")
    print("-" * 70)

    # 1. Discover classes & establish mappings
    tomato_dirs = find_tomato_classes(color_dir)
    class_to_idx, idx_to_class = get_class_mappings(tomato_dirs)
    print(f"✓ Alphabetical class indexing established for {len(class_to_idx)} classes:")
    for cls_name, idx in class_to_idx.items():
        print(f"  [{idx}] {cls_name}")

    # 2. Full image decoding audit
    print("\nRunning full image decoding verification (PIL Image.load())...")
    df, unreadable = verify_and_decode_all_images(tomato_dirs, class_to_idx, project_root)
    print(f"✓ Successfully decoded all {len(df):,} images. Unreadable files: {len(unreadable)}")

    if unreadable:
        print("⚠️ Warning: Unreadable files detected:")
        for uf in unreadable:
            print(f"  - {uf}")
        raise RuntimeError(f"Found {len(unreadable)} unreadable images during decoding audit.")

    # 3. Create stratified splits
    print("\nCreating stratified splits...")
    train_df, val_df, test_df = create_stratified_splits(
        df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    # 4. Validate zero leakage
    validate_no_leakage(train_df, val_df, test_df, expected_total=len(df), expected_num_classes=len(class_to_idx))
    print("✓ Zero data leakage verified: Train, Val, and Test sets have mutually exclusive image paths.")

    # 5. Save manifests
    manifest_paths = save_split_manifests(train_df, val_df, test_df, class_to_idx, splits_dir)
    print(f"✓ Saved manifests:")
    print(f"  - Train: {manifest_paths['train']} ({len(train_df):,} images)")
    print(f"  - Val:   {manifest_paths['val']} ({len(val_df):,} images)")
    print(f"  - Test:  {manifest_paths['test']} ({len(test_df):,} images)")
    print(f"  - Class Mapping: {manifest_paths['mapping']}")

    # 6. Distribution summary
    summary_df = compute_split_distribution_summary(train_df, val_df, test_df, idx_to_class)
    print("\n" + "=" * 70)
    print("SPLIT CLASS DISTRIBUTION SUMMARY")
    print("=" * 70)
    for _, row in summary_df.iterrows():
        print(
            f"[{row['class_index']}] {row['class_name'][:35]:<35} | "
            f"Train: {row['train_count']:>4} ({row['train_pct']:>5.2f}%) | "
            f"Val: {row['val_count']:>4} ({row['val_pct']:>5.2f}%) | "
            f"Test: {row['test_count']:>4} ({row['test_pct']:>5.2f}%) | "
            f"Total: {row['total_count']:>4}"
        )
    print("-" * 70)
    print(
        f"{'TOTALS':<40} | "
        f"Train: {len(train_df):>4} (100.00%) | "
        f"Val: {len(val_df):>4} (100.00%) | "
        f"Test: {len(test_df):>4} (100.00%) | "
        f"Total: {len(df):>4}"
    )
    print("=" * 70)

    return train_df, val_df, test_df, summary_df


if __name__ == "__main__":
    prepare_dataset()
