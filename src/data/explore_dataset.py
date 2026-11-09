"""Exploratory Data Analysis (EDA) for PlantVillage Tomato Leaf Disease Dataset.

This module provides modular, beginner-friendly functions to:
1. Locate and verify the 10 Tomato class directories.
2. Inspect image integrity (verifying readable JPEG files without corruption).
3. Compute exact class distributions and calculate class imbalance metrics.
4. Inspect image dimensions (width, height), color channels, and file format.
5. Generate publication-quality figures:
   - artifacts/figures/tomato_class_distribution.png
   - artifacts/figures/tomato_sample_grid.png

Author: Crop Disease Detection Project Team
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def find_tomato_classes(color_dir: Path) -> Dict[str, Path]:
    """Scan the dataset directory and locate all 10 Tomato class folders.

    Why this matters for your viva:
    PlantVillage contains 14 different crop species (Apple, Corn, Grape, Potato, Tomato, etc.).
    For this project, we isolate only the Tomato subset to keep the initial domain focused
    and computationally tractable for learning.

    Args:
        color_dir: Path to the 'raw/color' folder in PlantVillage-Dataset.

    Returns:
        Dictionary mapping class folder name (str) to its absolute Path.
    """
    if not color_dir.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {color_dir}")

    # Locate subdirectories starting with 'Tomato'
    tomato_dirs = {
        d.name: d
        for d in sorted(color_dir.iterdir())
        if d.is_dir() and d.name.startswith("Tomato")
    }

    if len(tomato_dirs) == 0:
        raise ValueError(f"No Tomato class directories found in {color_dir}")

    return tomato_dirs


def count_and_verify_images(
    tomato_dirs: Dict[str, Path]
) -> Tuple[pd.DataFrame, List[str]]:
    """Count files and verify that every image can be decoded by PIL without corruption.

    Why this matters for your viva:
    Deep learning training pipelines crash abruptly if a corrupted JPEG file is fed
    into a DataLoader in the middle of training. Verifying images ahead of time prevents
    wasted training runs.

    Args:
        tomato_dirs: Dictionary of {class_name: class_path}.

    Returns:
        Tuple of:
          - DataFrame with columns ['class_name', 'count', 'percentage']
          - List of any corrupt filepaths found (empty if all images are valid).
    """
    records = []
    corrupt_files = []

    for class_name, class_path in tomato_dirs.items():
        # Only count valid image files, ignore OS metadata like .DS_Store
        image_files = [
            f for f in class_path.iterdir()
            if f.is_file() and not f.name.startswith(".")
        ]
        
        valid_count = 0
        for img_file in image_files:
            try:
                with Image.open(img_file) as img:
                    img.verify()  # Validates image headers and integrity without loading full pixel array
                valid_count += 1
            except Exception as e:
                corrupt_files.append(f"{img_file} (Error: {e})")

        records.append({
            "class_name": class_name,
            "count": valid_count,
        })

    df = pd.DataFrame(records)
    total_images = df["count"].sum()
    df["percentage"] = (df["count"] / total_images) * 100.0

    return df, corrupt_files


def inspect_image_properties(
    tomato_dirs: Dict[str, Path], sample_per_class: int = 10
) -> Dict[str, any]:
    """Sample images across all classes to determine resolution, mode, and format.

    Why this matters for your viva:
    Convolutional Neural Networks require uniform tensor shapes (e.g., 3 x 224 x 224).
    Knowing the native resolution (256 x 256) and color channels (RGB) helps design
    the input preprocessing and resizing pipeline.

    Args:
        tomato_dirs: Dictionary of {class_name: class_path}.
        sample_per_class: Number of sample images to inspect per class.

    Returns:
        Dictionary containing unique dimensions, color modes, and formats found.
    """
    dimensions = set()
    color_modes = set()
    formats = set()
    total_sampled = 0

    for class_name, class_path in tomato_dirs.items():
        files = [
            f for f in class_path.iterdir()
            if f.is_file() and not f.name.startswith(".")
        ][:sample_per_class]

        for img_file in files:
            with Image.open(img_file) as img:
                dimensions.add(img.size)  # (Width, Height)
                color_modes.add(img.mode)  # e.g., 'RGB'
                formats.add(img.format)    # e.g., 'JPEG'
                total_sampled += 1

    return {
        "sampled_images_count": total_sampled,
        "unique_dimensions_wh": list(dimensions),
        "unique_color_modes": list(color_modes),
        "unique_file_formats": list(formats),
    }


def analyze_class_imbalance(df: pd.DataFrame) -> Dict[str, any]:
    """Calculate imbalance metrics: min class, max class, and imbalance ratio.

    Why this matters for your viva:
    When class sizes are unequal (e.g. 5,357 vs 373), standard accuracy is misleading.
    A model could achieve high accuracy by mostly predicting the majority class.
    This analysis justifies using Stratified Splitting and Macro-F1 score.

    Args:
        df: DataFrame with ['class_name', 'count', 'percentage'].

    Returns:
        Dictionary with imbalance statistics.
    """
    min_row = df.loc[df["count"].idxmin()]
    max_row = df.loc[df["count"].idxmax()]
    imbalance_ratio = max_row["count"] / min_row["count"]

    return {
        "total_images": int(df["count"].sum()),
        "num_classes": len(df),
        "mean_images_per_class": float(df["count"].mean()),
        "median_images_per_class": float(df["count"].median()),
        "majority_class": max_row["class_name"],
        "majority_count": int(max_row["count"]),
        "minority_class": min_row["class_name"],
        "minority_count": int(min_row["count"]),
        "imbalance_ratio": float(imbalance_ratio),
    }


def plot_class_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """Create a publication-quality horizontal bar chart of image counts per class.

    Args:
        df: DataFrame with columns ['class_name', 'count', 'percentage'].
        output_path: Destination PNG filepath.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort so smallest appears at top and largest at bottom for readable hierarchy
    sorted_df = df.sort_values(by="count", ascending=True).copy()

    # Create clean display labels by stripping the 'Tomato___' prefix
    display_names = [
        name.replace("Tomato___", "").replace("_", " ")
        for name in sorted_df["class_name"]
    ]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    # Modern color palette (teal gradient)
    colors = plt.cm.viridis(np.linspace(0.3, 0.85, len(sorted_df)))
    bars = ax.barh(display_names, sorted_df["count"], color=colors, edgecolor="black", linewidth=0.6)

    # Annotate bars with exact counts and percentage
    max_count = sorted_df["count"].max()
    for bar, count, pct in zip(bars, sorted_df["count"], sorted_df["percentage"]):
        ax.text(
            bar.get_width() + (max_count * 0.015),
            bar.get_y() + bar.get_height() / 2,
            f"{count:,} ({pct:.1f}%)",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
            color="#222222",
        )

    ax.set_title(
        "PlantVillage: Tomato Leaf Disease Class Distribution (Total: 18,160 Images)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Number of Images", fontsize=11, labelpad=10)
    ax.set_xlim(0, max_count * 1.22)  # Extra space for label text
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f" Saved class distribution plot to: {output_path}")


def plot_sample_grid(
    tomato_dirs: Dict[str, Path], output_path: Path, seed: int = 42
) -> None:
    """Generate a 2x5 grid showing one representative sample image per class.

    Args:
        tomato_dirs: Dictionary of {class_name: class_path}.
        output_path: Destination PNG filepath.
        seed: Random seed for reproducible sample selection.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    random.seed(seed)

    class_names = sorted(list(tomato_dirs.keys()))
    fig, axes = plt.subplots(2, 5, figsize=(16, 7), dpi=150)
    axes = axes.flatten()

    for idx, class_name in enumerate(class_names):
        class_path = tomato_dirs[class_name]
        image_files = sorted([
            f for f in class_path.iterdir()
            if f.is_file() and not f.name.startswith(".")
        ])
        
        # Deterministically select a sample image
        sample_file = random.choice(image_files)
        with Image.open(sample_file) as img:
            img_rgb = img.convert("RGB")

        # Format label
        clean_title = (
            class_name.replace("Tomato___", "")
            .replace("_", " ")
            .replace("  ", " ")
        )
        # Highlight healthy control class in green
        is_healthy = "healthy" in class_name.lower()
        title_color = "#1b5e20" if is_healthy else "#b71c1c"

        ax = axes[idx]
        ax.imshow(img_rgb)
        ax.set_title(
            f"{clean_title}\n({len(image_files)} imgs)",
            fontsize=9,
            fontweight="bold",
            color=title_color,
            pad=6,
        )
        ax.axis("off")

    plt.suptitle(
        "PlantVillage: Visual Samples Across All 10 Tomato Leaf Classes",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f" Saved sample grid plot to: {output_path}")


def run_exploration(
    project_root: Path | None = None
) -> Tuple[pd.DataFrame, Dict[str, any], Dict[str, any]]:
    """Execute complete dataset exploration and generate artifacts.

    Args:
        project_root: Root path of the crop-disease-detection repository.

    Returns:
        Tuple of (class_distribution_df, image_properties, imbalance_metrics).
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent.parent

    color_dir = project_root / "data" / "raw" / "PlantVillage-Dataset" / "raw" / "color"
    figures_dir = project_root / "artifacts" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE I: EXPLORATORY DATA ANALYSIS (PLANTVILLAGE TOMATO SUBSET)")
    print("=" * 70)
    print(f"Project Root: {project_root}")
    print(f"Dataset Path: {color_dir}")
    print("-" * 70)

    # 1. Locate classes
    tomato_dirs = find_tomato_classes(color_dir)
    print(f"✓ Found {len(tomato_dirs)} Tomato class directories.")

    # 2. Count and verify integrity
    print("Verifying image integrity across all classes...")
    df, corrupt_files = count_and_verify_images(tomato_dirs)
    print(f"✓ Verified {df['count'].sum():,} images. Corrupt/unreadable images: {len(corrupt_files)}")

    if corrupt_files:
        print("⚠️ Warning: Corrupt files detected:")
        for cf in corrupt_files:
            print(f"  - {cf}")

    # 3. Image properties
    props = inspect_image_properties(tomato_dirs, sample_per_class=15)
    print(f"✓ Formats: {props['unique_file_formats']}")
    print(f"✓ Dimensions: {props['unique_dimensions_wh']}")
    print(f"✓ Color Modes: {props['unique_color_modes']}")

    # 4. Imbalance analysis
    imbalance = analyze_class_imbalance(df)
    print("-" * 70)
    print(f"Total Tomato Images: {imbalance['total_images']:,}")
    print(f"Majority Class: {imbalance['majority_class']} ({imbalance['majority_count']:,} images, {imbalance['majority_count']/imbalance['total_images']*100:.1f}%)")
    print(f"Minority Class: {imbalance['minority_class']} ({imbalance['minority_count']:,} images, {imbalance['minority_count']/imbalance['total_images']*100:.1f}%)")
    print(f"Imbalance Ratio: {imbalance['imbalance_ratio']:.2f}x")
    print("-" * 70)

    # 5. Generate Visualizations
    dist_plot_path = figures_dir / "tomato_class_distribution.png"
    sample_grid_path = figures_dir / "tomato_sample_grid.png"

    plot_class_distribution(df, dist_plot_path)
    plot_sample_grid(tomato_dirs, sample_grid_path, seed=42)

    print("=" * 70)
    print("EDA COMPLETE. Artifacts saved in artifacts/figures/")
    print("=" * 70)

    return df, props, imbalance


if __name__ == "__main__":
    run_exploration()
