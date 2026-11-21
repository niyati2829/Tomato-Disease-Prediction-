"""Phase J: Tomato Lesion Segmentation Dataset Preparation Script.

Processes raw PlantSeg v7 dataset to create a cleaned, deduplicated, leakage-safe
tomato disease lesion segmentation benchmark for U-Net training.

Actions:
1. Filter 667 raw tomato records from PlantSeg v7 Metadata.csv.
2. Resolve 10 duplicate records via MD5 image content hashing -> 657 unique images.
3. Align 5 transposed image-mask pairs ($W x H$ vs $H x W$) via transposition.
4. Convert raw categorical integer masks (indices 96-102) to standard binary masks
   (0 = background/healthy foliage, 255 = diseased lesion).
5. Partition 657 images into stratified train/val/test splits (459 / 99 / 99).
6. Save images, masks, and split manifests to data/processed/tomato_lesions_plantseg/.
7. Generate visual overlays verifying corrected alignment and disease coverage in
   reports/plantseg_audit/cleaned_overlays/.

Author: Crop Disease Detection Project Team
Phase: Phase J
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

RAW_DATA_DIR = Path("data/raw/PlantSeg")
PROCESSED_DIR = Path("data/processed/tomato_lesions_plantseg")
IMAGES_OUT_DIR = PROCESSED_DIR / "images"
MASKS_OUT_DIR = PROCESSED_DIR / "masks"
SPLITS_OUT_DIR = PROCESSED_DIR / "splits"
OVERLAYS_OUT_DIR = Path("reports/plantseg_audit/cleaned_overlays")

SPLIT_MAP = {"Training": "train", "Validation": "val", "Test": "test"}

# 5 known transposed image-mask pairs from audit
TRANSPOSED_FILENAMES = {
    "tomato_bacterial_leaf_spot_42.jpg",
    "tomato_early_blight_google_0043.jpg",
    "tomato_early_blight_google_0000.jpg",
    "tomato_early_blight_google_0056.jpg",
    "tomato_septoria_leaf_spot_google_0033.jpg",
}


def prepare_dataset() -> pd.DataFrame:
    """Prepare cleaned tomato lesion segmentation dataset."""
    print("=" * 70)
    print("PHASE J: PREPARING CLEANED TOMATO LESION SEGMENTATION DATASET")
    print("=" * 70)

    # 1. Create output directories
    IMAGES_OUT_DIR.mkdir(parents=True, exist_ok=True)
    MASKS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAYS_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Read Metadata.csv
    metadata_path = RAW_DATA_DIR / "Metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    raw_df = pd.read_csv(metadata_path)
    tomato_df = raw_df[raw_df["Plant"] == "Tomato"].copy()
    print(f"Loaded {len(tomato_df)} raw tomato records from {metadata_path}")

    # 3. Compute MD5 image hash for deduplication
    records = []
    for idx, row in tomato_df.iterrows():
        sp = SPLIT_MAP[row["Split"]]
        img_p = RAW_DATA_DIR / "images" / sp / row["Name"]
        mask_p = RAW_DATA_DIR / "annotations" / sp / row["Label file"]

        if not img_p.exists():
            raise FileNotFoundError(f"Missing raw image: {img_p}")
        if not mask_p.exists():
            raise FileNotFoundError(f"Missing raw mask: {mask_p}")

        img_bytes = img_p.read_bytes()
        img_hash = hashlib.md5(img_bytes).hexdigest()

        records.append({
            "original_name": row["Name"],
            "original_label_file": row["Label file"],
            "disease": row["Disease"],
            "original_split": row["Split"],
            "raw_img_path": img_p,
            "raw_mask_path": mask_p,
            "img_hash": img_hash,
            "original_index": int(row["Index"]),
        })

    full_records_df = pd.DataFrame(records)
    print(f"Total tomato records processed: {len(full_records_df)}")

    # 4. Deduplicate by MD5 image content hash
    # Drop duplicates, keeping the first occurrence
    dedup_df = full_records_df.drop_duplicates(subset=["img_hash"], keep="first").copy()
    dedup_df = dedup_df.reset_index(drop=True)
    print(f"Unique images after MD5 deduplication: {len(dedup_df)} (dropped {len(full_records_df) - len(dedup_df)} duplicate records)")

    # 5. Process and save images and binary masks
    processed_records = []
    transposed_count = 0

    for idx, row in dedup_df.iterrows():
        img_p = row["raw_img_path"]
        mask_p = row["raw_mask_path"]

        # Read image
        pil_img = Image.open(img_p).convert("RGB")
        im_arr = np.array(pil_img)

        # Read mask
        pil_mask = Image.open(mask_p)
        m_arr = np.array(pil_mask)

        # Check alignment / transposition
        was_transposed = False
        if im_arr.shape[:2] != m_arr.shape[:2]:
            m_arr = np.transpose(m_arr)
            was_transposed = True
            transposed_count += 1
            if row["original_name"] in TRANSPOSED_FILENAMES:
                print(f"  [Alignment Corrected] {row['original_name']}: {pil_mask.size} -> {im_arr.shape[1], im_arr.shape[0]}")

        # Dimension assertion
        if im_arr.shape[:2] != m_arr.shape[:2]:
            raise ValueError(f"Dimension mismatch for {row['original_name']}: img={im_arr.shape[:2]}, mask={m_arr.shape[:2]}")

        # Convert mask to binary (0 = background / healthy foliar tissue, 255 = lesion)
        binary_mask_arr = (m_arr > 0).astype(np.uint8) * 255
        lesion_pixels = int(np.sum(binary_mask_arr > 0))
        total_pixels = binary_mask_arr.size
        lesion_ratio = float(lesion_pixels / max(total_pixels, 1))

        # Consistent naming
        img_out_name = row["original_name"]
        mask_out_name = Path(row["original_name"]).stem + ".png"

        dest_img_path = IMAGES_OUT_DIR / img_out_name
        dest_mask_path = MASKS_OUT_DIR / mask_out_name

        # Save cleaned RGB image
        pil_img.save(dest_img_path, quality=95)

        # Save cleaned binary mask as grayscale PNG
        binary_pil_mask = Image.fromarray(binary_mask_arr, mode="L")
        binary_pil_mask.save(dest_mask_path)

        processed_records.append({
            "image_name": img_out_name,
            "mask_name": mask_out_name,
            "disease": row["disease"],
            "image_hash": row["img_hash"],
            "width": int(im_arr.shape[1]),
            "height": int(im_arr.shape[0]),
            "lesion_pixels": lesion_pixels,
            "lesion_ratio": round(lesion_ratio, 6),
            "alignment_corrected": was_transposed,
            "original_disease_index": row["original_index"],
        })

    print(f"Saved {len(processed_records)} cleaned images to {IMAGES_OUT_DIR}")
    print(f"Saved {len(processed_records)} binary masks to {MASKS_OUT_DIR}")
    print(f"Total transposed masks corrected: {transposed_count}")

    manifest_df = pd.DataFrame(processed_records)

    # 6. Partition into stratified, leakage-safe splits (70% train: 459, 15% val: 99, 15% test: 99)
    train_df, temp_df = train_test_split(
        manifest_df,
        train_size=459,
        test_size=198,
        stratify=manifest_df["disease"],
        random_state=42,
    )

    val_df, test_df = train_test_split(
        temp_df,
        train_size=99,
        test_size=99,
        stratify=temp_df["disease"],
        random_state=42,
    )

    # Assign split column
    manifest_df["split"] = ""
    manifest_df.loc[train_df.index, "split"] = "train"
    manifest_df.loc[val_df.index, "split"] = "val"
    manifest_df.loc[test_df.index, "split"] = "test"

    train_df = manifest_df[manifest_df["split"] == "train"].copy().reset_index(drop=True)
    val_df = manifest_df[manifest_df["split"] == "val"].copy().reset_index(drop=True)
    test_df = manifest_df[manifest_df["split"] == "test"].copy().reset_index(drop=True)

    # Save manifest CSVs
    manifest_df.to_csv(SPLITS_OUT_DIR / "manifest_full.csv", index=False)
    train_df.to_csv(SPLITS_OUT_DIR / "train.csv", index=False)
    val_df.to_csv(SPLITS_OUT_DIR / "val.csv", index=False)
    test_df.to_csv(SPLITS_OUT_DIR / "test.csv", index=False)

    print("\nSaved split manifests to", SPLITS_OUT_DIR)
    print(f"  train.csv: {len(train_df)} images ({len(train_df)/len(manifest_df)*100:.1f}%)")
    print(f"  val.csv:   {len(val_df)} images ({len(val_df)/len(manifest_df)*100:.1f}%)")
    print(f"  test.csv:  {len(test_df)} images ({len(test_df)/len(manifest_df)*100:.1f}%)")

    # 7. Verification: Cross-split leakage check
    train_hashes = set(train_df["image_hash"])
    val_hashes = set(val_df["image_hash"])
    test_hashes = set(test_df["image_hash"])

    leakage_train_val = train_hashes.intersection(val_hashes)
    leakage_train_test = train_hashes.intersection(test_hashes)
    leakage_val_test = val_hashes.intersection(test_hashes)

    if leakage_train_val or leakage_train_test or leakage_val_test:
        raise AssertionError("Cross-split leakage detected across newly generated splits!")
    print("Cross-split leakage verification: 0 overlapping image hashes across splits (PASSED)")

    # 8. Generate Visual Overlays
    generate_cleaned_overlays(manifest_df)

    return manifest_df


def generate_cleaned_overlays(manifest_df: pd.DataFrame) -> None:
    """Generate visual overlay diagnostics for cleaned samples."""
    print("\nGenerating visual verification overlays in", OVERLAYS_OUT_DIR)

    # 1. Generate overlays for the 5 corrected alignment cases
    corrected_cases = manifest_df[manifest_df["alignment_corrected"]].copy()
    for _, row in corrected_cases.iterrows():
        create_overlay_plot(row, prefix="corrected_alignment_")

    # 2. Generate 2 sample overlays for each of the 7 disease categories
    for disease, group in manifest_df.groupby("disease"):
        samples = group.head(2)
        for _, row in samples.iterrows():
            create_overlay_plot(row, prefix="sample_")

    n_overlays = len(list(OVERLAYS_OUT_DIR.glob("*.png")))
    print(f"Generated {n_overlays} verification overlays in {OVERLAYS_OUT_DIR}")


def create_overlay_plot(row: pd.Series, prefix: str = "") -> None:
    """Generate a single 3-panel visualization: RGB image, Binary Mask, Red Overlay."""
    img_path = IMAGES_OUT_DIR / row["image_name"]
    mask_path = MASKS_OUT_DIR / row["mask_name"]

    img_rgb = np.array(Image.open(img_path).convert("RGB"))
    mask_arr = np.array(Image.open(mask_path))

    fg_mask = (mask_arr > 0)
    overlay = img_rgb.copy()
    # Apply semi-transparent red tint on lesion pixels
    overlay[fg_mask, 0] = np.clip(overlay[fg_mask, 0].astype(int) + 120, 0, 255).astype(np.uint8)
    overlay[fg_mask, 1] = (overlay[fg_mask, 1] * 0.4).astype(np.uint8)
    overlay[fg_mask, 2] = (overlay[fg_mask, 2] * 0.4).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_rgb)
    axes[0].set_title(f"Cleaned Image\n{row['image_name']}", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(mask_arr, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(f"Binary Mask (0=Health, 255=Lesion)\nLesion ratio: {row['lesion_ratio']:.3f}", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(f"Lesion Overlay\n{row['disease']} [{row['split'].upper()}]", fontsize=10)
    axes[2].axis("off")

    plt.tight_layout()
    safe_name = Path(row["image_name"]).stem
    out_file = OVERLAYS_OUT_DIR / f"{prefix}{safe_name}.png"
    plt.savefig(out_file, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    prepare_dataset()
