"""Isolated Phase J Corrections & Additional Overlay Generation Script.

1. Updates the 5 transposed image-mask pairs using ImageOps.exif_transpose:
   - Preserves RGB channels and aligns image with the ground-truth mask coordinates.
   - Saves oriented images and aligned binary masks.
   - Updates manifests in splits/.
2. Re-renders the 5 alignment overlays in reports/plantseg_audit/cleaned_overlays/.
3. Generates 7 additional representative overlays from the test partition,
   increasing total overlays from 19 to 26 (>= 20 requirement).

Author: Crop Disease Detection Project Team
Phase: Phase J Final Verification
"""

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

RAW_DIR = Path("data/raw/PlantSeg")
PROCESSED_DIR = Path("data/processed/tomato_lesions_plantseg")
IMAGES_DIR = PROCESSED_DIR / "images"
MASKS_DIR = PROCESSED_DIR / "masks"
SPLITS_DIR = PROCESSED_DIR / "splits"
OVERLAYS_DIR = Path("reports/plantseg_audit/cleaned_overlays")

TRANSPOSED_ENTRIES = [
    ("train", "tomato_bacterial_leaf_spot_42.jpg", "tomato_bacterial_leaf_spot_42.png"),
    ("train", "tomato_early_blight_google_0043.jpg", "tomato_early_blight_google_0043.png"),
    ("test", "tomato_early_blight_google_0000.jpg", "tomato_early_blight_google_0000.png"),
    ("test", "tomato_early_blight_google_0056.jpg", "tomato_early_blight_google_0056.png"),
    ("test", "tomato_septoria_leaf_spot_google_0033.jpg", "tomato_septoria_leaf_spot_google_0033.png"),
]


def update_transposed_pairs():
    print("--- 1. Applying Isolated Exif-Transpose Alignment to 5 Pairs ---")
    manifest_full = pd.read_csv(SPLITS_DIR / "manifest_full.csv")
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")

    for raw_split, iname, mname in TRANSPOSED_ENTRIES:
        raw_img_p = RAW_DIR / "images" / raw_split / iname
        raw_mask_p = RAW_DIR / "annotations" / raw_split / mname

        # Correct image using EXIF orientation tag 6 (rotate 90 deg clockwise to visual orientation)
        img_raw = Image.open(raw_img_p)
        img_oriented = ImageOps.exif_transpose(img_raw).convert("RGB")
        mask_raw = Image.open(raw_mask_p).convert("L")

        assert img_oriented.size == mask_raw.size, f"Size mismatch for {iname}: {img_oriented.size} vs {mask_raw.size}"

        dest_img_p = IMAGES_DIR / iname
        dest_mask_p = MASKS_DIR / (Path(iname).stem + ".png")

        # Save oriented RGB image
        img_oriented.save(dest_img_p, quality=95)

        # Save binary mask
        m_arr = (np.array(mask_raw) > 0).astype(np.uint8) * 255
        binary_pil_mask = Image.fromarray(m_arr, mode="L")
        binary_pil_mask.save(dest_mask_p)

        new_w, new_h = img_oriented.size
        img_bytes = dest_img_p.read_bytes()
        new_hash = hashlib.md5(img_bytes).hexdigest()
        lesion_pixels = int(np.sum(m_arr > 0))
        lesion_ratio = round(float(lesion_pixels / max(m_arr.size, 1)), 6)

        # Update manifest_full
        mask_idx = manifest_full["image_name"] == iname
        manifest_full.loc[mask_idx, "width"] = new_w
        manifest_full.loc[mask_idx, "height"] = new_h
        manifest_full.loc[mask_idx, "image_hash"] = new_hash
        manifest_full.loc[mask_idx, "lesion_pixels"] = lesion_pixels
        manifest_full.loc[mask_idx, "lesion_ratio"] = lesion_ratio

        # Update split df
        if iname in train_df["image_name"].values:
            t_idx = train_df["image_name"] == iname
            train_df.loc[t_idx, "width"] = new_w
            train_df.loc[t_idx, "height"] = new_h
            train_df.loc[t_idx, "image_hash"] = new_hash
            train_df.loc[t_idx, "lesion_pixels"] = lesion_pixels
            train_df.loc[t_idx, "lesion_ratio"] = lesion_ratio
        elif iname in test_df["image_name"].values:
            t_idx = test_df["image_name"] == iname
            test_df.loc[t_idx, "width"] = new_w
            test_df.loc[t_idx, "height"] = new_h
            test_df.loc[t_idx, "image_hash"] = new_hash
            test_df.loc[t_idx, "lesion_pixels"] = lesion_pixels
            test_df.loc[t_idx, "lesion_ratio"] = lesion_ratio

        print(f"  [Updated] {iname}: size=({new_w}, {new_h}), fg_pixels={lesion_pixels} ({lesion_ratio*100:.2f}%)")

        # Re-render overlay
        render_overlay(
            img_p=dest_img_p,
            mask_p=dest_mask_p,
            title_name=iname,
            title_sub=f"Corrected Orientation & Alignment",
            out_file=OVERLAYS_DIR / f"corrected_alignment_{Path(iname).stem}.png",
            lesion_ratio=lesion_ratio,
        )

    # Save updated manifests
    manifest_full.to_csv(SPLITS_DIR / "manifest_full.csv", index=False)
    train_df.to_csv(SPLITS_DIR / "train.csv", index=False)
    test_df.to_csv(SPLITS_DIR / "test.csv", index=False)
    print("Manifests updated successfully.")


def generate_additional_test_overlays():
    print("\n--- 2. Generating Additional Representative Overlays from Test Set ---")
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")

    added = 0
    for disease, group in test_df.groupby("disease"):
        # Select 1 representative sample from test partition
        row = group.iloc[0]
        img_p = IMAGES_DIR / row["image_name"]
        mask_p = MASKS_DIR / row["mask_name"]

        out_file = OVERLAYS_DIR / f"test_sample_{Path(row['image_name']).stem}.png"
        render_overlay(
            img_p=img_p,
            mask_p=mask_p,
            title_name=row["image_name"],
            title_sub=f"{disease} [TEST SET]",
            out_file=out_file,
            lesion_ratio=row["lesion_ratio"],
        )
        added += 1
        print(f"  Generated test overlay: {out_file.name}")

    total_overlays = len(list(OVERLAYS_DIR.glob("*.png")))
    print(f"\nTotal overlays in {OVERLAYS_DIR}: {total_overlays} (Requirement: >= 20)")


def render_overlay(img_p: Path, mask_p: Path, title_name: str, title_sub: str, out_file: Path, lesion_ratio: float):
    img_rgb = np.array(Image.open(img_p).convert("RGB"))
    mask_arr = np.array(Image.open(mask_p))

    fg_mask = (mask_arr > 0)
    overlay = img_rgb.copy()
    overlay[fg_mask, 0] = np.clip(overlay[fg_mask, 0].astype(int) + 120, 0, 255).astype(np.uint8)
    overlay[fg_mask, 1] = (overlay[fg_mask, 1] * 0.4).astype(np.uint8)
    overlay[fg_mask, 2] = (overlay[fg_mask, 2] * 0.4).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_rgb)
    axes[0].set_title(f"Image\n{title_name}", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(mask_arr, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(f"Binary Mask (0=Health, 255=Lesion)\nLesion ratio: {lesion_ratio:.3f}", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(f"Lesion Overlay\n{title_sub}", fontsize=10)
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(out_file, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    update_transposed_pairs()
    generate_additional_test_overlays()
