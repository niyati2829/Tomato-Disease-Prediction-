"""Script to perform detailed PlantSeg tomato subset audit, generate CSV summary and visual overlays."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

DATA_DIR = Path("data/raw/PlantSeg")
OVERLAY_DIR = Path("reports/plantseg_audit/overlays")
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

metadata_file = DATA_DIR / "Metadata.csv"
df = pd.read_csv(metadata_file)
tomato_df = df[df["Plant"] == "Tomato"].copy()

split_map = {"Training": "train", "Validation": "val", "Test": "test"}

# 1. Category Summary Calculation
summary_rows = []
for disease, group in tomato_df.groupby("Disease"):
    n_images = len(group)
    n_masks = len(group["Label file"].dropna())
    
    valid_pairs = 0
    invalid_pairs = 0
    dim_mismatches = 0
    empty_masks = 0
    res_list = []
    
    for _, row in group.iterrows():
        sp = split_map.get(row["Split"])
        img_p = DATA_DIR / "images" / sp / row["Name"]
        mask_p = DATA_DIR / "annotations" / sp / row["Label file"]
        
        if img_p.exists() and mask_p.exists():
            with Image.open(img_p) as im, Image.open(mask_p) as m:
                iw, ih = im.size
                mw, mh = m.size
                m_arr = np.array(m)
                
            res_list.append(f"{iw}x{ih}")
            if (iw, ih) == (mw, mh):
                valid_pairs += 1
            else:
                dim_mismatches += 1
                valid_pairs += 1 # file exists, dimension transposed
            
            if np.sum(m_arr > 0) == 0:
                empty_masks += 1
        else:
            invalid_pairs += 1
            
    summary_rows.append({
        "Disease Category": disease,
        "Class Index": int(group["Index"].iloc[0]),
        "Total Images": n_images,
        "Total Masks": n_masks,
        "Valid Pairs": valid_pairs,
        "Invalid Pairs": invalid_pairs,
        "Dimension Mismatches (Transposed)": dim_mismatches,
        "Empty Masks": empty_masks,
        "Common Resolutions": ", ".join(pd.Series(res_list).value_counts().head(3).index.tolist()),
        "Mean Mask Ratio": round(group["Mask ratio"].mean(), 4),
        "Training Count": len(group[group["Split"] == "Training"]),
        "Validation Count": len(group[group["Split"] == "Validation"]),
        "Test Count": len(group[group["Split"] == "Test"]),
    })

summary_df = pd.DataFrame(summary_rows)
summary_csv_path = Path("reports/plantseg_audit/tomato_dataset_summary.csv")
summary_df.to_csv(summary_csv_path, index=False)
print(f"Saved summary CSV to {summary_csv_path}")

# 2. Visual Overlay Generation (2 samples per category = 14 samples)
samples_generated = 0
for disease, group in tomato_df.groupby("Disease"):
    # Pick 2 samples from Training
    train_samples = group[group["Split"] == "Training"].head(2)
    for idx_row, row in train_samples.iterrows():
        sp = split_map.get(row["Split"])
        img_p = DATA_DIR / "images" / sp / row["Name"]
        mask_p = DATA_DIR / "annotations" / sp / row["Label file"]
        
        with Image.open(img_p) as im, Image.open(mask_p) as m:
            img_rgb = np.array(im.convert("RGB"))
            mask_arr = np.array(m)
            
        # Handle transposed dimensions if any
        if img_rgb.shape[:2] != mask_arr.shape[:2]:
            mask_arr = np.transpose(mask_arr)
            
        # Create blended overlay: red tint on foreground lesion
        fg_mask = (mask_arr > 0)
        overlay = img_rgb.copy()
        overlay[fg_mask, 0] = np.clip(overlay[fg_mask, 0].astype(int) + 120, 0, 255).astype(np.uint8)
        overlay[fg_mask, 1] = (overlay[fg_mask, 1] * 0.4).astype(np.uint8)
        overlay[fg_mask, 2] = (overlay[fg_mask, 2] * 0.4).astype(np.uint8)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_rgb)
        axes[0].set_title(f"Original Image\n{row['Name']}", fontsize=11)
        axes[0].axis("off")
        
        axes[1].imshow(mask_arr, cmap="magma")
        axes[1].set_title(f"Ground-Truth Mask (Index {row['Index']})\nRatio: {row['Mask ratio']:.3f}", fontsize=11)
        axes[1].axis("off")
        
        axes[2].imshow(overlay)
        axes[2].set_title(f"Lesion Overlay\n{disease}", fontsize=11)
        axes[2].axis("off")
        
        plt.tight_layout()
        safe_name = row["Name"].replace(".jpg", "").replace(".png", "")
        out_fig_path = OVERLAY_DIR / f"overlay_{safe_name}.png"
        plt.savefig(out_fig_path, dpi=150)
        plt.close(fig)
        samples_generated += 1
        print(f"Generated overlay: {out_fig_path.name}")

print(f"Total visual overlays generated: {samples_generated}")
