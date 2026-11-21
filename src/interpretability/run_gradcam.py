"""Execute Grad-CAM Interpretability Demonstrations (REQ-08).

Applies Grad-CAM to our frozen production ResNet18 model across:
- Correctly classified disease lesions (Early Blight, Bacterial Spot, Yellow Leaf Curl).
- Healthy foliage (demonstrating dispersed non-lesion attention).
- Comparative multi-class diagnostic panel.

Visualizations are saved to artifacts/figures/interpretability/.

Author: Crop Disease Detection Project Team
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.interpretability.grad_cam import GradCAM
from src.interpretability.visualization import overlay_heatmap, plot_gradcam_explanation
from src.models.transfer_learning import TomatoResNet18
from src.utils.device import get_device


def get_inference_transform() -> transforms.Compose:
    """Standard ImageNet-compatible deterministic inference transform."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_production_model(checkpoint_path: Path, device: torch.device) -> TomatoResNet18:
    """Load frozen production TomatoResNet18 checkpoint."""
    model = TomatoResNet18(num_classes=10)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def find_sample_image(class_folder_name: str, base_dir: Path) -> Path:
    """Find the first valid image in a raw class directory."""
    folder = base_dir / class_folder_name
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    for file in sorted(folder.iterdir()):
        if file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            return file
    raise FileNotFoundError(f"No image files found in: {folder}")


def run_interpretability_demonstrations(
    checkpoint_path: Path | str = "artifacts/checkpoints/best_resnet18_weighted.pt",
    class_mapping_path: Path | str = "data/splits/class_mapping.json",
    raw_data_dir: Path | str = "data/raw/PlantVillage-Dataset/raw/color",
    out_dir: Path | str = "artifacts/figures/interpretability",
) -> List[Path]:
    """Generate Grad-CAM interpretability figures."""
    checkpoint_p = Path(checkpoint_path)
    mapping_p = Path(class_mapping_path)
    raw_dir_p = Path(raw_data_dir)
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    with open(mapping_p, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    device = get_device(preferred="mps")
    model = load_production_model(checkpoint_p, device)
    transform = get_inference_transform()

    # Target the last convolutional layer in layer4 of ResNet18
    target_layer = model.layer4[-1].conv2
    grad_cam = GradCAM(model=model, target_layer=target_layer)

    # Key diagnostic classes to evaluate
    eval_cases = [
        ("Tomato___Early_blight", "Early Blight (Target-board Concentric Rings)"),
        ("Tomato___Bacterial_spot", "Bacterial Spot (Water-soaked Lesions)"),
        ("Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Yellow Leaf Curl Virus (Chlorotic Margins)"),
        ("Tomato___healthy", "Healthy Leaf (Uniform Green Lamina)"),
    ]

    saved_figures = []

    print("[Phase D] Generating individual Grad-CAM diagnostic plots...")
    for class_folder, desc in eval_cases:
        img_path = find_sample_image(class_folder, raw_dir_p)
        pil_img = Image.open(img_path).convert("RGB")
        rgb_raw = np.array(pil_img.resize((224, 224)))

        tensor = transform(pil_img).unsqueeze(0).to(device)
        tensor.requires_grad = True

        target_idx = class_to_idx[class_folder]
        heatmap, pred_idx, conf = grad_cam.generate_heatmap(tensor, target_class=target_idx)
        pred_name = idx_to_class[pred_idx]

        slug = class_folder.replace("Tomato___", "").lower()
        save_file = out_p / f"gradcam_{slug}.png"

        plot_gradcam_explanation(
            rgb_image=rgb_raw,
            heatmap=heatmap,
            predicted_label=pred_name.replace("Tomato___", "").replace("_", " "),
            confidence=conf,
            ground_truth_label=class_folder.replace("Tomato___", "").replace("_", " "),
            save_path=save_file,
            target_layer_name="layer4[-1].conv2",
        )
        plt.close("all")
        saved_figures.append(save_file)
        print(f"  -> Saved: {save_file.name} (Conf: {conf*100:.2f}%)")

    # Multi-class comparative grid figure
    print("[Phase D] Generating 4-class Grad-CAM comparative panel...")
    fig, axes = plt.subplots(4, 3, figsize=(15, 16), dpi=300)
    for row_idx, (class_folder, desc) in enumerate(eval_cases):
        img_path = find_sample_image(class_folder, raw_dir_p)
        pil_img = Image.open(img_path).convert("RGB")
        rgb_raw = np.array(pil_img.resize((224, 224)))

        tensor = transform(pil_img).unsqueeze(0).to(device)
        tensor.requires_grad = True

        target_idx = class_to_idx[class_folder]
        heatmap, pred_idx, conf = grad_cam.generate_heatmap(tensor, target_class=target_idx)
        overlay = overlay_heatmap(rgb_raw, heatmap, alpha=0.45)

        # Col 0: Original
        axes[row_idx, 0].imshow(rgb_raw)
        axes[row_idx, 0].set_title(f"Input: {desc.split('(')[0].strip()}", fontsize=10, fontweight="bold")
        axes[row_idx, 0].axis("off")

        # Col 1: Heatmap
        axes[row_idx, 1].imshow(heatmap, cmap="jet", vmin=0.0, vmax=1.0)
        axes[row_idx, 1].set_title("Grad-CAM Saliency", fontsize=10, fontweight="bold")
        axes[row_idx, 1].axis("off")

        # Col 2: Overlay
        axes[row_idx, 2].imshow(overlay)
        axes[row_idx, 2].set_title(f"Overlay ({conf*100:.1f}%)", fontsize=10, fontweight="bold", color="#1f77b4")
        axes[row_idx, 2].axis("off")

    fig.suptitle(
        "Comparative Model Interpretability: Grad-CAM Explanations Across Pathology Categories",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    multi_file = out_p / "gradcam_comparative_panel.png"
    fig.savefig(multi_file, bbox_inches="tight")
    plt.close(fig)
    saved_figures.append(multi_file)
    print(f"  -> Saved: {multi_file.name}")

    grad_cam.remove_hooks()
    return saved_figures


if __name__ == "__main__":
    run_interpretability_demonstrations()
