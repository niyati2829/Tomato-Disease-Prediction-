"""Execute Convolution & Feature Map Demonstrations (REQ-08).

Outputs:
1. `artifacts/figures/convolution/classical_2d_convolution_kernels.png`:
   Scratch NumPy 2D spatial convolution with canonical kernels (Sobel X/Y, Gaussian, Laplacian, Sharpen, Ridge).
2. `artifacts/figures/convolution/resnet18_feature_map_hierarchy.png`:
   Intermediate activation maps extracted across ResNet18 depth (`conv1`, `layer1`, `layer2`, `layer3`, `layer4`).

Author: Crop Disease Detection Project Team
"""

import json
import os
import sys
from pathlib import Path
from typing import List

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

from src.models.transfer_learning import TomatoResNet18
from src.utils.device import get_device
from src.visualization.convolution_visualizer import visualize_classical_filters
from src.visualization.feature_maps import ResNetFeatureExtractor, plot_feature_map_hierarchy


def run_convolution_and_feature_demonstrations(
    checkpoint_path: Path | str = "artifacts/checkpoints/best_resnet18_weighted.pt",
    sample_img_path: Path | str = "data/raw/PlantVillage-Dataset/raw/color/Tomato___Early_blight",
    out_dir: Path | str = "artifacts/figures/convolution",
) -> List[Path]:
    """Execute end-to-end convolution and feature map demonstrations."""
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    sample_dir = Path(sample_img_path)

    # Locate first sample image
    sample_files = [f for f in sorted(sample_dir.iterdir()) if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    if not sample_files:
        raise FileNotFoundError(f"No sample images found in {sample_dir}")
    leaf_path = sample_files[0]

    pil_img = Image.open(leaf_path).convert("RGB")
    rgb_224 = np.array(pil_img.resize((224, 224)))
    gray_224 = cv2.cvtColor(rgb_224, cv2.COLOR_RGB2GRAY)

    saved_figures = []

    # 1. Classical 2D Spatial Convolution Demonstration
    print("[Phase D] Generating 2D spatial convolution demonstration...")
    fig1_path = out_p / "classical_2d_convolution_kernels.png"
    fig1 = visualize_classical_filters(gray_224, save_path=fig1_path)
    plt.close(fig1)
    saved_figures.append(fig1_path)
    print(f"  -> Saved: {fig1_path.name}")

    # 2. ResNet18 Deep Hierarchical Feature Map Extraction
    print("[Phase D] Extracting deep ResNet18 feature map activations...")
    device = get_device(preferred="mps")
    model = TomatoResNet18(num_classes=10)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(pil_img).unsqueeze(0).to(device)

    with ResNetFeatureExtractor(model) as extractor:
        feature_maps = extractor.extract(tensor)

    fig2_path = out_p / "resnet18_feature_map_hierarchy.png"
    fig2 = plot_feature_map_hierarchy(rgb_224, feature_maps, n_channels_per_layer=6, save_path=fig2_path)
    plt.close(fig2)
    saved_figures.append(fig2_path)
    print(f"  -> Saved: {fig2_path.name}")

    return saved_figures


if __name__ == "__main__":
    run_convolution_and_feature_demonstrations()

