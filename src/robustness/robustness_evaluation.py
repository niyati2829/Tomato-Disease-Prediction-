"""Robustness benchmarking and visualization across corruptions and adversarial attacks."""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

from .corruptions import CORRUPTION_REGISTRY, apply_corruption, _to_pil
from .adversarial_tests import evaluate_fgsm_robustness

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def evaluate_corruption_robustness(
    model: nn.Module,
    dataset,
    device: torch.device,
    corruption_names: Optional[List[str]] = None,
    severities: List[int] = [1, 2, 3],
    max_samples: int = 500,
    batch_size: int = 32,
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Evaluate classification performance across corruptions and severities.

    Args:
        model: PyTorch model in eval mode.
        dataset: PlantVillageTomatoDataset instance.
        device: Target compute device.
        corruption_names: List of corruption names from registry. Defaults to all 8.
        severities: Severity levels to evaluate (default [1, 2, 3]).
        max_samples: Number of samples from dataset to evaluate.
        batch_size: Inference batch size.

    Returns:
        Dict: corruption_name -> severity -> {'accuracy': float, 'mean_confidence': float}
    """
    if corruption_names is None:
        corruption_names = list(CORRUPTION_REGISTRY.keys())

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # Sample images and labels from dataset
    n_total = len(dataset)
    indices = np.linspace(0, n_total - 1, min(max_samples, n_total), dtype=int)

    raw_images = []
    labels = []
    print(f"Loading {len(indices)} evaluation samples for robustness testing...")
    for idx in indices:
        row = dataset.df.iloc[idx]
        img_path = dataset.root_dir / row["image_path"]
        with Image.open(img_path) as img:
            raw_images.append(img.convert("RGB"))
        labels.append(int(row["class_index"]))

    labels = np.array(labels)
    model.eval()

    # 1. Clean Baseline Performance on this subset
    clean_tensors = torch.stack([eval_transform(img) for img in raw_images])
    clean_preds = []
    clean_confs = []
    with torch.inference_mode():
        for i in range(0, len(clean_tensors), batch_size):
            batch = clean_tensors[i : i + batch_size].to(device)
            out = model(batch)
            probs = torch.softmax(out, dim=-1)
            conf, pred = torch.max(probs, dim=-1)
            clean_preds.extend(pred.cpu().numpy().tolist())
            clean_confs.extend(conf.cpu().numpy().tolist())

    clean_acc = float(np.mean(np.array(clean_preds) == labels))
    clean_mean_conf = float(np.mean(clean_confs))
    print(f"Clean Subset Baseline Accuracy: {clean_acc * 100:.2f}%, Mean Conf: {clean_mean_conf * 100:.2f}%")

    results = {"clean": {0: {"accuracy": clean_acc, "mean_confidence": clean_mean_conf}}}

    # 2. Corruptions
    for corr in corruption_names:
        results[corr] = {}
        for sev in severities:
            corrupted_tensors = []
            for img in raw_images:
                corr_img = apply_corruption(img, corr, severity=sev)
                corrupted_tensors.append(eval_transform(corr_img))

            corrupted_tensors = torch.stack(corrupted_tensors)
            corr_preds = []
            corr_confs = []
            with torch.inference_mode():
                for i in range(0, len(corrupted_tensors), batch_size):
                    batch = corrupted_tensors[i : i + batch_size].to(device)
                    out = model(batch)
                    probs = torch.softmax(out, dim=-1)
                    conf, pred = torch.max(probs, dim=-1)
                    corr_preds.extend(pred.cpu().numpy().tolist())
                    corr_confs.extend(conf.cpu().numpy().tolist())

            acc = float(np.mean(np.array(corr_preds) == labels))
            mean_conf = float(np.mean(corr_confs))
            results[corr][sev] = {"accuracy": acc, "mean_confidence": mean_conf}
            print(f"  [{corr:<20} | Severity {sev}] Acc: {acc * 100:5.2f}% | Conf: {mean_conf * 100:5.2f}% | Drop: {(clean_acc - acc) * 100:5.2f}%")

    return results


def plot_robustness_curves(
    results: Dict[str, Dict[int, Dict[str, float]]],
    save_path: Optional[str] = None,
    title: str = "Model Robustness Under Corruptions (Severity 0 to 3)",
) -> plt.Figure:
    """Plot accuracy degradation curves across corruption severities."""
    clean_acc = results["clean"][0]["accuracy"]

    fig, ax = plt.subplots(figsize=(10, 6))

    markers = ["o", "s", "^", "D", "v", "<", ">", "p"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(results) - 1))

    corr_idx = 0
    for corr, sev_dict in results.items():
        if corr == "clean":
            continue
        severities = [0] + sorted(sev_dict.keys())
        accuracies = [clean_acc * 100] + [sev_dict[s]["accuracy"] * 100 for s in sorted(sev_dict.keys())]

        ax.plot(
            severities,
            accuracies,
            marker=markers[corr_idx % len(markers)],
            color=colors[corr_idx],
            linewidth=2.0,
            markersize=7,
            label=corr.replace("_", " ").title(),
        )
        corr_idx += 1

    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["0 (Clean)", "1 (Mild)", "2 (Moderate)", "3 (Severe)"], fontsize=11)
    ax.set_xlabel("Corruption Severity Level", fontsize=12, weight="bold")
    ax.set_ylabel("Classification Accuracy (%)", fontsize=12, weight="bold")
    ax.set_ylim(-5, 105)
    ax.set_title(title, fontsize=14, weight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", framealpha=0.9, fontsize=10)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_corruption_panel(
    sample_image: Image.Image,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Create a 3x3 visual panel showing clean leaf vs all 8 corruptions at severity 2."""
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()

    # Panel 0: Clean
    axes[0].imshow(sample_image)
    axes[0].set_title("Clean Original", fontsize=12, weight="bold")
    axes[0].axis("off")

    corr_keys = list(CORRUPTION_REGISTRY.keys())
    for i, corr in enumerate(corr_keys[:8]):
        corrupted = apply_corruption(sample_image, corr, severity=2)
        axes[i + 1].imshow(corrupted)
        axes[i + 1].set_title(corr.replace("_", " ").title() + " (Sev 2)", fontsize=11, weight="bold")
        axes[i + 1].axis("off")

    plt.suptitle("Visual Corruption Modes (Tomato Leaf at Moderate Severity)", fontsize=14, weight="bold", y=0.98)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_fgsm_curve(
    fgsm_results: Dict[float, Dict[str, float]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot accuracy vs FGSM perturbation budget epsilon."""
    epsilons = sorted(fgsm_results.keys())
    accuracies = [fgsm_results[e]["accuracy"] * 100 for e in epsilons]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epsilons, accuracies, marker="o", color="#c0392b", linewidth=2.5, markersize=8)
    ax.set_xlabel("Adversarial Perturbation Budget ($\epsilon$)", fontsize=12, weight="bold")
    ax.set_ylabel("Classification Accuracy (%)", fontsize=12, weight="bold")
    ax.set_title("FGSM Adversarial Attack Robustness Curve", fontsize=13, weight="bold", pad=12)
    ax.set_ylim(-5, 105)
    ax.grid(True, linestyle=":", alpha=0.6)

    for eps, acc in zip(epsilons, accuracies):
        ax.annotate(f"{acc:.1f}%", (eps, acc), textcoords="offset points", xytext=(0, 8), ha="center", weight="bold")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig
