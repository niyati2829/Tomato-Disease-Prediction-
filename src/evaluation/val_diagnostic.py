"""Stage 5 — Validation-Only Diagnostic Audit.

Runs inference on data/splits/val.csv ONLY using both saved checkpoints:
  - artifacts/checkpoints/best_baseline_cnn.pt
  - artifacts/checkpoints/best_resnet18.pt

Computes:
  - Overall accuracy
  - Per-class precision, recall, F1-score
  - Macro and weighted averages
  - Confusion matrix (saved as PNG)
  - Per-class counts for imbalance investigation

Saves results to:
  - reports/val_diagnostic_baseline_cnn.json
  - reports/val_diagnostic_resnet18.json
  - artifacts/figures/confusion_matrix_baseline_cnn.png
  - artifacts/figures/confusion_matrix_resnet18.png

Does NOT touch data/splits/test.csv.
Does NOT modify checkpoints or dataset splits.

Author: Crop Disease Detection Project Team
"""

import json
import os
import sys
from pathlib import Path

# Fix matplotlib cache in sandboxed/restricted environments
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# Add project root to sys.path so src.* imports resolve
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from src.data.dataset import PlantVillageTomatoDataset, get_transforms
from src.models.cnn import BaselineCNN
from src.models.transfer_learning import TomatoResNet18
import torchvision.models as tv_models


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VAL_CSV = PROJECT_ROOT / "data/splits/val.csv"
CLASS_MAPPING_JSON = PROJECT_ROOT / "data/splits/class_mapping.json"
CHECKPOINTS = {
    "baseline_cnn": PROJECT_ROOT / "artifacts/checkpoints/best_baseline_cnn.pt",
    "resnet18": PROJECT_ROOT / "artifacts/checkpoints/best_resnet18.pt",
}
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "artifacts/figures"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 64
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 10


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def load_baseline_cnn(checkpoint_path: Path, device: torch.device) -> BaselineCNN:
    """Load BaselineCNN from saved checkpoint."""
    model = BaselineCNN(num_classes=NUM_CLASSES)
    ckpt = torch.load(checkpoint_path, map_location=device)
    # Checkpoint may store raw state_dict or a dict with model_state_dict key
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"  [BaselineCNN] Loaded from {checkpoint_path.name}")
    if "epoch" in ckpt:
        print(f"  [BaselineCNN] Checkpoint epoch: {ckpt['epoch']+1}, val_loss: {ckpt.get('val_loss', 'N/A'):.4f}")
    return model


def load_resnet18(checkpoint_path: Path, device: torch.device) -> TomatoResNet18:
    """Load TomatoResNet18 from saved checkpoint (Phase B state)."""
    model = TomatoResNet18(num_classes=NUM_CLASSES, weights=None)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"  [ResNet18] Loaded from {checkpoint_path.name}")
    print(f"  [ResNet18] Checkpoint epoch: {ckpt.get('epoch', '?') + 1}, "
          f"phase: {ckpt.get('phase', '?')}, "
          f"val_loss: {ckpt.get('val_loss', 0):.4f}, "
          f"val_acc: {ckpt.get('val_acc', 0):.2f}%")
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    """Run inference on a DataLoader; return (y_true, y_pred) as numpy arrays."""
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.numpy())
            if (batch_idx + 1) % 10 == 0:
                print(f"    Batch {batch_idx+1}/{len(loader)} processed...")

    return np.concatenate(all_labels), np.concatenate(all_preds)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list) -> dict:
    """Compute comprehensive classification metrics."""
    overall_acc = accuracy_score(y_true, y_pred) * 100

    # Per-class metrics (zero_division=0 avoids warnings for rare classes)
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    per_class = {}
    for idx, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(per_class_precision[idx]),
            "recall": float(per_class_recall[idx]),
            "f1": float(per_class_f1[idx]),
        }

    return {
        "overall_accuracy": float(overall_acc),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
    }


# ---------------------------------------------------------------------------
# Confusion matrix plot
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    title: str,
    save_path: Path,
) -> None:
    """Plot and save a normalised confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # Row-normalise

    # Shorten class names for readability on the plot axes
    short_names = [
        name.replace("Tomato___", "").replace("_", " ").replace(" Two-spotted spider mite", "\n(Spider mite)")
        for name in class_names
    ]

    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Recall (row-normalised)", fontsize=10)

    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(short_names, fontsize=9)
    ax.set_xlabel("Predicted Label", fontsize=11, labelpad=10)
    ax.set_ylabel("True Label", fontsize=11, labelpad=10)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)

    # Annotate each cell with count and percentage
    thresh = cm_norm.max() / 2.0
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            color = "white" if cm_norm[i, j] > thresh else "black"
            ax.text(
                j, i,
                f"{cm_norm[i, j]:.2f}\n({cm[i, j]})",
                ha="center", va="center",
                fontsize=7, color=color,
            )

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix saved → {save_path.name}")


# ---------------------------------------------------------------------------
# Main diagnostic pipeline
# ---------------------------------------------------------------------------

def run_diagnostic():
    print("\n" + "=" * 70)
    print("STAGE 5 — VALIDATION DIAGNOSTIC AUDIT")
    print("=" * 70)

    # --- Sanity check: test.csv must NOT be loaded
    print(f"\n[Safety] Val CSV: {VAL_CSV}")
    assert "val" in str(VAL_CSV), "SAFETY: val.csv path check failed!"
    assert "test" not in str(VAL_CSV), "SAFETY: test.csv must not be loaded here!"

    # --- Load class mapping
    with open(CLASS_MAPPING_JSON) as f:
        class_to_idx = json.load(f)
    class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]
    print(f"\n[Classes] {len(class_names)} classes loaded (alphabetical index order).")

    # --- Build validation DataLoader (deterministic, no augmentation)
    val_dataset = PlantVillageTomatoDataset(
        split_csv=VAL_CSV,
        root_dir=PROJECT_ROOT,
        transform=get_transforms(image_size=IMAGE_SIZE, is_training=False),
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"[Val set] {len(val_dataset)} images, {len(val_loader)} batches.")

    # --- Class distribution in val set
    val_df = pd.read_csv(VAL_CSV)
    val_counts = val_df["class_name"].value_counts().sort_index()
    print("\n[Val class distribution]")
    for name, count in val_counts.items():
        print(f"  {name:52s} {count:5d} images")

    device = get_device()
    print(f"\n[Device] {device}")

    all_results = {}

    # -----------------------------------------------------------------------
    # 1. BaselineCNN
    # -----------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("RUNNING: BaselineCNN")
    print("-" * 60)
    cnn_model = load_baseline_cnn(CHECKPOINTS["baseline_cnn"], device)
    y_true_cnn, y_pred_cnn = run_inference(cnn_model, val_loader, device)
    del cnn_model  # Free memory before loading ResNet

    cnn_metrics = compute_metrics(y_true_cnn, y_pred_cnn, class_names)
    all_results["baseline_cnn"] = cnn_metrics

    # Save JSON
    cnn_report_path = REPORTS_DIR / "val_diagnostic_baseline_cnn.json"
    with open(cnn_report_path, "w") as f:
        json.dump(cnn_metrics, f, indent=2)
    print(f"\n  Metrics saved → {cnn_report_path.name}")

    # Confusion matrix
    plot_confusion_matrix(
        y_true_cnn, y_pred_cnn, class_names,
        title="BaselineCNN — Validation Confusion Matrix (row-normalised recall)",
        save_path=FIGURES_DIR / "confusion_matrix_baseline_cnn.png",
    )

    # -----------------------------------------------------------------------
    # 2. ResNet18
    # -----------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("RUNNING: ResNet18")
    print("-" * 60)
    rn18_model = load_resnet18(CHECKPOINTS["resnet18"], device)
    y_true_rn, y_pred_rn = run_inference(rn18_model, val_loader, device)
    del rn18_model

    rn18_metrics = compute_metrics(y_true_rn, y_pred_rn, class_names)
    all_results["resnet18"] = rn18_metrics

    rn18_report_path = REPORTS_DIR / "val_diagnostic_resnet18.json"
    with open(rn18_report_path, "w") as f:
        json.dump(rn18_metrics, f, indent=2)
    print(f"\n  Metrics saved → {rn18_report_path.name}")

    plot_confusion_matrix(
        y_true_rn, y_pred_rn, class_names,
        title="ResNet18 — Validation Confusion Matrix (row-normalised recall)",
        save_path=FIGURES_DIR / "confusion_matrix_resnet18.png",
    )

    # -----------------------------------------------------------------------
    # 3. Side-by-side summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)

    header = f"{'Class':<52} {'CNN P':>7} {'CNN R':>7} {'CNN F1':>7} | {'RN18 P':>7} {'RN18 R':>7} {'RN18 F1':>7}"
    print(header)
    print("-" * len(header))
    for name in class_names:
        c = all_results["baseline_cnn"]["per_class"][name]
        r = all_results["resnet18"]["per_class"][name]
        print(
            f"{name:<52} {c['precision']:7.4f} {c['recall']:7.4f} {c['f1']:7.4f} | "
            f"{r['precision']:7.4f} {r['recall']:7.4f} {r['f1']:7.4f}"
        )
    print("-" * len(header))

    for key, label in [("baseline_cnn", "BaselineCNN"), ("resnet18", "ResNet18")]:
        m = all_results[key]
        print(f"\n{label}:")
        print(f"  Overall Accuracy : {m['overall_accuracy']:.4f}%")
        print(f"  Macro Precision  : {m['macro_precision']:.4f}")
        print(f"  Macro Recall     : {m['macro_recall']:.4f}")
        print(f"  Macro F1         : {m['macro_f1']:.4f}")
        print(f"  Weighted F1      : {m['weighted_f1']:.4f}")

    # Find weakest classes (by recall) for each model
    print("\n[Weakest classes by recall]")
    for key, label in [("baseline_cnn", "BaselineCNN"), ("resnet18", "ResNet18")]:
        per_class = all_results[key]["per_class"]
        sorted_by_recall = sorted(per_class.items(), key=lambda x: x[1]["recall"])
        print(f"  {label}:")
        for name, m in sorted_by_recall[:3]:
            short = name.replace("Tomato___", "")
            print(f"    {short:<45} recall={m['recall']:.4f}  f1={m['f1']:.4f}")

    # Save train class counts for imbalance analysis
    train_df = pd.read_csv(PROJECT_ROOT / "data/splits/train.csv")
    train_counts = train_df["class_name"].value_counts().to_dict()
    imbalance_data = {
        "train_class_counts": train_counts,
        "val_class_counts": val_counts.to_dict(),
        "majority_class": max(train_counts, key=train_counts.get),
        "minority_class": min(train_counts, key=train_counts.get),
        "imbalance_ratio": max(train_counts.values()) / min(train_counts.values()),
    }
    imbalance_path = REPORTS_DIR / "class_imbalance_analysis.json"
    with open(imbalance_path, "w") as f:
        json.dump(imbalance_data, f, indent=2)
    print(f"\n[Imbalance] Ratio: {imbalance_data['imbalance_ratio']:.2f}x  "
          f"(majority: {imbalance_data['majority_class'].replace('Tomato___', '')}  "
          f"minority: {imbalance_data['minority_class'].replace('Tomato___', '')})")
    print(f"  Imbalance analysis saved → {imbalance_path.name}")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)
    return all_results


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    results = run_diagnostic()

