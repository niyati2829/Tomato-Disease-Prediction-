"""Stage 5 — Experiment B: Class-Weighted Loss + Stronger Augmentation.

INDEPENDENTLY loads the original best_resnet18.pt (Epoch 9, Phase B) —
does NOT start from Experiment A's checkpoint.

Applies:
  - Square-root inverse-frequency class weights (identical to Exp A).
  - Stronger training augmentation:
      * ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)
      * RandomResizedCrop(224, scale=(0.85, 1.0))
      * Keep existing RandomHorizontalFlip, RandomRotation(15)
  - Validation transforms remain deterministic (unchanged).
  - Layer4 lr = 5e-5, fc lr = 2e-4.
  - Up to 5 additional Phase B epochs.
  - Best checkpoint selected by Macro F1.

This experiment targets ONLY val.csv for evaluation.
test.csv is never accessed.

Saves:
  - artifacts/checkpoints/best_resnet18_aug_weighted.pt
  - reports/resnet18_exp_b_history.json
  - reports/val_diagnostic_resnet18_exp_b.json
  - artifacts/figures/confusion_matrix_resnet18_exp_b.png
  - artifacts/figures/resnet18_exp_b_learning_curves.png (etc.)

Author: Crop Disease Detection Project Team
"""

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.data.dataset import PlantVillageTomatoDataset, get_transforms
from src.evaluation.plots import plot_learning_curves
from src.models.transfer_learning import (
    TomatoResNet18,
    count_trainable_parameters,
    get_fine_tuning_param_groups,
    unfreeze_layer4_and_fc,
)
from src.utils.device import get_device, get_device_info
from src.utils.reproducibility import set_seed


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 42
NUM_CLASSES = 10
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
NUM_WORKERS = 0

LR_LAYER4 = 5e-5
LR_FC = 2e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 5

ORIGINAL_CHECKPOINT = PROJECT_ROOT / "artifacts/checkpoints/best_resnet18.pt"
EXP_B_CHECKPOINT = PROJECT_ROOT / "artifacts/checkpoints/best_resnet18_aug_weighted.pt"
EXP_B_HISTORY = PROJECT_ROOT / "reports/resnet18_exp_b_history.json"
EXP_B_DIAGNOSTIC = PROJECT_ROOT / "reports/val_diagnostic_resnet18_exp_b.json"
FIGURES_DIR = PROJECT_ROOT / "artifacts/figures"
CLASS_MAPPING_JSON = PROJECT_ROOT / "data/splits/class_mapping.json"
TRAIN_CSV = PROJECT_ROOT / "data/splits/train.csv"
VAL_CSV = PROJECT_ROOT / "data/splits/val.csv"

# Safety guard
assert "test" not in str(VAL_CSV), "SAFETY: test set must not be accessed in Experiment B!"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Stronger training augmentation (Experiment B specific)
# ---------------------------------------------------------------------------

def get_exp_b_train_transforms(image_size=(224, 224)) -> transforms.Compose:
    """Build Experiment B training transform with stronger augmentation.

    Additions vs original training transforms:
      - ColorJitter: Reduces over-reliance on exact colour tone so the model
        learns lesion SHAPE and TEXTURE rather than the shade of brown.
        This directly targets the Early blight / Late blight / Target Spot
        confusion trio which share similar warm-brown hues.
      - RandomResizedCrop: Provides scale variation; original images are
        already 256x256 so scale=(0.85, 1.0) gives mild zoom variation without
        losing critical disease lesion information from the edges.
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomResizedCrop(image_size[0], scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_deterministic_val_transforms(image_size=(224, 224)) -> transforms.Compose:
    """Deterministic validation transforms — identical to original."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Utility: compute sqrt inverse-frequency class weights (same as Exp A)
# ---------------------------------------------------------------------------

def compute_sqrt_inverse_freq_weights(train_csv: Path, num_classes: int) -> torch.Tensor:
    """Square-root inverse-frequency class weights.
    See train_exp_a.py docstring for rationale.
    """
    df = pd.read_csv(train_csv)
    total = len(df)
    counts = df.groupby("class_index").size().sort_index()
    assert len(counts) == num_classes
    weights = torch.zeros(num_classes, dtype=torch.float32)
    for class_idx, count in counts.items():
        weights[int(class_idx)] = (total / (num_classes * count)) ** 0.5
    return weights


# ---------------------------------------------------------------------------
# Utility: per-class metrics on val set
# ---------------------------------------------------------------------------

def run_val_inference_with_metrics(
    model: nn.Module,
    val_loader: DataLoader,
    class_names: list,
    device: torch.device,
) -> dict:
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    per_class_p = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_r = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    per_class = {}
    for idx, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(per_class_p[idx]),
            "recall": float(per_class_r[idx]),
            "f1": float(per_class_f1[idx]),
        }

    early_blight_recall = per_class.get("Tomato___Early_blight", {}).get("recall", 0.0)

    return {
        "overall_accuracy": float(accuracy_score(y_true, y_pred) * 100),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "early_blight_recall": float(early_blight_recall),
        "per_class": per_class,
        "y_true": y_true,
        "y_pred": y_pred,
    }


# ---------------------------------------------------------------------------
# Confusion matrix plot
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    import matplotlib.pyplot as plt
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    short_names = [n.replace("Tomato___", "").replace("_", " ") for n in class_names]
    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Greens", vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Recall (row-normalised)", fontsize=10)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_xlabel("Predicted Label", fontsize=11, labelpad=10)
    ax.set_ylabel("True Label", fontsize=11, labelpad=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=14)
    thresh = cm_norm.max() / 2.0
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            color = "white" if cm_norm[i, j] > thresh else "black"
            ax.text(j, i, f"{cm_norm[i, j]:.2f}\n({cm[i, j]})",
                    ha="center", va="center", fontsize=7, color=color)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix saved → {Path(save_path).name}")


# ---------------------------------------------------------------------------
# Main Experiment B training loop
# ---------------------------------------------------------------------------

def run_experiment_b():
    print("\n" + "=" * 72)
    print("STAGE 5 — EXPERIMENT B: WEIGHTED LOSS + STRONGER AUGMENTATION")
    print("Starting from: best_resnet18.pt (Epoch 9, Phase B) — INDEPENDENT")
    print("=" * 72)

    # Safety checks
    assert ORIGINAL_CHECKPOINT.exists(), f"Original checkpoint not found: {ORIGINAL_CHECKPOINT}"
    assert str(EXP_B_CHECKPOINT) != str(ORIGINAL_CHECKPOINT), \
        "SAFETY: Experiment B must save to a DIFFERENT checkpoint path!"
    print(f"[Safety] Original checkpoint: {ORIGINAL_CHECKPOINT.name}  — PRESERVED")
    print(f"[Safety] Experiment B saves to: {EXP_B_CHECKPOINT.name}")
    print(f"[Safety] Val set: {VAL_CSV} (test.csv NOT accessed)")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "reports").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "artifacts/checkpoints").mkdir(parents=True, exist_ok=True)

    set_seed(SEED)
    print(f"[Seed] {SEED}")

    device = get_device(preferred="mps")
    info = get_device_info(preferred="mps")
    print(f"[Device] {device} ({info.get('device_name', 'CPU')})")

    # Class names
    with open(CLASS_MAPPING_JSON) as f:
        class_to_idx = json.load(f)
    class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]

    # Class weights (identical formula to Exp A)
    class_weights = compute_sqrt_inverse_freq_weights(TRAIN_CSV, NUM_CLASSES)
    print("\n[Class Weights — sqrt inverse-frequency (same as Exp A)]")
    for idx, name in enumerate(class_names):
        short = name.replace("Tomato___", "")
        print(f"  idx={idx}  {short:<45}  weight={class_weights[idx]:.4f}")

    class_weights_device = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_device)

    # Build DataLoaders with STRONGER TRAINING TRANSFORMS
    exp_b_train_transform = get_exp_b_train_transforms(image_size=IMAGE_SIZE)
    val_transform = get_deterministic_val_transforms(image_size=IMAGE_SIZE)

    print(f"\n[Augmentation — Experiment B Training Transform]")
    print(f"  Resize({IMAGE_SIZE[0]}) → RandomResizedCrop(scale=(0.85,1.0))")
    print(f"  → RandomHorizontalFlip(0.5) → RandomRotation(15)")
    print(f"  → ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)")
    print(f"  → ToTensor → Normalize(ImageNet)")
    print(f"[Augmentation — Validation Transform] DETERMINISTIC (Resize, ToTensor, Normalize only)")

    train_dataset = PlantVillageTomatoDataset(
        split_csv=TRAIN_CSV,
        root_dir=PROJECT_ROOT,
        transform=exp_b_train_transform,
    )
    val_dataset = PlantVillageTomatoDataset(
        split_csv=VAL_CSV,
        root_dir=PROJECT_ROOT,
        transform=val_transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    print(f"[Data] Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # Load model from ORIGINAL checkpoint (independent of Exp A)
    print(f"\n[Model] Loading TomatoResNet18 from {ORIGINAL_CHECKPOINT.name}...")
    model = TomatoResNet18(num_classes=NUM_CLASSES, weights=None)
    ckpt = torch.load(ORIGINAL_CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    orig_epoch = ckpt.get("epoch", "?")
    orig_val_loss = ckpt.get("val_loss", float("nan"))
    orig_val_acc = ckpt.get("val_acc", float("nan"))
    print(f"  Original checkpoint epoch: {orig_epoch + 1 if isinstance(orig_epoch, int) else orig_epoch}")
    print(f"  Original val_loss: {orig_val_loss:.4f} | val_acc: {orig_val_acc:.2f}%")

    # Set Phase B
    unfreeze_layer4_and_fc(model)
    model.to(device)
    total_params, trainable_params = count_trainable_parameters(model)
    print(f"  Total params: {total_params:,} | Trainable (Phase B): {trainable_params:,}")

    param_groups = get_fine_tuning_param_groups(
        model,
        lr_fc=LR_FC,
        lr_layer4=LR_LAYER4,
        weight_decay=WEIGHT_DECAY,
    )
    optimizer = optim.Adam(param_groups)
    print(f"\n[Optimizer] Adam | layer4 lr={LR_LAYER4} | fc lr={LR_FC} | wd={WEIGHT_DECAY}")
    print(f"[Epochs] Up to {MAX_EPOCHS} Phase B continuation epochs")
    print("-" * 72)

    from src.training.train import train_one_epoch, validate_one_epoch

    history = {
        "epoch": [],
        "phase": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "macro_f1": [],
        "early_blight_recall": [],
        "epoch_duration": [],
    }

    best_macro_f1 = -1.0
    best_epoch = 0
    best_val_loss_at_best_f1 = float("inf")
    start_time = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        criterion_unweighted = nn.CrossEntropyLoss()
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion_unweighted, device)

        metrics = run_val_inference_with_metrics(model, val_loader, class_names, device)
        macro_f1 = metrics["macro_f1"]
        early_blight_recall = metrics["early_blight_recall"]
        duration = time.time() - epoch_start

        history["epoch"].append(epoch)
        history["phase"].append("B_aug_weighted_continuation")
        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["train_acc"].append(round(train_acc, 2))
        history["val_acc"].append(round(val_acc, 2))
        history["macro_f1"].append(round(macro_f1, 4))
        history["early_blight_recall"].append(round(early_blight_recall, 4))
        history["epoch_duration"].append(round(duration, 2))

        is_best = macro_f1 > best_macro_f1
        best_marker = ""
        if is_best:
            best_macro_f1 = macro_f1
            best_epoch = epoch
            best_val_loss_at_best_f1 = val_loss
            best_marker = " ★ BEST (Macro F1)"
            torch.save({
                "epoch": epoch,
                "phase": "B_aug_weighted_continuation",
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "macro_f1": macro_f1,
                "early_blight_recall": early_blight_recall,
                "per_class_metrics": metrics["per_class"],
                "config": {
                    "experiment": "B_aug_weighted_loss",
                    "seed": SEED,
                    "lr_layer4": LR_LAYER4,
                    "lr_fc": LR_FC,
                    "weight_decay": WEIGHT_DECAY,
                    "max_epochs": MAX_EPOCHS,
                    "weighting": "sqrt_inverse_frequency",
                    "augmentation": {
                        "RandomResizedCrop": "scale=(0.85, 1.0)",
                        "RandomHorizontalFlip": "p=0.5",
                        "RandomRotation": "degrees=15",
                        "ColorJitter": "brightness=0.2, contrast=0.2, saturation=0.1",
                    },
                    "original_checkpoint": str(ORIGINAL_CHECKPOINT),
                },
            }, EXP_B_CHECKPOINT)

        print(
            f"[Exp B] Epoch [{epoch}/{MAX_EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
            f"MacroF1: {macro_f1:.4f} | EarlyBR: {early_blight_recall:.4f} | "
            f"Time: {duration:.1f}s{best_marker}"
        )

    total_time = time.time() - start_time
    print("-" * 72)
    print(f"Experiment B Complete in {total_time / 60:.2f} minutes.")
    print(f"Best Macro F1: {best_macro_f1:.4f} at Epoch {best_epoch}")
    print(f"Best checkpoint: {EXP_B_CHECKPOINT}")

    with open(EXP_B_HISTORY, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved → {EXP_B_HISTORY.name}")

    plot_learning_curves(history, FIGURES_DIR, prefix="resnet18_exp_b")

    # Final per-class diagnostic on best checkpoint
    print("\n[Loading best checkpoint for final diagnostic...]")
    best_ckpt = torch.load(EXP_B_CHECKPOINT, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    final_metrics = run_val_inference_with_metrics(model, val_loader, class_names, device)

    y_true_final = final_metrics.pop("y_true")
    y_pred_final = final_metrics.pop("y_pred")

    with open(EXP_B_DIAGNOSTIC, "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"Final diagnostic saved → {EXP_B_DIAGNOSTIC.name}")

    plot_confusion_matrix(
        y_true_final, y_pred_final, class_names,
        title="Experiment B (Weighted Loss + Aug) — Validation Confusion Matrix",
        save_path=FIGURES_DIR / "confusion_matrix_resnet18_exp_b.png",
    )

    print("\n[Experiment B — Final Class Metrics at Best Epoch]")
    header = f"{'Class':<48} {'Precision':>10} {'Recall':>10} {'F1':>10}"
    print(header)
    print("-" * len(header))
    for name in class_names:
        m = final_metrics["per_class"][name]
        short = name.replace("Tomato___", "")
        print(f"  {short:<46} {m['precision']:10.4f} {m['recall']:10.4f} {m['f1']:10.4f}")
    print("-" * len(header))
    print(f"  {'Overall Accuracy':<46} {'':>10} {'':>10} {final_metrics['overall_accuracy']:9.4f}%")
    print(f"  {'Macro F1':<46} {'':>10} {'':>10} {final_metrics['macro_f1']:10.4f}")
    print(f"  {'Weighted F1':<46} {'':>10} {'':>10} {final_metrics['weighted_f1']:10.4f}")
    print(f"  {'Early Blight Recall':<46} {'':>10} {'':>10} {final_metrics['early_blight_recall']:10.4f}")

    print("=" * 72)
    return {
        "best_epoch": best_epoch,
        "best_macro_f1": best_macro_f1,
        "val_loss_at_best": best_val_loss_at_best_f1,
        "overall_accuracy": final_metrics["overall_accuracy"],
        "early_blight_recall": final_metrics["early_blight_recall"],
        "checkpoint": str(EXP_B_CHECKPOINT),
    }


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    run_experiment_b()

