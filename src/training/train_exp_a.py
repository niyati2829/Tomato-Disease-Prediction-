"""Stage 5 — Experiment A: Class-Weighted CrossEntropyLoss Fine-Tuning.

Loads the original best_resnet18.pt (Epoch 9, Phase B) and continues
Phase B fine-tuning with:
  - Square-root inverse-frequency class weights in CrossEntropyLoss.
  - Layer4 lr = 5e-5, fc lr = 2e-4 (lower than original).
  - Up to 5 additional epochs.
  - Best checkpoint selected by Macro F1 (not val loss) to prioritise
    class-wise balance over aggregate loss.

This experiment targets ONLY val.csv for evaluation.
test.csv is never accessed.

Saves:
  - artifacts/checkpoints/best_resnet18_weighted.pt
  - reports/resnet18_exp_a_history.json
  - reports/val_diagnostic_resnet18_exp_a.json
  - artifacts/figures/confusion_matrix_resnet18_exp_a.png
  - artifacts/figures/resnet18_exp_a_learning_curves.png (etc.)

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
from torch.utils.data import DataLoader

from src.data.dataset import PlantVillageTomatoDataset, create_dataloaders, get_transforms
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

# Experiment A — Phase B continuation learning rates
LR_LAYER4 = 5e-5
LR_FC = 2e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 5

ORIGINAL_CHECKPOINT = PROJECT_ROOT / "artifacts/checkpoints/best_resnet18.pt"
EXP_A_CHECKPOINT = PROJECT_ROOT / "artifacts/checkpoints/best_resnet18_weighted.pt"
EXP_A_HISTORY = PROJECT_ROOT / "reports/resnet18_exp_a_history.json"
EXP_A_DIAGNOSTIC = PROJECT_ROOT / "reports/val_diagnostic_resnet18_exp_a.json"
FIGURES_DIR = PROJECT_ROOT / "artifacts/figures"
CLASS_MAPPING_JSON = PROJECT_ROOT / "data/splits/class_mapping.json"
TRAIN_CSV = PROJECT_ROOT / "data/splits/train.csv"
VAL_CSV = PROJECT_ROOT / "data/splits/val.csv"

# Safety guard — test CSV must never be accessed
assert "test" not in str(VAL_CSV), "SAFETY: test set must not be accessed in Experiment A!"


# ---------------------------------------------------------------------------
# Utility: compute sqrt inverse-frequency class weights
# ---------------------------------------------------------------------------

def compute_sqrt_inverse_freq_weights(train_csv: Path, num_classes: int) -> torch.Tensor:
    """Compute square-root inverse-frequency class weights.

    Why sqrt instead of raw inverse-frequency?
    Raw inverse-frequency gives Mosaic Virus (261 samples) a weight of 4.88x.
    But Mosaic Virus already achieves 100% recall — aggressively upweighting
    a perfectly-classified class wastes gradient capacity and may destabilise
    training for other classes.

    Square-root dampens extreme weights while still giving minority classes
    meaningful upweighting:
        weight_c = sqrt(total / (num_classes * count_c))

    Args:
        train_csv: Path to training split CSV.
        num_classes: Number of target classes.

    Returns:
        FloatTensor of shape (num_classes,) with class weights.
    """
    df = pd.read_csv(train_csv)
    total = len(df)

    # Build an ordered list (index 0 to 9) using class_index column
    counts = df.groupby("class_index").size().sort_index()
    assert len(counts) == num_classes, f"Expected {num_classes} classes, found {len(counts)}"

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
    """Run full val-set inference and return comprehensive per-class metrics."""
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
# Confusion matrix plot (import from val_diagnostic or inline here)
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    import matplotlib.pyplot as plt
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    short_names = [
        n.replace("Tomato___", "").replace("_", " ")
        for n in class_names
    ]
    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
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
# Main Experiment A training loop
# ---------------------------------------------------------------------------

def run_experiment_a():
    print("\n" + "=" * 72)
    print("STAGE 5 — EXPERIMENT A: CLASS-WEIGHTED CROSSENTROPYLOSS")
    print("Starting from: best_resnet18.pt (Epoch 9, Phase B)")
    print("=" * 72)

    # Safety
    assert ORIGINAL_CHECKPOINT.exists(), f"Original checkpoint not found: {ORIGINAL_CHECKPOINT}"
    assert "best_resnet18.pt" in str(EXP_A_CHECKPOINT) or "weighted" in str(EXP_A_CHECKPOINT)
    assert str(EXP_A_CHECKPOINT) != str(ORIGINAL_CHECKPOINT), \
        "SAFETY: Experiment A must save to a DIFFERENT checkpoint path!"
    print(f"[Safety] Original checkpoint: {ORIGINAL_CHECKPOINT.name}  — PRESERVED")
    print(f"[Safety] Experiment A saves to: {EXP_A_CHECKPOINT.name}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "reports").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "artifacts/checkpoints").mkdir(parents=True, exist_ok=True)

    # Seed
    set_seed(SEED)
    print(f"[Seed] {SEED}")

    # Device
    device = get_device(preferred="mps")
    info = get_device_info(preferred="mps")
    print(f"[Device] {device} ({info.get('device_name', 'CPU')})")

    # Class names (alphabetical index order)
    with open(CLASS_MAPPING_JSON) as f:
        class_to_idx = json.load(f)
    class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]
    print(f"[Classes] {class_names}")

    # Class weights
    class_weights = compute_sqrt_inverse_freq_weights(TRAIN_CSV, NUM_CLASSES)
    print("\n[Class Weights — sqrt inverse-frequency]")
    for idx, name in enumerate(class_names):
        short = name.replace("Tomato___", "")
        print(f"  idx={idx}  {short:<45}  weight={class_weights[idx]:.4f}")

    class_weights_device = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_device)
    print(f"\n[Loss] CrossEntropyLoss with class weights (sqrt inverse-frequency)")

    # DataLoaders (standard transforms — no augmentation changes in Exp A)
    dataloaders = create_dataloaders(
        splits_dir=PROJECT_ROOT / "data/splits",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        root_dir=PROJECT_ROOT,
    )
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    print(f"[Data] Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    # Load model from original checkpoint
    print(f"\n[Model] Loading TomatoResNet18 from {ORIGINAL_CHECKPOINT.name}...")
    model = TomatoResNet18(num_classes=NUM_CLASSES, weights=None)
    ckpt = torch.load(ORIGINAL_CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    orig_epoch = ckpt.get("epoch", "?")
    orig_val_loss = ckpt.get("val_loss", float("nan"))
    orig_val_acc = ckpt.get("val_acc", float("nan"))
    print(f"  Original checkpoint epoch: {orig_epoch + 1 if isinstance(orig_epoch, int) else orig_epoch}")
    print(f"  Original val_loss: {orig_val_loss:.4f} | val_acc: {orig_val_acc:.2f}%")
    print(f"  Original phase: {ckpt.get('phase', '?')}")

    # Set Phase B (layer4 + fc trainable, early layers frozen)
    unfreeze_layer4_and_fc(model)
    model.to(device)
    total_params, trainable_params = count_trainable_parameters(model)
    print(f"  Total params: {total_params:,} | Trainable (Phase B): {trainable_params:,}")

    # Optimizer with differential learning rates
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

    # Import train/validate helpers from existing module
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

        # Training step (uses weighted criterion)
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validation — standard unweighted loss for comparability
        criterion_unweighted = nn.CrossEntropyLoss()
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion_unweighted, device)

        # Per-class metrics
        metrics = run_val_inference_with_metrics(model, val_loader, class_names, device)
        macro_f1 = metrics["macro_f1"]
        early_blight_recall = metrics["early_blight_recall"]
        duration = time.time() - epoch_start

        history["epoch"].append(epoch)
        history["phase"].append("B_weighted_continuation")
        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["train_acc"].append(round(train_acc, 2))
        history["val_acc"].append(round(val_acc, 2))
        history["macro_f1"].append(round(macro_f1, 4))
        history["early_blight_recall"].append(round(early_blight_recall, 4))
        history["epoch_duration"].append(round(duration, 2))

        # Best checkpoint: selected by Macro F1
        is_best = macro_f1 > best_macro_f1
        best_marker = ""
        if is_best:
            best_macro_f1 = macro_f1
            best_epoch = epoch
            best_val_loss_at_best_f1 = val_loss
            best_marker = " ★ BEST (Macro F1)"
            torch.save({
                "epoch": epoch,
                "phase": "B_weighted_continuation",
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "macro_f1": macro_f1,
                "early_blight_recall": early_blight_recall,
                "per_class_metrics": metrics["per_class"],
                "config": {
                    "experiment": "A_weighted_loss",
                    "seed": SEED,
                    "lr_layer4": LR_LAYER4,
                    "lr_fc": LR_FC,
                    "weight_decay": WEIGHT_DECAY,
                    "max_epochs": MAX_EPOCHS,
                    "weighting": "sqrt_inverse_frequency",
                    "original_checkpoint": str(ORIGINAL_CHECKPOINT),
                },
            }, EXP_A_CHECKPOINT)

        print(
            f"[Exp A] Epoch [{epoch}/{MAX_EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
            f"MacroF1: {macro_f1:.4f} | EarlyBR: {early_blight_recall:.4f} | "
            f"Time: {duration:.1f}s{best_marker}"
        )

    total_time = time.time() - start_time
    print("-" * 72)
    print(f"Experiment A Complete in {total_time / 60:.2f} minutes.")
    print(f"Best Macro F1: {best_macro_f1:.4f} at Epoch {best_epoch}")
    print(f"Best checkpoint: {EXP_A_CHECKPOINT}")

    # Save history
    with open(EXP_A_HISTORY, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved → {EXP_A_HISTORY.name}")

    # Learning curves
    plot_learning_curves(history, FIGURES_DIR, prefix="resnet18_exp_a")

    # Final per-class diagnostic on best checkpoint
    print("\n[Loading best checkpoint for final diagnostic...]")
    best_ckpt = torch.load(EXP_A_CHECKPOINT, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    final_metrics = run_val_inference_with_metrics(model, val_loader, class_names, device)

    # Remove numpy arrays before JSON serialisation
    y_true_final = final_metrics.pop("y_true")
    y_pred_final = final_metrics.pop("y_pred")

    with open(EXP_A_DIAGNOSTIC, "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"Final diagnostic saved → {EXP_A_DIAGNOSTIC.name}")

    # Confusion matrix
    plot_confusion_matrix(
        y_true_final, y_pred_final, class_names,
        title="Experiment A (Weighted Loss) — Validation Confusion Matrix",
        save_path=FIGURES_DIR / "confusion_matrix_resnet18_exp_a.png",
    )

    # Print summary
    print("\n[Experiment A — Final Class Metrics at Best Epoch]")
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
        "checkpoint": str(EXP_A_CHECKPOINT),
    }


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    run_experiment_a()

