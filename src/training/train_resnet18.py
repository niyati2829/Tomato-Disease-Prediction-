"""Two-Phase Transfer Learning Pipeline for ResNet18 on Tomato Leaf Disease.

This module executes:
1. Phase A (Feature Extraction):
   - Backbone is completely frozen.
   - Trains newly initialized fc layer for N epochs (default 5).
   - Optimizer: Adam(lr=0.001).
2. Phase B (Fine-Tuning):
   - Unfreezes high-level residual block (layer4) and fc layer.
   - Low- and mid-level feature extractors (conv1, layer1, layer2, layer3) remain frozen.
   - Optimizer: Adam with differential learning rates (1e-4 for layer4, 1e-3 for fc).
3. Checkpoint Management:
   - Saves best model based on validation loss to `artifacts/checkpoints/best_resnet18.pt`.
   - Never overwrites baseline CNN checkpoints.
4. History and Visualizations:
   - Saves metrics history to `reports/resnet18_history.json`.
   - Produces learning curves in `artifacts/figures/resnet18_*.png`.

Author: Crop Disease Detection Project Team
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

# Ensure repository root is on sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from src.data.dataset import create_dataloaders
from src.evaluation.plots import plot_learning_curves
from src.models.transfer_learning import (
    count_trainable_parameters,
    freeze_backbone,
    get_fine_tuning_param_groups,
    get_resnet18,
    unfreeze_layer4_and_fc,
)
from src.training.train import train_one_epoch, validate_one_epoch
from src.utils.device import get_device, get_device_info
from src.utils.reproducibility import set_seed


def run_resnet18_sanity_check(
    config_path: Path | str | None = None,
) -> Dict[str, any]:
    """Perform a single mini-batch forward, loss, backward, and optimization sanity check on ResNet18.

    Verifies:
    1. ResNet18 forward pass outputs (batch_size, 10).
    2. Initial cross-entropy loss is near theoretical ~2.30.
    3. Frozen backbone parameters receive zero gradient.
    4. Classification layer updates cleanly on Apple Silicon MPS/CPU.

    Returns:
        Dictionary of sanity check diagnostics.
    """
    if config_path is None:
        config_path = project_root / "configs" / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = config.get("seed", 42)
    set_seed(seed)

    device = get_device(preferred=config.get("device", {}).get("preferred", "mps"))
    device_info = get_device_info()

    model = get_resnet18(num_classes=10, pretrained=True, freeze_backbone_weights=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    dataloaders = create_dataloaders(
        splits_dir=project_root / "data" / "splits",
        batch_size=8,
        num_workers=0,
        root_dir=project_root,
    )

    images, labels = next(iter(dataloaders["train"]))
    images = images.to(device)
    labels = labels.to(device)

    initial_fc_weight = model.fc.weight.clone()

    optimizer.zero_grad()
    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()

    # Verify frozen layers have no gradients
    frozen_grad_none = (model.conv1.weight.grad is None) and (model.layer1[0].conv1.weight.grad is None)

    optimizer.step()
    updated_fc_weight = model.fc.weight
    fc_delta = torch.norm(initial_fc_weight - updated_fc_weight).item()

    _, preds = torch.max(logits, dim=1)
    acc = ((preds == labels).sum().item() / labels.size(0)) * 100.0

    return {
        "status": "success",
        "device": str(device),
        "device_name": device_info.get("device_name", "Unknown"),
        "input_shape": list(images.shape),
        "logits_shape": list(logits.shape),
        "initial_loss": float(loss.item()),
        "batch_accuracy": float(acc),
        "fc_weight_delta": float(fc_delta),
        "fc_updated": bool(fc_delta > 0.0),
        "frozen_backbone_verified": frozen_grad_none,
    }


def train_resnet18(
    config_path: Path | str | None = None,
    phase_a_epochs_override: int | None = None,
    phase_b_epochs_override: int | None = None,
) -> Dict[str, any]:
    """Execute complete two-phase transfer learning and fine-tuning for ResNet18.

    Args:
        config_path: Path to config.yaml.
        phase_a_epochs_override: Optional override for Phase A epochs.
        phase_b_epochs_override: Optional override for Phase B epochs.

    Returns:
        Dictionary of training results, best metrics, and paths.
    """
    if config_path is None:
        config_path = project_root / "configs" / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 1. Reproducibility
    seed = config.get("seed", 42)
    set_seed(seed)

    # 2. Hardware Device
    preferred_device = config.get("device", {}).get("preferred", "mps")
    device = get_device(preferred=preferred_device)
    device_info = get_device_info(preferred=preferred_device)

    # 3. Directories & Paths
    checkpoints_dir = project_root / config["logging"]["checkpoints_dir"]
    figures_dir = project_root / config["logging"]["figures_dir"]
    reports_dir = project_root / "reports"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 4. Hyperparameters
    res_cfg = config.get("resnet18", {})
    num_classes = res_cfg.get("num_classes", 10)
    batch_size = res_cfg.get("batch_size", 32)
    num_workers = res_cfg.get("num_workers", 0)
    lr_fc = res_cfg.get("lr_fc", 0.001)
    lr_layer4 = res_cfg.get("lr_layer4", 0.0001)
    weight_decay = res_cfg.get("weight_decay", 0.0001)
    epochs_phase_a = phase_a_epochs_override if phase_a_epochs_override is not None else res_cfg.get("feature_extraction_epochs", 5)
    epochs_phase_b = phase_b_epochs_override if phase_b_epochs_override is not None else res_cfg.get("finetuning_epochs", 5)
    total_epochs = epochs_phase_a + epochs_phase_b

    best_checkpoint_path = project_root / res_cfg.get("checkpoint", "artifacts/checkpoints/best_resnet18.pt")
    history_path = project_root / res_cfg.get("history", "reports/resnet18_history.json")

    print("=" * 78)
    print("STAGE 4: RESNET18 TWO-PHASE TRANSFER LEARNING PIPELINE")
    print("=" * 78)
    print(f"Device: {device} ({device_info.get('device_name', 'Host')})")
    print(f"Total Epochs: {total_epochs} (Phase A Feature Extraction: {epochs_phase_a} | Phase B Fine-Tuning: {epochs_phase_b})")
    print(f"Batch Size: {batch_size} | Weight Decay: {weight_decay}")
    print(f"Learning Rates -> Phase A: fc={lr_fc} | Phase B: layer4={lr_layer4}, fc={lr_fc}")
    print("-" * 78)

    # 5. DataLoaders (Train and Val only; Test set remains strictly untouched)
    splits_dir = project_root / "data" / "splits"
    dataloaders = create_dataloaders(
        splits_dir=splits_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        root_dir=project_root,
    )
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    # 6. Initialize Model in Phase A (Frozen Backbone)
    model = get_resnet18(num_classes=num_classes, pretrained=True, freeze_backbone_weights=True).to(device)
    total_params, trainable_params_a = count_trainable_parameters(model)
    print(f"Pretrained Model: ResNet18 (ImageNet Weights)")
    print(f"Total Parameters: {total_params:,}")
    print(f"Phase A Active Trainable Parameters: {trainable_params_a:,} (Backbone Frozen, fc only)")
    print("-" * 78)

    criterion = nn.CrossEntropyLoss()
    optimizer_phase_a = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_fc,
        weight_decay=weight_decay,
    )

    history = {
        "epoch": [],
        "phase": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "epoch_duration": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0
    start_total_time = time.time()

    # ---------------------------------------------------------
    # PHASE A: FEATURE EXTRACTION (Epochs 1 to epochs_phase_a)
    # ---------------------------------------------------------
    print(">>> STARTING PHASE A: FEATURE EXTRACTION (Train linear head only)")
    for epoch in range(1, epochs_phase_a + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer_phase_a, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)

        duration = time.time() - epoch_start

        history["epoch"].append(epoch)
        history["phase"].append("A_feature_extraction")
        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["train_acc"].append(round(train_acc, 2))
        history["val_acc"].append(round(val_acc, 2))
        history["epoch_duration"].append(round(duration, 2))

        is_best = val_loss < best_val_loss
        best_marker = ""
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            best_marker = " ★ (Best ResNet18 Saved)"
            torch.save({
                "epoch": epoch,
                "phase": "A_feature_extraction",
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer_phase_a.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "config": config,
            }, best_checkpoint_path)

        print(
            f"[Phase A] Epoch [{epoch:>2}/{total_epochs:>2}] | "
            f"Train Loss: {train_loss:.4f} - Acc: {train_acc:>5.2f}% | "
            f"Val Loss: {val_loss:.4f} - Acc: {val_acc:>5.2f}% | "
            f"Time: {duration:>4.1f}s{best_marker}"
        )

    # ---------------------------------------------------------
    # PHASE B: FINE-TUNING (Epochs epochs_phase_a + 1 to total_epochs)
    # ---------------------------------------------------------
    print("\n" + "-" * 78)
    print(">>> TRANSITIONING TO PHASE B: FINE-TUNING (Unfreezing layer4 + fc)")
    unfreeze_layer4_and_fc(model)
    _, trainable_params_b = count_trainable_parameters(model)
    print(f"Phase B Active Trainable Parameters: {trainable_params_b:,} (conv1-layer3 remain frozen)")

    param_groups_phase_b = get_fine_tuning_param_groups(
        model,
        lr_fc=lr_fc,
        lr_layer4=lr_layer4,
        weight_decay=weight_decay,
    )
    optimizer_phase_b = optim.Adam(param_groups_phase_b)

    for epoch in range(epochs_phase_a + 1, total_epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer_phase_b, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)

        duration = time.time() - epoch_start

        history["epoch"].append(epoch)
        history["phase"].append("B_finetuning")
        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["train_acc"].append(round(train_acc, 2))
        history["val_acc"].append(round(val_acc, 2))
        history["epoch_duration"].append(round(duration, 2))

        is_best = val_loss < best_val_loss
        best_marker = ""
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            best_marker = " ★ (Best ResNet18 Saved)"
            torch.save({
                "epoch": epoch,
                "phase": "B_finetuning",
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer_phase_b.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "config": config,
            }, best_checkpoint_path)

        print(
            f"[Phase B] Epoch [{epoch:>2}/{total_epochs:>2}] | "
            f"Train Loss: {train_loss:.4f} - Acc: {train_acc:>5.2f}% | "
            f"Val Loss: {val_loss:.4f} - Acc: {val_acc:>5.2f}% | "
            f"Time: {duration:>4.1f}s{best_marker}"
        )

    total_training_time = time.time() - start_total_time
    print("-" * 78)
    print(f"ResNet18 Training Complete in {total_training_time/60:.2f} minutes.")
    print(f"Best Validation Loss: {best_val_loss:.4f} at Epoch {best_epoch}")
    print(f"Best Checkpoint: {best_checkpoint_path}")

    # 7. Export History & Plots
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    plot_paths = plot_learning_curves(history, figures_dir, prefix="resnet18")
    print(f"Saved ResNet18 Learning Curves to:")
    for k, p in plot_paths.items():
        print(f"  - {k.title()}: {p}")
    print("=" * 78)

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "final_val_acc": history["val_acc"][-1],
        "total_time_seconds": total_training_time,
        "checkpoint": str(best_checkpoint_path),
        "history_file": str(history_path),
        "figures": {k: str(v) for k, v in plot_paths.items()},
    }


if __name__ == "__main__":
    train_resnet18()

