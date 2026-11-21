"""Model Training and Validation Pipeline for Crop Disease Detection.

This module provides:
1. `train_one_epoch()`: Forward pass, CrossEntropyLoss, backprop, optimizer step, running metrics.
2. `validate_one_epoch()`: Evaluation loop with `torch.no_grad()`.
3. `train_model()`: Epoch orchestration, learning curve tracking, best model checkpointing.
4. `run_one_batch_sanity_check()`: Fast, deterministic single-batch sanity check.

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
from torch.utils.data import DataLoader

from src.data.dataset import create_dataloaders
from src.evaluation.plots import plot_learning_curves
from src.models.cnn import BaselineCNN, count_parameters
from src.utils.device import get_device, get_device_info
from src.utils.reproducibility import set_seed


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Execute one training epoch.

    Args:
        model: PyTorch neural network.
        dataloader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimization algorithm.
        device: Hardware device (MPS/CPU).

    Returns:
        Tuple of (average_loss, accuracy_percentage).
    """
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # 1. Zero gradients from previous iteration
        optimizer.zero_grad()

        # 2. Forward pass: compute raw class logits
        logits = model(images)

        # 3. Calculate Cross-Entropy Loss
        loss = criterion(logits, labels)

        # 4. Backward pass: compute loss gradients w.r.t parameters
        loss.backward()

        # 5. Optimizer step: update model weights
        optimizer.step()

        # 6. Accumulate metrics
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(logits, dim=1)
        correct_predictions += (preds == labels).sum().item()
        total_samples += images.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = (correct_predictions / total_samples) * 100.0
    return epoch_loss, epoch_acc


def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Execute one validation epoch without gradient calculations.

    Args:
        model: PyTorch neural network.
        dataloader: Validation DataLoader.
        criterion: Loss function.
        device: Hardware device.

    Returns:
        Tuple of (average_loss, accuracy_percentage).
    """
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(logits, dim=1)
            correct_predictions += (preds == labels).sum().item()
            total_samples += images.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = (correct_predictions / total_samples) * 100.0
    return epoch_loss, epoch_acc


def run_one_batch_sanity_check(
    config_path: Path | str | None = None,
) -> Dict[str, any]:
    """Perform a single mini-batch forward, loss, backward, and optimization sanity check.

    Why this matters for your viva:
    A one-batch sanity check catches runtime incompatibilities (e.g. MPS tensor mismatches,
    dtype errors, gradient breaks) in 2 seconds before committing to a full multi-epoch run.

    Returns:
        Dictionary of verification diagnostics.
    """
    if config_path is None:
        config_path = project_root / "configs" / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = config.get("seed", 42)
    set_seed(seed)

    device = get_device(preferred=config.get("device", {}).get("preferred", "mps"))
    device_info = get_device_info()

    model = BaselineCNN(num_classes=config["model"]["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    dataloaders = create_dataloaders(
        splits_dir=project_root / "data" / "splits",
        batch_size=8,  # Small batch for quick check
        num_workers=0,
        root_dir=project_root,
    )

    images, labels = next(iter(dataloaders["train"]))
    images = images.to(device)
    labels = labels.to(device)

    # Track initial parameter value
    initial_param = model.conv_block1[0].weight.clone()

    optimizer.zero_grad()
    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()

    updated_param = model.conv_block1[0].weight
    weight_delta = torch.norm(initial_param - updated_param).item()

    _, preds = torch.max(logits, dim=1)
    acc = ((preds == labels).sum().item() / labels.size(0)) * 100.0

    return {
        "status": "success",
        "device": str(device),
        "device_name": device_info.get("device_name", "Unknown"),
        "batch_size": int(images.size(0)),
        "input_shape": list(images.shape),
        "logits_shape": list(logits.shape),
        "initial_loss": float(loss.item()),
        "batch_accuracy": float(acc),
        "weight_delta": float(weight_delta),
        "weights_updated": bool(weight_delta > 0.0),
    }


def train_model(
    config_path: Path | str | None = None,
    epochs_override: int | None = None,
) -> Dict[str, any]:
    """Execute complete multi-epoch training and validation pipeline.

    Args:
        config_path: Path to config.yaml.
        epochs_override: Optional number of epochs to run (overrides config).

    Returns:
        Training run summary dictionary.
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

    # 3. Directories
    checkpoints_dir = project_root / config["logging"]["checkpoints_dir"]
    figures_dir = project_root / config["logging"]["figures_dir"]
    reports_dir = project_root / "reports"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 4. Hyperparameters
    batch_size = config["training"].get("batch_size", 32)
    num_workers = config["training"].get("num_workers", 0)
    learning_rate = config["training"].get("learning_rate", 0.001)
    weight_decay = config["training"].get("weight_decay", 0.0001)
    epochs = epochs_override if epochs_override is not None else config["training"].get("epochs", 10)
    num_classes = config["model"].get("num_classes", 10)
    dropout_rate = config["model"].get("dropout_rate", 0.3)

    print("=" * 75)
    print("STAGE 3: BASELINE CNN TRAINING PIPELINE")
    print("=" * 75)
    print(f"Device: {device} ({device_info.get('device_name', 'Host')})")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | Learning Rate: {learning_rate}")
    print(f"Regularization: Weight Decay={weight_decay}, Dropout={dropout_rate}")
    print("-" * 75)

    # 5. DataLoaders
    splits_dir = project_root / "data" / "splits"
    dataloaders = create_dataloaders(
        splits_dir=splits_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        root_dir=project_root,
    )
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    # 6. Model, Loss, Optimizer
    model = BaselineCNN(num_classes=num_classes, dropout_rate=dropout_rate).to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"Model: BaselineCNN | Trainable Parameters: {trainable_params:,}")
    print("-" * 75)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 7. Training Loop
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "epoch_duration": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0
    best_checkpoint_path = checkpoints_dir / "best_baseline_cnn.pt"

    start_total_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)

        duration = time.time() - epoch_start

        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["train_acc"].append(round(train_acc, 2))
        history["val_acc"].append(round(val_acc, 2))
        history["epoch_duration"].append(round(duration, 2))

        # Checkpoint saving on improvement
        is_best = val_loss < best_val_loss
        best_marker = ""
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            best_marker = " ★ (Best Model Saved)"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "config": config,
            }, best_checkpoint_path)

        print(
            f"Epoch [{epoch:>2}/{epochs:>2}] | "
            f"Train Loss: {train_loss:.4f} - Acc: {train_acc:>5.2f}% | "
            f"Val Loss: {val_loss:.4f} - Acc: {val_acc:>5.2f}% | "
            f"Time: {duration:>4.1f}s{best_marker}"
        )

    total_training_time = time.time() - start_total_time
    print("-" * 75)
    print(f"Training Complete in {total_training_time/60:.2f} minutes.")
    print(f"Best Validation Loss: {best_val_loss:.4f} at Epoch {best_epoch}")
    print(f"Best Checkpoint: {best_checkpoint_path}")

    # 8. Save History & Plots
    history_path = reports_dir / "baseline_cnn_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    plot_paths = plot_learning_curves(history, figures_dir, prefix="baseline_cnn")
    print(f"Saved Learning Curves to:")
    for k, p in plot_paths.items():
        print(f"  - {k.title()}: {p}")
    print("=" * 75)

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
    train_model()
