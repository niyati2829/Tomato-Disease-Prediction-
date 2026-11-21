"""Adversarial robustness evaluation using Fast Gradient Sign Method (FGSM).

Reference:
    Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015).
    Explaining and harnessing adversarial examples. ICLR.
"""

from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def unnormalize_imagenet(tensor: torch.Tensor) -> torch.Tensor:
    """Unnormalize normalized ImageNet tensor back to [0, 1] range."""
    mean = IMAGENET_MEAN.to(tensor.device)
    std = IMAGENET_STD.to(tensor.device)
    return (tensor * std) + mean


def normalize_imagenet(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize [0, 1] range tensor to ImageNet statistics."""
    mean = IMAGENET_MEAN.to(tensor.device)
    std = IMAGENET_STD.to(tensor.device)
    return (tensor - mean) / std


def fgsm_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 0.01,
    criterion: nn.Module = None,
) -> torch.Tensor:
    """Generate FGSM adversarial perturbations.

    Perturbations are computed in pixel space [0, 1] and re-normalized.

    Args:
        model: Target PyTorch model in eval mode.
        images: Batch of normalized image tensors (B, 3, H, W).
        labels: True integer labels (B,).
        epsilon: Maximum perturbation magnitude in pixel space [0, 1].
        criterion: Loss function (defaults to nn.CrossEntropyLoss()).

    Returns:
        perturbed_images: Batch of perturbed, normalized image tensors.
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    # Clone and convert to unnormalized space [0, 1]
    raw_images = unnormalize_imagenet(images).clone().detach()
    raw_images.requires_grad = True

    # Re-normalize to pass through model
    norm_images = normalize_imagenet(raw_images)
    outputs = model(norm_images)
    loss = criterion(outputs, labels)

    model.zero_grad()
    loss.backward()

    # Get sign of gradient with respect to raw pixel inputs
    data_grad = raw_images.grad.data
    sign_data_grad = data_grad.sign()

    # Add perturbation and clamp to valid [0, 1] pixel values
    perturbed_raw = raw_images + epsilon * sign_data_grad
    perturbed_raw = torch.clamp(perturbed_raw, 0.0, 1.0)

    # Re-normalize to standard model input distribution
    perturbed_norm = normalize_imagenet(perturbed_raw)
    return perturbed_norm.detach()


def evaluate_fgsm_robustness(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    epsilons: List[float] = [0.0, 0.005, 0.01, 0.02, 0.05],
    device: torch.device = torch.device("cpu"),
    max_samples: int = 500,
) -> Dict[float, Dict[str, float]]:
    """Evaluate classification accuracy under varying FGSM perturbation budgets.

    Args:
        model: Evaluated neural network.
        dataloader: Test or evaluation DataLoader.
        epsilons: List of perturbation values.
        device: Compute device.
        max_samples: Maximum samples to test for fast runtime.

    Returns:
        Dict mapping epsilon -> {'accuracy': float, 'mean_confidence': float}
    """
    model.eval()
    results = {}

    for eps in epsilons:
        correct = 0
        total = 0
        confs = []

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            if eps > 0.0:
                perturbed_images = fgsm_attack(model, images, labels, epsilon=eps)
            else:
                perturbed_images = images

            with torch.no_grad():
                outputs = model(perturbed_images)
                probs = torch.softmax(outputs, dim=-1)
                conf, preds = torch.max(probs, dim=-1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)
            confs.extend(conf.cpu().numpy().tolist())

            if total >= max_samples:
                break

        acc = correct / max(total, 1)
        mean_conf = float(np.mean(confs)) if confs else 0.0
        results[eps] = {"accuracy": acc, "mean_confidence": mean_conf, "samples": total}

    return results
