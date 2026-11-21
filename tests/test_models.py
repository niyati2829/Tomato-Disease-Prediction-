"""Unit Tests for Baseline CNN Architecture.

Verifies:
1. Model accepts input shape [32, 3, 224, 224] and produces output shape [32, 10].
2. Model parameters are trainable and parameter count matches expectation (~620K).
3. Forward pass yields finite values (no NaNs or Infs).
4. Model successfully transfers and runs on the detected hardware device (MPS or CPU).
5. A single optimization step properly updates model weights (gradient sanity check).

Author: Crop Disease Detection Project Team
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.cnn import BaselineCNN, count_parameters
from src.models.transfer_learning import (
    count_trainable_parameters,
    freeze_backbone,
    get_fine_tuning_param_groups,
    get_resnet18,
    unfreeze_layer4_and_fc,
)
from src.utils.device import get_device


def test_baseline_cnn_forward_shape():
    """Verify that input tensor of shape (32, 3, 224, 224) yields output of shape (32, 10)."""
    model = BaselineCNN(num_classes=10)
    model.eval()

    dummy_input = torch.randn(32, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (32, 10), f"Expected shape (32, 10), got {output.shape}"
    assert output.dtype == torch.float32, f"Expected float32 logits, got {output.dtype}"


def test_baseline_cnn_parameters():
    """Verify that model contains trainable parameters and non-zero parameter count."""
    model = BaselineCNN(num_classes=10)
    total_params, trainable_params = count_parameters(model)

    assert total_params > 0, "Model has 0 parameters!"
    assert total_params == trainable_params, "All baseline CNN parameters should be trainable by default"
    # Expected: 620,362 parameters
    assert 600_000 < total_params < 700_000, f"Expected ~620k parameters, got {total_params}"


def test_baseline_cnn_finite_outputs():
    """Verify that model output does not contain NaN or Inf values."""
    model = BaselineCNN(num_classes=10)
    model.eval()

    dummy_input = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)

    assert torch.isfinite(output).all(), "Model output contains non-finite values (NaN/Inf)!"


def test_baseline_cnn_device_compatibility():
    """Verify that model and tensors run seamlessly on the selected hardware device (MPS/CPU)."""
    device = get_device()
    model = BaselineCNN(num_classes=10).to(device)
    model.eval()

    dummy_input = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.device.type == device.type, f"Expected output on {device.type}, got {output.device.type}"
    assert output.shape == (2, 10)


def test_baseline_cnn_gradient_update():
    """Verify that backward pass computes valid gradients and updates weights on an optimization step."""
    model = BaselineCNN(num_classes=10)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Capture initial weights of first conv layer
    initial_weights = model.conv_block1[0].weight.clone()

    # Synthetic single forward + backward step
    dummy_input = torch.randn(4, 3, 224, 224)
    dummy_target = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    optimizer.zero_grad()
    logits = model(dummy_input)
    loss = criterion(logits, dummy_target)
    loss.backward()
    optimizer.step()

    # Verify weights changed
    updated_weights = model.conv_block1[0].weight
    assert not torch.equal(initial_weights, updated_weights), "Model weights did not change after gradient step!"


def test_resnet18_forward_shape_and_logits():
    """Verify that adapted ResNet18 produces shape (32, 10) raw logits for (32, 3, 224, 224)."""
    model = get_resnet18(num_classes=10, pretrained=True, freeze_backbone_weights=True)
    model.eval()

    dummy_input = torch.randn(32, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (32, 10), f"Expected shape (32, 10), got {output.shape}"
    assert output.dtype == torch.float32, f"Expected float32 logits, got {output.dtype}"
    assert torch.isfinite(output).all(), "ResNet18 output contains NaN or Inf values!"


def test_resnet18_num_classes():
    """Verify that ResNet18 final classification layer outputs exactly 10 classes."""
    model = get_resnet18(num_classes=10, pretrained=True)
    assert isinstance(model.fc, nn.Linear)
    assert model.fc.in_features == 512
    assert model.fc.out_features == 10


def test_resnet18_phase_a_feature_extraction_freezing():
    """Verify Phase A properly freezes all backbone weights and keeps only fc trainable."""
    model = get_resnet18(num_classes=10, pretrained=True, freeze_backbone_weights=True)

    total_params, trainable_params = count_trainable_parameters(model)
    # fc layer: 512 * 10 + 10 = 5,130 parameters
    assert trainable_params == 5_130, f"Expected 5,130 trainable params in Phase A, got {trainable_params}"
    assert total_params > 11_000_000

    for name, param in model.named_parameters():
        if "fc" in name:
            assert param.requires_grad is True, f"{name} should have requires_grad=True"
        else:
            assert param.requires_grad is False, f"{name} should have requires_grad=False"


def test_resnet18_phase_b_finetuning_unfreezing():
    """Verify Phase B unfreezes layer4 and fc while keeping conv1 through layer3 frozen."""
    model = get_resnet18(num_classes=10, pretrained=True, freeze_backbone_weights=True)
    unfreeze_layer4_and_fc(model)

    total_params, trainable_params = count_trainable_parameters(model)
    # layer4 (~8.4M) + fc (5,130) -> ~8.4M trainable
    assert trainable_params > 8_000_000, f"Expected >8M trainable params in Phase B, got {trainable_params}"

    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            assert param.requires_grad is True, f"{name} should have requires_grad=True"
        else:
            assert param.requires_grad is False, f"{name} should have requires_grad=False"


def test_resnet18_frozen_parameters_receive_no_gradients():
    """Verify that frozen backbone parameters receive no gradient updates during backward pass."""
    model = get_resnet18(num_classes=10, pretrained=True, freeze_backbone_weights=True)
    model.train()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    dummy_input = torch.randn(4, 3, 224, 224)
    dummy_target = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    optimizer.zero_grad()
    logits = model(dummy_input)
    loss = criterion(logits, dummy_target)
    loss.backward()

    # Verify frozen layers have no gradients
    assert model.conv1.weight.grad is None, "Frozen conv1 should have None grad!"
    assert model.layer1[0].conv1.weight.grad is None, "Frozen layer1 should have None grad!"
    assert model.layer2[0].conv1.weight.grad is None, "Frozen layer2 should have None grad!"
    assert model.layer3[0].conv1.weight.grad is None, "Frozen layer3 should have None grad!"

    # Verify trainable fc has valid gradients
    assert model.fc.weight.grad is not None, "Trainable fc should have non-None grad!"
    assert torch.isfinite(model.fc.weight.grad).all()


def test_resnet18_device_compatibility():
    """Verify that ResNet18 runs forward pass seamlessly on selected hardware device."""
    device = get_device()
    model = get_resnet18(num_classes=10, pretrained=True).to(device)
    model.eval()

    dummy_input = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.device.type == device.type
    assert output.shape == (2, 10)


def test_resnet18_phase_a_batchnorm_running_stats_frozen():
    """Verify that in Phase A, calling model.train() leaves all BatchNorm layers in eval mode

    and running_mean / running_var do not drift during forward pass.
    """
    model = get_resnet18(num_classes=10, pretrained=True, freeze_backbone_weights=True)
    model.train()  # Explicitly call .train()

    # Capture initial buffers
    init_bn1_mean = model.bn1.running_mean.clone()
    init_bn1_var = model.bn1.running_var.clone()
    init_l1_mean = model.layer1[0].bn1.running_mean.clone()
    init_l2_mean = model.layer2[0].bn1.running_mean.clone()
    init_l3_mean = model.layer3[0].bn1.running_mean.clone()
    init_l4_mean = model.layer4[0].bn1.running_mean.clone()

    # All BatchNorm modules must have .training == False
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            assert module.training is False, "BatchNorm in Phase A should be in eval mode!"

    # Forward pass with random tensor
    dummy_input = torch.randn(8, 3, 224, 224)
    _ = model(dummy_input)

    # Running statistics must remain completely unchanged
    assert torch.equal(init_bn1_mean, model.bn1.running_mean), "bn1 running_mean modified!"
    assert torch.equal(init_bn1_var, model.bn1.running_var), "bn1 running_var modified!"
    assert torch.equal(init_l1_mean, model.layer1[0].bn1.running_mean), "layer1 bn running_mean modified!"
    assert torch.equal(init_l2_mean, model.layer2[0].bn1.running_mean), "layer2 bn running_mean modified!"
    assert torch.equal(init_l3_mean, model.layer3[0].bn1.running_mean), "layer3 bn running_mean modified!"
    assert torch.equal(init_l4_mean, model.layer4[0].bn1.running_mean), "layer4 bn running_mean modified!"


def test_resnet18_phase_b_batchnorm_selective_behavior():
    """Verify that in Phase B, early BatchNorm layers remain in eval mode (frozen stats)

    while layer4 BatchNorm layers run in train mode (adapting stats).
    """
    model = get_resnet18(num_classes=10, pretrained=True, freeze_backbone_weights=True)
    unfreeze_layer4_and_fc(model)
    model.train()  # Explicitly call .train()

    # Verify mode flags
    assert model.bn1.training is False, "Early bn1 should be in eval mode in Phase B"
    assert model.layer1[0].bn1.training is False, "layer1 bn should be in eval mode in Phase B"
    assert model.layer2[0].bn1.training is False, "layer2 bn should be in eval mode in Phase B"
    assert model.layer3[0].bn1.training is False, "layer3 bn should be in eval mode in Phase B"
    assert model.layer4[0].bn1.training is True, "layer4 bn should be in train mode in Phase B"

    # Capture initial buffers
    init_bn1_mean = model.bn1.running_mean.clone()
    init_l3_mean = model.layer3[0].bn1.running_mean.clone()
    init_l4_mean = model.layer4[0].bn1.running_mean.clone()

    # Forward pass with random tensor
    dummy_input = torch.randn(8, 3, 224, 224)
    _ = model(dummy_input)

    # Early layers must be identical
    assert torch.equal(init_bn1_mean, model.bn1.running_mean), "Early bn1 running_mean modified in Phase B!"
    assert torch.equal(init_l3_mean, model.layer3[0].bn1.running_mean), "layer3 running_mean modified in Phase B!"

    # layer4 BatchNorm should have updated running statistics
    assert not torch.equal(init_l4_mean, model.layer4[0].bn1.running_mean), "layer4 running_mean should update in Phase B!"


