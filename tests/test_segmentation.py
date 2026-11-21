"""Unit Tests for U-Net Semantic Segmentation (REQ-08).

Verifies:
1. U-Net architecture forward pass output dimensions and class channels.
2. DiceLoss and CombinedLoss computation and backpropagation.
3. Exact analytical accuracy of IoU, Dice Score, and Pixel Accuracy functions.
4. Nearest-neighbor categorical mask resizing preserves strictly discrete integer IDs.

Author: Crop Disease Detection Project Team
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.segmentation.dataset import LeafSegmentationDataset
from src.segmentation.losses_metrics import (
    CombinedLoss,
    DiceLoss,
    calculate_dice_score,
    calculate_iou,
    calculate_pixel_accuracy,
)
from src.segmentation.unet import UNet


def test_unet_forward_pass_dimensions():
    """Verify U-Net produces pixel logits matching spatial dimensions of input."""
    model = UNet(in_channels=3, num_classes=3, init_features=16)
    x = torch.randn(2, 3, 128, 128)
    out = model(x)

    assert out.shape == (2, 3, 128, 128), f"Expected (2, 3, 128, 128), got {out.shape}"
    assert torch.all(torch.isfinite(out))


def test_unet_odd_input_dimensions_skip_connection():
    """Verify U-Net handles non-power-of-two input dimensions via internal padding."""
    model = UNet(in_channels=3, num_classes=2, init_features=16)
    x = torch.randn(1, 3, 113, 129)
    out = model(x)
    assert out.shape == (1, 2, 113, 129)


def test_segmentation_losses_and_gradients():
    """Verify DiceLoss and CombinedLoss produce positive finite scalars and valid gradients."""
    logits = torch.randn(2, 3, 64, 64, requires_grad=True)
    targets = torch.randint(0, 3, (2, 64, 64))

    dice_loss = DiceLoss()
    comb_loss = CombinedLoss(weight_ce=0.5)

    l_d = dice_loss(logits, targets)
    l_c = comb_loss(logits, targets)

    assert 0.0 <= l_d.item() <= 1.0
    assert l_c.item() > 0.0

    l_c.backward()
    assert logits.grad is not None
    assert torch.all(torch.isfinite(logits.grad))


def test_iou_and_dice_analytical_exactness():
    """Verify IoU and Dice calculations against manually calculated ground truth."""
    # 2x2 grid: Class 0 and Class 1
    # Pred:   [[1, 1], [0, 0]]
    # Target: [[1, 0], [1, 0]]
    # Class 1: Intersection = 1 ((0,0)), Union = 3 ((0,0),(0,1),(1,0)) -> IoU = 1/3, Dice = 2*1 / (2+2) = 0.5
    # Class 0: Intersection = 1 ((1,1)), Union = 3 ((0,1),(1,0),(1,1)) -> IoU = 1/3, Dice = 0.5
    preds = torch.tensor([[[1, 1], [0, 0]]])
    targets = torch.tensor([[[1, 0], [1, 0]]])

    miou, c_ious = calculate_iou(preds, targets, num_classes=2, smooth=0.0)
    mdice, c_dices = calculate_dice_score(preds, targets, num_classes=2, smooth=0.0)
    acc = calculate_pixel_accuracy(preds, targets)

    assert c_ious[0] == pytest.approx(1.0 / 3.0, abs=1e-5)
    assert c_ious[1] == pytest.approx(1.0 / 3.0, abs=1e-5)
    assert miou == pytest.approx(1.0 / 3.0, abs=1e-5)

    assert c_dices[0] == pytest.approx(0.5, abs=1e-5)
    assert c_dices[1] == pytest.approx(0.5, abs=1e-5)
    assert mdice == pytest.approx(0.5, abs=1e-5)
    assert acc == pytest.approx(0.5, abs=1e-5)


def test_mask_nearest_neighbor_interpolation():
    """Verify resizing masks preserves strictly discrete class indices {0, 2} with zero intermediate {1}."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_p = Path(tmpdir)
        img_p = tmp_p / "test_img.png"
        mask_p = tmp_p / "test_mask.png"

        # 64x64 image and binary mask containing ONLY values 0 and 2 (NO class 1)
        img_arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        mask_arr = np.zeros((64, 64), dtype=np.uint8)
        mask_arr[16:48, 16:48] = 2

        Image.fromarray(img_arr).save(img_p)
        Image.fromarray(mask_arr).save(mask_p)

        # Resize to (97, 97) - bilinear would create interpolated values of 1 at edges!
        ds = LeafSegmentationDataset([img_p], [mask_p], image_size=(97, 97))
        img_t, mask_t = ds[0]

        assert img_t.shape == (3, 97, 97)
        assert mask_t.shape == (97, 97)
        unique_vals = set(torch.unique(mask_t).tolist())
        assert unique_vals == {0, 2}, f"Interpolation corrupted categorical mask! Found: {unique_vals}"

