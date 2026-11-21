"""Unit tests for Disease Classification 10-class probability distribution."""

from pathlib import Path
import pytest
from PIL import Image
import torch

from app.app import (
    CHECKPOINT_PATH,
    CLASS_MAPPING_PATH,
    CALIBRATION_TEMPERATURE,
    load_class_mappings,
    load_production_model,
    predict_leaf,
)


@pytest.fixture(scope="module")
def model_and_mapping():
    model, device, _ = load_production_model(CHECKPOINT_PATH)
    _, idx_to_class, class_names_sorted = load_class_mappings(CLASS_MAPPING_PATH)
    return model, device, idx_to_class, class_names_sorted


def test_predict_leaf_default_top3(model_and_mapping):
    """Ensure backward-compatible default top_k=3."""
    model, device, idx_to_class, _ = model_and_mapping
    dummy = Image.new("RGB", (224, 224), color=(60, 140, 50))
    preds = predict_leaf(dummy, model, device, idx_to_class)
    assert len(preds) == 3
    assert preds[0]["rank"] == 1
    assert preds[1]["rank"] == 2
    assert preds[2]["rank"] == 3
    assert preds[0]["probability"] >= preds[1]["probability"] >= preds[2]["probability"]


def test_predict_leaf_all_10_classes(model_and_mapping):
    """Verify inference output across all 10 classes with calibration."""
    model, device, idx_to_class, class_names_sorted = model_and_mapping
    dummy = Image.new("RGB", (224, 224), color=(60, 140, 50))
    
    # Request all 10 classes
    preds = predict_leaf(dummy, model, device, idx_to_class, top_k=len(idx_to_class))
    
    # Check count
    assert len(preds) == 10, f"Expected 10 classes, got {len(preds)}"
    
    # Check ranks 1 through 10
    ranks = [p["rank"] for p in preds]
    assert ranks == list(range(1, 11))
    
    # Check descending order
    probs = [p["probability"] for p in preds]
    for i in range(len(probs) - 1):
        assert probs[i] >= probs[i + 1], f"Probabilities not descending at index {i}"
        
    # Check mapping consistency (no invented/hardcoded names)
    returned_raw_names = {p["raw_class"] for p in preds}
    expected_raw_names = set(class_names_sorted)
    assert returned_raw_names == expected_raw_names
    
    # Check probabilities sum to approximately 100%
    total_pct = sum(p["percentage"] for p in preds)
    assert pytest.approx(100.0, abs=0.01) == total_pct
    
    # Check formatting with two decimal places
    for p in preds:
        formatted = f"{p['percentage']:.2f}%"
        assert "%" in formatted
        # parse back
        num = float(formatted.replace("%", ""))
        assert 0.0 <= num <= 100.0
