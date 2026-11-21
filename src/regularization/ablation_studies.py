"""Ablation Studies: Controlled Experimental Benchmarking (REQ-05).

Executes controlled, single-factor ablation experiments across:
1. Baseline (Unregularized)
2. L1 Regularization (Feature Sparsity)
3. L2 Regularization (Weight Decay)
4. Dropout Regularization (Co-adaptation Prevention)
5. Synthetic Data Augmentation (Input Perturbation / Jitter on training set only)
6. Initialization Strategies: He vs Xavier vs Random Normal

Tracks:
- Train Loss & Accuracy
- Validation Loss & Accuracy
- Generalization Gap (Train Acc - Val Acc)
- Weight Norms and Parameter Sparsity
- Convergence Speed

Author: Crop Disease Detection Project Team
"""

import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

from src.regularization.regularizers import (
    RegularizedMLP,
    generate_overfitting_dataset,
    train_regularized_model,
)


def apply_synthetic_augmentation(X_train: np.ndarray, seed: int = 42) -> np.ndarray:
    """Apply synthetic feature jitter / noise to training inputs without touching original dataset."""
    rng = np.random.RandomState(seed)
    # Add mild Gaussian noise + random feature masking (simulating image crop / cutout)
    noise = rng.randn(*X_train.shape) * 0.08
    mask = (rng.rand(*X_train.shape) > 0.05).astype(float)
    return (X_train + noise) * mask


def run_ablation_benchmark(seed: int = 42) -> Dict[str, Any]:
    """Execute rigorous single-factor ablation study on identical train/validation splits."""
    X_train_raw, y_train, X_val, y_val = generate_overfitting_dataset(
        n_train=60,
        n_val=140,
        n_features=30,
        noise=0.25,
        seed=seed,
    )
    n_in = X_train_raw.shape[1]
    n_hidden = 40
    n_out = 2
    epochs = 90
    lr = 0.12

    ablations = [
        # 1. Baseline
        {
            "name": "Exp-1: Baseline (Unregularized)",
            "l1": 0.0,
            "l2": 0.0,
            "dropout": 0.0,
            "augmented": False,
            "init": "he",
            "desc": "Standard 2-layer MLP without regularization or augmentation",
        },
        # 2. L1 Regularization
        {
            "name": "Exp-2: L1 Regularization",
            "l1": 0.015,
            "l2": 0.0,
            "dropout": 0.0,
            "augmented": False,
            "init": "he",
            "desc": "Lasso penalty driving uninformative weights to exact zero",
        },
        # 3. L2 Regularization
        {
            "name": "Exp-3: L2 Regularization",
            "l1": 0.0,
            "l2": 0.02,
            "dropout": 0.0,
            "augmented": False,
            "init": "he",
            "desc": "Weight decay shrinking large parameter magnitudes",
        },
        # 4. Dropout
        {
            "name": "Exp-4: Inverted Dropout (p=0.35)",
            "l1": 0.0,
            "l2": 0.0,
            "dropout": 0.35,
            "augmented": False,
            "init": "he",
            "desc": "Random unit deactivation to prevent co-adaptation",
        },
        # 5. Data Augmentation
        {
            "name": "Exp-5: Training Data Augmentation",
            "l1": 0.0,
            "l2": 0.0,
            "dropout": 0.0,
            "augmented": True,
            "init": "he",
            "desc": "Input feature jitter applied to training partition only",
        },
        # 6. Initialization: Xavier on ReLU
        {
            "name": "Exp-6: Xavier Init (on ReLU)",
            "l1": 0.0,
            "l2": 0.0,
            "dropout": 0.0,
            "augmented": False,
            "init": "xavier",
            "desc": "Glorot initialization without ReLU factor of 2",
        },
        # 7. Initialization: Random Normal (unscaled)
        {
            "name": "Exp-7: Random Normal Init",
            "l1": 0.0,
            "l2": 0.0,
            "dropout": 0.0,
            "augmented": False,
            "init": "random",
            "desc": "Standard normal initialization N(0, 1) prone to saturation",
        },
    ]

    benchmark_records: List[Dict[str, Any]] = []

    for cfg in ablations:
        # Prepare training data
        if cfg["augmented"]:
            X_tr = apply_synthetic_augmentation(X_train_raw, seed=seed)
        else:
            X_tr = X_train_raw

        # Build model with specific initialization
        model = RegularizedMLP(
            n_in=n_in,
            n_hidden=n_hidden,
            n_out=n_out,
            l1_reg=cfg["l1"],
            l2_reg=cfg["l2"],
            dropout_p=cfg["dropout"],
            seed=seed,
        )

        # Override weights if testing alternate initializations
        rng = np.random.RandomState(seed)
        if cfg["init"] == "xavier":
            model.W1 = rng.randn(n_in, n_hidden) * np.sqrt(2.0 / (n_in + n_hidden))
            model.W2 = rng.randn(n_hidden, n_out) * np.sqrt(2.0 / (n_hidden + n_out))
        elif cfg["init"] == "random":
            model.W1 = rng.randn(n_in, n_hidden) * 0.8
            model.W2 = rng.randn(n_hidden, n_out) * 0.8

        train_res = train_regularized_model(model, X_tr, y_train, X_val, y_val, epochs=epochs, lr=lr)

        gen_gap = (train_res["final_train_acc"] - train_res["final_val_acc"]) * 100.0

        record = {
            "experiment": cfg["name"],
            "description": cfg["desc"],
            "train_loss": round(train_res["final_train_loss"], 4),
            "val_loss": round(train_res["final_val_loss"], 4),
            "train_accuracy_pct": round(train_res["final_train_acc"] * 100.0, 2),
            "val_accuracy_pct": round(train_res["final_val_acc"] * 100.0, 2),
            "generalization_gap_pct": round(gen_gap, 2),
            "w_norm_l2": round(train_res["w_norm_l2"], 3),
            "sparsity_pct": round(train_res["sparsity_percentage"], 2),
        }
        benchmark_records.append(record)

    output_dir = Path("reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "phase_B_ablation_results.json"
    with open(out_file, "w") as f:
        json.dump(benchmark_records, f, indent=2)

    return {"records": benchmark_records, "json_path": str(out_file)}

