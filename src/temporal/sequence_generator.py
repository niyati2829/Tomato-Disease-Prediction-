"""Synthetic Longitudinal Plant Disease Progression Benchmark Generator (REQ-09).

ACADEMIC INTEGRITY STATEMENT:
PlantVillage contains static leaf photographs without timestamps or repeated plant observations.
This module generates a purely synthetic mathematical simulation of longitudinal disease
progression for educational evaluation of RNNs and LSTMs. It does NOT represent validated
biological epidemiology or predict real-world agricultural outcomes.

Simulated Patterns:
0. Healthy / Stable:           s_t = clip(s_0 + eps_t, 0, 1)
1. Linear Progression:         s_t = clip(s_0 + r * t + eps_t, 0, 1)
2. Logistic Outbreak (S-Curve): s_t = clip(1 / (1 + exp(-k*(t - t_mid))) + eps_t, 0, 1)
3. Treatment Recovery Decay:   Rise followed by exponential decay after intervention step t_treat

Features per time step t (D = 3):
  - Feature 0: Observed lesion severity index s_t in [0.0, 1.0]
  - Feature 1: First-order symptom rate of change ds_t = s_t - s_{t-1}
  - Feature 2: Simulated relative leaf wetness / humidity stress h_t in [0.0, 1.0]

Leakage Prevention:
Splits are performed strictly by `plant_id` (whole simulated sequence isolation), ensuring
zero observations from the same simulated plant appear in multiple splits.

Author: Crop Disease Detection Project Team
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


PATTERN_NAMES = {
    0: "Healthy / Stable",
    1: "Linear Spread",
    2: "Logistic Outbreak",
    3: "Treatment Recovery",
}


def generate_synthetic_progression_sequences(
    n_plants: int = 600,
    seq_len: int = 10,
    noise_std: float = 0.025,
    seed: int = 42,
    save_dir: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Generate reproducible synthetic disease progression sequences split by plant ID.

    Task: Given observed multivariate sequence X of shape (seq_len, 3) for t=0..(seq_len-1),
    predict the disease severity s_{seq_len} at the next time step t = seq_len.

    Args:
        n_plants: Total number of distinct simulated plant trajectories.
        seq_len: Number of historical input time steps per sequence (default: 10).
        noise_std: Standard deviation of Gaussian observation noise.
        seed: Reproducible random seed.
        save_dir: Optional directory to save metadata & split summary.

    Returns:
        Dictionary with keys 'train', 'val', 'test', and 'metadata'.
    """
    rng = np.random.RandomState(seed)

    total_steps = seq_len + 1  # seq_len inputs + 1 next-step target
    X_all = np.zeros((n_plants, seq_len, 3), dtype=np.float32)
    y_next_all = np.zeros(n_plants, dtype=np.float32)
    pattern_all = np.zeros(n_plants, dtype=np.int64)
    plant_ids = np.arange(n_plants, dtype=np.int64)

    for p_id in range(n_plants):
        pattern_type = p_id % 4
        pattern_all[p_id] = pattern_type

        t_arr = np.arange(total_steps, dtype=float)
        eps = rng.normal(0.0, noise_std, size=total_steps)

        # Base humidity / environmental driver correlated with outbreak rate
        base_hum = rng.uniform(0.35, 0.90)
        hum_series = np.clip(base_hum + rng.normal(0.0, 0.05, size=total_steps), 0.1, 1.0)

        if pattern_type == 0:
            # Healthy / Stable near 0.02-0.08
            s0 = rng.uniform(0.01, 0.06)
            hum_series = np.clip(0.30 + rng.normal(0.0, 0.05, size=total_steps), 0.1, 0.6)
            s_clean = np.full(total_steps, s0)
        elif pattern_type == 1:
            # Linear Progression: s_t = s_0 + r * t
            s0 = rng.uniform(0.05, 0.15)
            r = rng.uniform(0.03, 0.07) * (0.7 + 0.5 * base_hum)
            s_clean = s0 + r * t_arr
        elif pattern_type == 2:
            # Logistic Outbreak S-Curve
            k = rng.uniform(0.45, 0.85)
            t_mid = rng.uniform(3.5, 6.5)
            s_clean = 0.05 + 0.90 / (1.0 + np.exp(-k * (t_arr - t_mid)))
        else:
            # Treatment Recovery: spreads until t=4, then decays exponentially after fungicide
            s0 = rng.uniform(0.10, 0.25)
            peak_step = rng.randint(4, 6)
            rate_up = rng.uniform(0.06, 0.10)
            rate_down = rng.uniform(0.20, 0.35)
            s_clean = np.zeros(total_steps)
            for idx_t in range(total_steps):
                if idx_t <= peak_step:
                    s_clean[idx_t] = s0 + rate_up * idx_t
                else:
                    peak_val = s0 + rate_up * peak_step
                    s_clean[idx_t] = 0.05 + (peak_val - 0.05) * np.exp(-rate_down * (idx_t - peak_step))

        s_noisy = np.clip(s_clean + eps, 0.0, 1.0)

        # Compute first-order finite difference ds_t = s_t - s_{t-1} (ds_0 = 0)
        ds = np.zeros(total_steps)
        ds[1:] = s_noisy[1:] - s_noisy[:-1]

        X_all[p_id, :, 0] = s_noisy[:seq_len]
        X_all[p_id, :, 1] = ds[:seq_len]
        X_all[p_id, :, 2] = hum_series[:seq_len]
        y_next_all[p_id] = float(s_noisy[seq_len])

    # Strictly permute by plant_id to partition 70% train / 15% val / 15% test
    perm = rng.permutation(n_plants)
    n_train = int(0.70 * n_plants)
    n_val = int(0.15 * n_plants)

    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    splits = {
        "train": {
            "X": X_all[train_idx],
            "y": y_next_all[train_idx],
            "patterns": pattern_all[train_idx],
            "plant_ids": plant_ids[train_idx],
        },
        "val": {
            "X": X_all[val_idx],
            "y": y_next_all[val_idx],
            "patterns": pattern_all[val_idx],
            "plant_ids": plant_ids[val_idx],
        },
        "test": {
            "X": X_all[test_idx],
            "y": y_next_all[test_idx],
            "patterns": pattern_all[test_idx],
            "plant_ids": plant_ids[test_idx],
        },
        "metadata": {
            "is_synthetic": True,
            "n_plants": n_plants,
            "seq_len": seq_len,
            "n_features": 3,
            "noise_std": noise_std,
            "seed": seed,
            "split_sizes": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
            "feature_names": ["severity_s_t", "rate_of_change_ds_t", "humidity_stress_h_t"],
            "pattern_names": PATTERN_NAMES,
        },
    }

    if save_dir is not None:
        out_p = Path(save_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        with open(out_p / "synthetic_benchmark_metadata.json", "w", encoding="utf-8") as f:
            json.dump(splits["metadata"], f, indent=2)

    return splits
