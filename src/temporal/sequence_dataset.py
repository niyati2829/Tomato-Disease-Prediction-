"""PyTorch Dataset Wrapper for Temporal Progression Sequences (REQ-09).

Author: Crop Disease Detection Project Team
"""

from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class DiseaseSequenceDataset(Dataset):
    """Dataset returning (sequence_tensor, target_severity, pattern_id, plant_id)."""

    def __init__(self, split_dict: Dict[str, np.ndarray]) -> None:
        self.X = torch.tensor(split_dict["X"], dtype=torch.float32)
        self.y = torch.tensor(split_dict["y"], dtype=torch.float32).unsqueeze(-1)  # (N, 1)
        self.patterns = torch.tensor(split_dict["patterns"], dtype=torch.long)
        self.plant_ids = torch.tensor(split_dict["plant_ids"], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx], self.patterns[idx], self.plant_ids[idx]

