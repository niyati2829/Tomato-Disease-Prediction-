"""Temperature scaling for post-hoc deep neural network probability calibration.

Reference:
    Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017).
    On calibration of modern neural networks. ICML.
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class ModelWithTemperature(nn.Module):
    """Decorator / wrapper module that applies temperature scaling to logits.

    The model output logits are scaled by 1 / T before softmax:
        p_i = exp(z_i / T) / sum_j exp(z_j / T)

    Attributes:
        model: Base PyTorch neural network.
        temperature: nn.Parameter containing scalar temperature T.
    """

    def __init__(self, model: nn.Module, initial_temperature: float = 1.5):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * float(initial_temperature))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with temperature-scaled logits."""
        logits = self.model(x)
        return self.scale_logits(logits)

    def scale_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale arbitrary logits by current temperature."""
        temperature = self.temperature.clamp(min=1e-3)
        return logits / temperature

    def get_temperature(self) -> float:
        """Return current scalar temperature value."""
        return float(self.temperature.detach().cpu().item())

    def set_temperature(self, t: float):
        """Manually set temperature value."""
        with torch.no_grad():
            self.temperature.copy_(torch.tensor([float(t)], dtype=self.temperature.dtype, device=self.temperature.device))


class TemperatureScaler:
    """Standalone temperature scaler optimized on validation logits and labels.

    Does not require raw images or gradients through the base model, making it
    fast, memory-efficient, and cleanly decoupled.
    """

    def __init__(self, initial_temp: float = 1.5, max_iter: int = 50, lr: float = 0.05):
        self.temperature = nn.Parameter(torch.ones(1) * float(initial_temp))
        self.max_iter = max_iter
        self.lr = lr
        self.fitted = False

    def fit(
        self,
        val_logits: Union[torch.Tensor, np.ndarray],
        val_labels: Union[torch.Tensor, np.ndarray],
        verbose: bool = False,
    ) -> float:
        """Find optimal temperature T on validation set using L-BFGS optimizer.

        Args:
            val_logits: Tensor or array of shape (N, C) containing raw validation logits.
            val_labels: Tensor or array of shape (N,) containing true integer class indices.
            verbose: If True, prints optimization progress.

        Returns:
            optimal_temperature: Scalar float T*.
        """
        if isinstance(val_logits, np.ndarray):
            logits_tensor = torch.from_numpy(val_logits).float()
        else:
            logits_tensor = val_logits.clone().detach().float()

        if isinstance(val_labels, np.ndarray):
            labels_tensor = torch.from_numpy(val_labels).long()
        else:
            labels_tensor = val_labels.clone().detach().long()

        device = torch.device("cpu")
        logits_tensor = logits_tensor.to(device)
        labels_tensor = labels_tensor.to(device)

        temp_param = nn.Parameter(self.temperature.clone().to(device))
        criterion = nn.CrossEntropyLoss()

        optimizer = optim.LBFGS([temp_param], lr=self.lr, max_iter=self.max_iter, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            temp_clamped = temp_param.clamp(min=1e-3)
            scaled_logits = logits_tensor / temp_clamped
            loss = criterion(scaled_logits, labels_tensor)
            loss.backward()
            return loss

        optimizer.step(closure)

        optimal_t = float(temp_param.clamp(min=1e-3).detach().cpu().item())
        with torch.no_grad():
            self.temperature.copy_(torch.tensor([optimal_t]))
        self.fitted = True

        if verbose:
            print(f"Optimal Temperature Scaling parameter T: {optimal_t:.4f}")

        return optimal_t

    def transform(self, logits: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Apply fitted temperature scaling to raw logits and return calibrated probabilities."""
        if not self.fitted:
            raise RuntimeError("TemperatureScaler must be fitted with .fit() before calling .transform()")

        is_numpy = isinstance(logits, np.ndarray)
        if is_numpy:
            logits_tensor = torch.from_numpy(logits).float()
        else:
            logits_tensor = logits.clone().detach().float()

        with torch.no_grad():
            scaled = logits_tensor / self.temperature.clamp(min=1e-3)
            probs = torch.softmax(scaled, dim=-1).cpu().numpy()

        return probs

    def get_temperature(self) -> float:
        return float(self.temperature.item())
