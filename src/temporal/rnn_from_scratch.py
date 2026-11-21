"""Vanilla Recurrent Neural Network (RNN) Implemented from Scratch in NumPy (REQ-09).

Mathematical Formulation:
At discrete time step t = 1, ..., T:
    h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)
    y_t = W_hy @ h_t + b_y

Educational Note:
This NumPy implementation demonstrates recurrent hidden-state propagation and
matches PyTorch's `torch.nn.RNN` forward pass numerically to floating-point precision.
It is intended for algorithmic transparency, not production deployment.

Author: Crop Disease Detection Project Team
"""

from typing import Optional, Tuple

import numpy as np


class VanillaRNNNumPy:
    """Single-layer Vanilla RNN implemented in pure NumPy."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int = 1,
        seed: int = 42,
    ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        rng = np.random.RandomState(seed)
        scale = 1.0 / np.sqrt(hidden_size)
        self.W_xh = rng.uniform(-scale, scale, size=(hidden_size, input_size))
        self.W_hh = rng.uniform(-scale, scale, size=(hidden_size, hidden_size))
        self.b_h = np.zeros(hidden_size, dtype=float)

        self.W_hy = rng.uniform(-scale, scale, size=(output_size, hidden_size))
        self.b_y = np.zeros(output_size, dtype=float)

    def forward_sequence(
        self,
        X: np.ndarray,
        h_0: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process a batch of input sequences over time.

        Args:
            X: Input sequences of shape (Batch, Seq_Len, Input_Size).
            h_0: Optional initial hidden state of shape (Batch, Hidden_Size).

        Returns:
            Tuple of:
                - hidden_states: All hidden states of shape (Batch, Seq_Len, Hidden_Size)
                - h_last: Final hidden state of shape (Batch, Hidden_Size)
                - outputs: Linear projections y_t of shape (Batch, Seq_Len, Output_Size)
        """
        if X.ndim != 3:
            raise ValueError(f"Expected 3D sequence input (B, T, D), got {X.shape}")

        B, T, D = X.shape
        if D != self.input_size:
            raise ValueError(f"Expected input_size={self.input_size}, got {D}")

        h_t = np.zeros((B, self.hidden_size), dtype=float) if h_0 is None else h_0.copy()

        hidden_states = np.zeros((B, T, self.hidden_size), dtype=float)
        outputs = np.zeros((B, T, self.output_size), dtype=float)

        for t in range(T):
            x_t = X[:, t, :]  # (B, D)
            # h_t = tanh(x_t @ W_xh^T + h_{t-1} @ W_hh^T + b_h)
            pre_act = x_t @ self.W_xh.T + h_t @ self.W_hh.T + self.b_h
            h_t = np.tanh(pre_act)
            y_t = h_t @ self.W_hy.T + self.b_y

            hidden_states[:, t, :] = h_t
            outputs[:, t, :] = y_t

        return hidden_states, h_t, outputs

