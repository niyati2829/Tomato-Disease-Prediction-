"""Long Short-Term Memory (LSTM) Network Implemented from Scratch in NumPy (REQ-09).

Mathematical Formulations (Hochreiter & Schmidhuber, 1997):
Let z_t = [x_t, h_{t-1}] be the concatenated input and previous hidden state:
    1. Forget gate:          f_t = sigmoid(W_f @ z_t + b_f)
    2. Input gate:           i_t = sigmoid(W_i @ z_t + b_i)
    3. Candidate cell state: g_t = tanh(W_c @ z_t + b_c)
    4. Cell state update:    C_t = f_t * C_{t-1} + i_t * g_t
    5. Output gate:          o_t = sigmoid(W_o @ z_t + b_o)
    6. Hidden state:         h_t = o_t * tanh(C_t)

Why LSTM mitigates vanishing gradients:
The cell state C_t acts as an additive "constant error carousel" (CEC). During
Backpropagation Through Time (BPTT), the gradient dC_t / dC_{t-1} equals the
forget gate activation f_t rather than repeated nonlinear matrix multiplications
by W_hh^T, preserving gradient signal across long horizons when f_t ~ 1.0.

Author: Crop Disease Detection Project Team
"""

from typing import Dict, Optional, Tuple

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid activation."""
    z_clip = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z_clip))


class LSTMCellNumPy:
    """Single-layer LSTM network implemented in pure NumPy."""

    def __init__(self, input_size: int, hidden_size: int, seed: int = 42) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        concat_size = input_size + hidden_size

        rng = np.random.RandomState(seed)
        scale = 1.0 / np.sqrt(hidden_size)

        # Gate weights over concatenated [x_t, h_{t-1}] -> shape (H, D + H)
        self.W_i = rng.uniform(-scale, scale, size=(hidden_size, concat_size))
        self.b_i = np.zeros(hidden_size, dtype=float)

        self.W_f = rng.uniform(-scale, scale, size=(hidden_size, concat_size))
        self.b_f = np.zeros(hidden_size, dtype=float)

        self.W_c = rng.uniform(-scale, scale, size=(hidden_size, concat_size))
        self.b_c = np.zeros(hidden_size, dtype=float)

        self.W_o = rng.uniform(-scale, scale, size=(hidden_size, concat_size))
        self.b_o = np.zeros(hidden_size, dtype=float)

    def load_from_pytorch_weights(
        self,
        weight_ih: np.ndarray,
        weight_hh: np.ndarray,
        bias_ih: np.ndarray,
        bias_hh: np.ndarray,
    ) -> None:
        """Load weights directly from a single-layer PyTorch nn.LSTM for exact parity testing.

        PyTorch packs gates in order: (input, forget, cell, output), each of size H.
        """
        H = self.hidden_size
        w_ii, w_if, w_ic, w_io = np.split(weight_ih, 4, axis=0)
        w_hi, w_hf, w_hc, w_ho = np.split(weight_hh, 4, axis=0)
        b_i = bias_ih + bias_hh
        b_ii, b_if, b_ic, b_io = np.split(b_i, 4, axis=0)

        self.W_i = np.concatenate([w_ii, w_hi], axis=1)
        self.W_f = np.concatenate([w_if, w_hf], axis=1)
        self.W_c = np.concatenate([w_ic, w_hc], axis=1)
        self.W_o = np.concatenate([w_io, w_ho], axis=1)

        self.b_i = b_ii
        self.b_f = b_if
        self.b_c = b_ic
        self.b_o = b_io

    def forward_sequence(
        self,
        X: np.ndarray,
        h_0: Optional[np.ndarray] = None,
        c_0: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Process batch of sequences and return all states and internal gate trajectories.

        Args:
            X: Input tensor of shape (Batch, Seq_Len, Input_Size).

        Returns:
            Dictionary containing hidden_states, cell_states, h_last, c_last, and gate histories.
        """
        if X.ndim != 3:
            raise ValueError(f"Expected 3D input (B, T, D), got {X.shape}")

        B, T, D = X.shape
        if D != self.input_size:
            raise ValueError(f"Expected input_size={self.input_size}, got {D}")

        H = self.hidden_size
        h_t = np.zeros((B, H), dtype=float) if h_0 is None else h_0.copy()
        c_t = np.zeros((B, H), dtype=float) if c_0 is None else c_0.copy()

        h_hist = np.zeros((B, T, H), dtype=float)
        c_hist = np.zeros((B, T, H), dtype=float)
        f_hist = np.zeros((B, T, H), dtype=float)
        i_hist = np.zeros((B, T, H), dtype=float)
        o_hist = np.zeros((B, T, H), dtype=float)

        for t in range(T):
            x_t = X[:, t, :]
            # Concatenate [x_t, h_{t-1}] -> shape (B, D + H)
            z_t = np.concatenate([x_t, h_t], axis=1)

            f_t = _sigmoid(z_t @ self.W_f.T + self.b_f)
            i_t = _sigmoid(z_t @ self.W_i.T + self.b_i)
            g_t = np.tanh(z_t @ self.W_c.T + self.b_c)
            o_t = _sigmoid(z_t @ self.W_o.T + self.b_o)

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * np.tanh(c_t)

            h_hist[:, t, :] = h_t
            c_hist[:, t, :] = c_t
            f_hist[:, t, :] = f_t
            i_hist[:, t, :] = i_t
            o_hist[:, t, :] = o_t

        return {
            "hidden_states": h_hist,
            "cell_states": c_hist,
            "h_last": h_t,
            "c_last": c_t,
            "forget_gates": f_hist,
            "input_gates": i_hist,
            "output_gates": o_hist,
        }

