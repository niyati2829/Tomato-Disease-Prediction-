"""PyTorch Temporal Regression Architectures: Vanilla RNN vs LSTM (REQ-09).

Provides comparable sequence-to-scalar regressors predicting next-step
synthetic disease severity s_{T} from historical observations X_{0..T-1}.

Author: Crop Disease Detection Project Team
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn


class VanillaRNNRegressor(nn.Module):
    """Single-layer Vanilla RNN sequence regressor."""

    def __init__(self, input_size: int = 3, hidden_size: int = 32, num_layers: int = 1) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(
        self,
        x: torch.Tensor,
        return_states: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass.

        Args:
            x: Tensor of shape (Batch, Seq_Len, Input_Size).
            return_states: If True, also return dictionary of hidden state sequence.

        Returns:
            Prediction tensor of shape (Batch, 1), or (pred, state_dict).
        """
        h_seq, h_last = self.rnn(x)
        # h_last: (num_layers, Batch, Hidden_Size) -> take last layer
        pred = self.fc(h_last[-1])
        if return_states:
            return pred, {"hidden_states": h_seq.detach()}
        return pred


class LSTMRegressor(nn.Module):
    """Single-layer LSTM sequence regressor with internal gate trace extraction."""

    def __init__(self, input_size: int = 3, hidden_size: int = 32, num_layers: int = 1) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(
        self,
        x: torch.Tensor,
        return_states: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass.

        Args:
            x: Tensor of shape (Batch, Seq_Len, Input_Size).
            return_states: If True, extract full hidden, cell, and gate trajectories across time.
        """
        h_seq, (h_last, c_last) = self.lstm(x)
        pred = self.fc(h_last[-1])

        if not return_states:
            return pred

        # Compute exact step-by-step cell and gate states for activation analysis
        B, T, D = x.shape
        H = self.hidden_size
        device = x.device

        w_ih = self.lstm.weight_ih_l0
        w_hh = self.lstm.weight_hh_l0
        b_ih = self.lstm.bias_ih_l0
        b_hh = self.lstm.bias_hh_l0

        h_t = torch.zeros(B, H, device=device)
        c_t = torch.zeros(B, H, device=device)

        h_hist = []
        c_hist = []
        f_hist = []
        i_hist = []
        o_hist = []

        with torch.no_grad():
            for t in range(T):
                x_t = x[:, t, :]
                gates = x_t @ w_ih.T + b_ih + h_t @ w_hh.T + b_hh
                i_g, f_g, g_g, o_g = gates.chunk(4, dim=1)

                i_t = torch.sigmoid(i_g)
                f_t = torch.sigmoid(f_g)
                c_cand = torch.tanh(g_g)
                o_t = torch.sigmoid(o_g)

                c_t = f_t * c_t + i_t * c_cand
                h_t = o_t * torch.tanh(c_t)

                h_hist.append(h_t)
                c_hist.append(c_t)
                f_hist.append(f_t)
                i_hist.append(i_t)
                o_hist.append(o_t)

        states = {
            "hidden_states": torch.stack(h_hist, dim=1),
            "cell_states": torch.stack(c_hist, dim=1),
            "forget_gates": torch.stack(f_hist, dim=1),
            "input_gates": torch.stack(i_hist, dim=1),
            "output_gates": torch.stack(o_hist, dim=1),
        }
        return pred, states

