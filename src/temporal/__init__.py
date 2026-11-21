"""Temporal Disease Progression & Sequence Modeling Package (REQ-09).

Provides:
- Pure NumPy Vanilla RNN and LSTM forward-pass implementations (from scratch).
- Synthetic longitudinal plant disease progression benchmark generator with sequence-level isolation.
- PyTorch VanillaRNNRegressor and LSTMRegressor models with internal state/gate inspection.
- Training, evaluation, and state-dynamics visualization tools.
"""

from src.temporal.lstm_from_scratch import LSTMCellNumPy
from src.temporal.rnn_from_scratch import VanillaRNNNumPy
from src.temporal.sequence_generator import generate_synthetic_progression_sequences
from src.temporal.temporal_model import LSTMRegressor, VanillaRNNRegressor

__all__ = [
    "VanillaRNNNumPy",
    "LSTMCellNumPy",
    "generate_synthetic_progression_sequences",
    "VanillaRNNRegressor",
    "LSTMRegressor",
]

