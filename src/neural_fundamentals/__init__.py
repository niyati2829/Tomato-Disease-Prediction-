"""Neural Network Fundamentals Package.

Demonstrates core foundational concepts:
- McCulloch-Pitts biological threshold neuron
- Continuous Sigmoid neuron
- Rosenblatt's Perceptron learning algorithm
- Activation function comparative landscape
- Vectorized Multilayer Perceptron from scratch with manual backpropagation

Author: Crop Disease Detection Project Team
"""

from src.neural_fundamentals.mcculloch_pitts import (
    McCullochPittsNeuron,
    create_mp_logic_gate,
    demonstrate_mp_gates,
)
from src.neural_fundamentals.sigmoid_neuron import (
    SigmoidNeuron,
    sigmoid,
    sigmoid_derivative,
)
from src.neural_fundamentals.perceptron import (
    Perceptron,
    plot_perceptron_decision_boundaries,
)
from src.neural_fundamentals.activations import (
    step,
    sigmoid_act,
    tanh_act,
    relu_act,
    leaky_relu_act,
    get_activation_specifications,
    plot_activation_comparison,
)
from src.neural_fundamentals.mlp_from_scratch import (
    MultilayerPerceptron,
    plot_mlp_training_results,
)

__all__ = [
    "McCullochPittsNeuron",
    "create_mp_logic_gate",
    "demonstrate_mp_gates",
    "SigmoidNeuron",
    "sigmoid",
    "sigmoid_derivative",
    "Perceptron",
    "plot_perceptron_decision_boundaries",
    "step",
    "sigmoid_act",
    "tanh_act",
    "relu_act",
    "leaky_relu_act",
    "get_activation_specifications",
    "plot_activation_comparison",
    "MultilayerPerceptron",
    "plot_mlp_training_results",
]

