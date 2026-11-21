"""McCulloch-Pitts (M-P) Neuron Implementation.

The historic 1943 McCulloch-Pitts model was the first computational abstraction of a biological neuron:
    Inputs: x_i in {0, 1} (excitatory or inhibitory)
    Weights: w_i = +1 (excitatory) or -inf (inhibitory veto)
    Activation: y = 1 if sum_{i=1}^n w_i * x_i >= theta, else 0

Demonstrates:
- Logical AND (theta = n)
- Logical OR (theta = 1)
- Logical NOT (inhibitory input veto)
- Logical NAND, NOR
- Mathematical proof of XOR failure without hidden layers.

Author: Crop Disease Detection Project Team
"""

from typing import Any, Dict, List, Tuple
import numpy as np


class McCullochPittsNeuron:
    """McCulloch-Pitts binary threshold neuron."""

    def __init__(self, weights: List[int], threshold: int, inhibitory_mask: List[bool] | None = None) -> None:
        """Initialize M-P neuron with discrete integer weights and firing threshold.

        Args:
            weights: Integer weights for each input.
            threshold: Firing threshold theta.
            inhibitory_mask: Boolean list where True indicates absolute inhibitory input.
        """
        self.weights = np.array(weights, dtype=int)
        self.threshold = threshold
        self.inhibitory_mask = np.array(inhibitory_mask) if inhibitory_mask is not None else np.zeros(len(weights), dtype=bool)

    def activate(self, inputs: List[int] | np.ndarray) -> int:
        """Compute firing state (0 or 1).

        If any active input is inhibitory, the neuron is inhibited (output = 0).
        Otherwise, it fires if weighted sum >= threshold.
        """
        x = np.array(inputs, dtype=int)
        # Check inhibitory veto
        if np.any(x[self.inhibitory_mask] == 1):
            return 0

        # Excitatory summation
        weighted_sum = np.sum(self.weights * x)
        return 1 if weighted_sum >= self.threshold else 0

    def predict(self, inputs: List[int] | np.ndarray) -> int:
        """Alias for activate."""
        return self.activate(inputs)


def create_mp_logic_gate(gate_type: str) -> McCullochPittsNeuron:
    """Factory function for fundamental Boolean logic gates."""
    gate = gate_type.upper()
    if gate == "AND":
        # 2 inputs, both must fire: w1=1, w2=1, theta=2
        return McCullochPittsNeuron(weights=[1, 1], threshold=2)
    elif gate == "OR":
        # At least one must fire: w1=1, w2=1, theta=1
        return McCullochPittsNeuron(weights=[1, 1], threshold=1)
    elif gate == "NOT":
        # Single input: 0 -> 1, 1 -> 0
        # Implemented with fixed excitatory threshold 0 and inhibitory input
        return McCullochPittsNeuron(weights=[0], threshold=0, inhibitory_mask=[True])
    elif gate == "NOR":
        # Neither can fire: both inputs inhibitory, threshold 0
        return McCullochPittsNeuron(weights=[0, 0], threshold=0, inhibitory_mask=[True, True])
    elif gate == "NAND":
        # Returns 0 only when both inputs are 1
        # In M-P: w1=-1, w2=-1, theta=-1 (or inhibitory network)
        return McCullochPittsNeuron(weights=[-1, -1], threshold=-1)
    else:
        raise ValueError(f"Unsupported logic gate: {gate_type}")


def test_truth_table(neuron: McCullochPittsNeuron, truth_table: List[Tuple[List[int], int]]) -> bool:
    """Verify neuron output against canonical truth table."""
    for inputs, expected in truth_table:
        output = neuron.activate(inputs)
        if output != expected:
            return False
    return True


def demonstrate_mp_gates() -> Dict[str, Dict[str, Any]]:
    """Test and document canonical Boolean gates."""
    tables = {
        "AND": [([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1)],
        "OR": [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 1)],
        "NOT": [([0], 1), ([1], 0)],
        "NOR": [([0, 0], 1), ([0, 1], 0), ([1, 0], 0), ([1, 1], 0)],
        "NAND": [([0, 0], 1), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)],
    }

    report = {}
    for gate_name, table in tables.items():
        neuron = create_mp_logic_gate(gate_name)
        passed = test_truth_table(neuron, table)
        outputs = [neuron.activate(inp) for inp, _ in table]
        report[gate_name] = {
            "passed": passed,
            "weights": neuron.weights.tolist(),
            "threshold": neuron.threshold,
            "truth_table": [(inp, exp, out) for (inp, exp), out in zip(table, outputs)],
        }
    return report


def build_and_gate(n_inputs: int = 2) -> McCullochPittsNeuron:
    """Convenience builder for n-input AND gate."""
    return McCullochPittsNeuron(weights=[1] * n_inputs, threshold=n_inputs)


def build_or_gate(n_inputs: int = 2) -> McCullochPittsNeuron:
    """Convenience builder for n-input OR gate."""
    return McCullochPittsNeuron(weights=[1] * n_inputs, threshold=1)


def build_not_gate() -> McCullochPittsNeuron:
    """Convenience builder for NOT gate."""
    return create_mp_logic_gate("NOT")


def build_nand_gate(n_inputs: int = 2) -> McCullochPittsNeuron:
    """Convenience builder for NAND gate."""
    return create_mp_logic_gate("NAND")


def build_nor_gate(n_inputs: int = 2) -> McCullochPittsNeuron:
    """Convenience builder for NOR gate."""
    return create_mp_logic_gate("NOR")


def prove_xor_impossibility() -> Dict[str, Any]:
    """Formal mathematical proof that XOR is unsolvable by a single linear threshold neuron."""
    return {
        "is_solvable_single_neuron": False,
        "mathematical_contradiction": (
            "System of inequalities for XOR demonstrates an inherent mathematical contradiction: "
            "(1) x=(0,0) -> y=0: 0 < theta. "
            "(2) x=(0,1) -> y=1: w2 >= theta. "
            "(3) x=(1,0) -> y=1: w1 >= theta. "
            "(4) x=(1,1) -> y=0: w1 + w2 < theta. "
            "Summing (2) and (3) yields w1 + w2 >= 2*theta. "
            "Combining with (4) implies 2*theta <= w1 + w2 < theta, which simplifies to theta < 0. "
            "However, this directly contradicts inequality (1) stating theta > 0. "
            "Thus, no real weights (w1, w2) and threshold theta can satisfy XOR simultaneously."
        ),
    }
