"""Classical Learning Paradigms & Energy-Based Neural Networks (REQ-06).

Implements from scratch:
1. Hebbian Learning & Oja's Modified Learning Rule (Principal Component extraction)
2. Competitive Learning (Winner-Takes-All) & Kohonen Self-Organizing Maps (SOM)
3. Error-Correction Learning (Delta Rule / Least Mean Squares / Adaline)
4. Discrete Hopfield Networks (Associative Memory & Lyapunov Energy Minimization)
5. Boltzmann Machines & Restricted Boltzmann Machines (Probabilistic Contrastive Divergence)

Author: Crop Disease Detection Project Team
"""

from src.learning_paradigms.boltzmann import (
    RestrictedBoltzmannMachine,
    demonstrate_boltzmann_sampling,
)
from src.learning_paradigms.competitive import (
    CompetitiveLearningWTA,
    KohonenSOM,
    demonstrate_competitive_and_som,
)
from src.learning_paradigms.error_correction import (
    DeltaRuleLearner,
    demonstrate_error_correction,
)
from src.learning_paradigms.hebbian import (
    BasicHebbianLearner,
    OjaLearner,
    compare_hebbian_and_oja,
)
from src.learning_paradigms.hopfield import (
    HopfieldNetwork,
    demonstrate_hopfield_associative_memory,
)

__all__ = [
    "BasicHebbianLearner",
    "OjaLearner",
    "compare_hebbian_and_oja",
    "CompetitiveLearningWTA",
    "KohonenSOM",
    "demonstrate_competitive_and_som",
    "DeltaRuleLearner",
    "demonstrate_error_correction",
    "HopfieldNetwork",
    "demonstrate_hopfield_associative_memory",
    "RestrictedBoltzmannMachine",
    "demonstrate_boltzmann_sampling",
]

