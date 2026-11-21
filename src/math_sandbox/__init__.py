"""Mathematical Sandbox Package.

Demonstrates mathematical foundations of deep learning:
- Tensor operations & broadcasting rules
- SVD image compression
- Numerical vs. analytical gradient checking

Author: Crop Disease Detection Project Team
"""

from src.math_sandbox.tensor_ops import (
    demonstrate_tensor_operations,
    demonstrate_broadcasting_rules,
    plot_broadcasting_examples,
)
from src.math_sandbox.svd_compression import (
    compress_matrix_svd,
    compress_image_svd,
    plot_svd_compression_comparison,
)
from src.math_sandbox.gradient_checker import (
    forward_finite_difference,
    central_finite_difference,
    check_gradient_scalar,
    check_gradient_vector,
)

__all__ = [
    "demonstrate_tensor_operations",
    "demonstrate_broadcasting_rules",
    "plot_broadcasting_examples",
    "compress_matrix_svd",
    "compress_image_svd",
    "plot_svd_compression_comparison",
    "forward_finite_difference",
    "central_finite_difference",
    "check_gradient_scalar",
    "check_gradient_vector",
]

