"""Tensor Operations & Broadcasting Demonstration Module.

Demonstrates core mathematical tensor operations using NumPy:
1. Tensor creation, inspection (shape, ndim, size, strides, dtype).
2. Reshaping, contiguous flattening, and dimension permutation.
3. Inner products: Dot products, element-wise multiplication, and matrix multiplication.
4. Broadcasting mechanics: rules, matching trailing dimensions, and failure modes.
5. Graphical visualization of 1D -> 2D and 2D -> 3D broadcasting.

Author: Crop Disease Detection Project Team
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def inspect_tensor(arr: np.ndarray, name: str = "Tensor") -> Dict[str, Any]:
    """Inspect tensor structural metadata."""
    return {
        "name": name,
        "shape": arr.shape,
        "ndim": arr.ndim,
        "size": arr.size,
        "dtype": str(arr.dtype),
        "strides": arr.strides,
        "is_contiguous": arr.flags["C_CONTIGUOUS"],
        "is_c_contiguous": arr.flags["C_CONTIGUOUS"],
    }


def reshape_and_flatten(arr: np.ndarray, target_shape: Tuple[int, ...]) -> Dict[str, np.ndarray]:
    """Demonstrate contiguous flattening and dimension reshaping."""
    return {
        "flattened_1d": arr.ravel(),
        "reshaped_nd": arr.reshape(target_shape),
    }


def compute_inner_products(u: np.ndarray, v: np.ndarray, M1: np.ndarray, M2: np.ndarray) -> Dict[str, Any]:
    """Compute vector dot product, matrix multiplication, and elementwise Hadamard product."""
    return {
        "vector_dot_product": float(np.dot(u, v)),
        "matrix_multiplication": M1 @ M2,
        "elementwise_product": M1 * M2,
    }


def compute_frobenius_norm(M: np.ndarray) -> float:
    """Compute Frobenius matrix norm: sqrt(sum(M_ij^2))."""
    return float(np.linalg.norm(M, "fro"))


def check_broadcasting_compatibility(shape_a: Tuple[int, ...], shape_b: Tuple[int, ...]) -> Tuple[bool, Tuple[int, ...] | None, str]:
    """Check broadcasting compatibility between two tensor shapes."""
    success, res = test_broadcasting(shape_a, shape_b)
    if success:
        return True, res, ""
    else:
        return False, None, str(res)


def demonstrate_tensor_operations() -> Dict[str, Any]:
    """Execute and return core tensor operations."""
    # 1. Creation of scalars, vectors, matrices, and 3D tensors
    scalar = np.array(42.0)
    vector = np.array([1.0, 2.0, 3.0, 4.0])
    matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    tensor_3d = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

    # 2. Reshaping & flattening
    reshaped_2d = tensor_3d.reshape(6, 4)
    flattened = tensor_3d.ravel()

    # 3. Inner products and multiplications
    vec_a = np.array([1.0, 2.0, 3.0])
    vec_b = np.array([4.0, 5.0, 6.0])
    dot_prod = float(np.dot(vec_a, vec_b))  # 1*4 + 2*5 + 3*6 = 32

    mat_a = np.array([[1.0, 2.0], [3.0, 4.0]])
    mat_b = np.array([[5.0, 6.0], [7.0, 8.0]])
    elem_mult = mat_a * mat_b  # Hadamard product
    matmul_res = mat_a @ mat_b  # Matrix multiplication

    # Frobenius norm
    frob_norm = float(np.linalg.norm(mat_a, "fro"))

    return {
        "scalar_info": inspect_tensor(scalar, "Scalar (0D)"),
        "vector_info": inspect_tensor(vector, "Vector (1D)"),
        "matrix_info": inspect_tensor(matrix, "Matrix (2D)"),
        "tensor_3d_info": inspect_tensor(tensor_3d, "Volume (3D)"),
        "reshaped_shape": reshaped_2d.shape,
        "flattened_shape": flattened.shape,
        "dot_product": dot_prod,
        "elementwise_multiplication": elem_mult.tolist(),
        "matrix_multiplication": matmul_res.tolist(),
        "frobenius_norm": frob_norm,
    }


def test_broadcasting(shape_a: Tuple[int, ...], shape_b: Tuple[int, ...]) -> Tuple[bool, Tuple[int, ...] | str]:
    """Test whether two shapes are broadcast-compatible under NumPy rules."""
    a = np.zeros(shape_a)
    b = np.zeros(shape_b)
    try:
        c = a + b
        return True, c.shape
    except ValueError as e:
        return False, str(e)


def demonstrate_broadcasting_rules() -> List[Dict[str, Any]]:
    """Demonstrate compatible and incompatible broadcasting cases."""
    cases = [
        ((3, 4), (4,), "Trailing dimension matches (4 == 4), prepends 1"),
        ((3, 1), (1, 4), "Both expand along singleton axes to (3, 4)"),
        ((2, 3, 4), (3, 1), "Broadcasting across batch and inner dims -> (2, 3, 4)"),
        ((3, 4), (3,), "Incompatible: trailing dims 4 != 3 and neither is 1"),
        ((2, 3), (3, 2), "Incompatible: transpose shapes cannot broadcast"),
    ]

    results = []
    for s_a, s_b, description in cases:
        compatible, out_shape = test_broadcasting(s_a, s_b)
        results.append({
            "shape_a": s_a,
            "shape_b": s_b,
            "compatible": compatible,
            "output_or_error": out_shape,
            "rule_description": description,
        })
    return results


def plot_broadcasting_examples(output_path: Path | str) -> Path:
    """Generate visual explanation of tensor broadcasting."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=150)

    # Example: Matrix (3, 1) + Row Vector (1, 4) -> (3, 4)
    A = np.array([[10], [20], [30]])
    B = np.array([[1, 2, 3, 4]])
    C = A + B

    # Plot A
    im1 = axes[0].imshow(np.repeat(A, 4, axis=1), cmap="Blues", alpha=0.6)
    axes[0].set_title(f"Array A: shape (3, 1)\nBroadcast along cols", fontsize=10, fontweight="bold")
    for i in range(3):
        for j in range(4):
            axes[0].text(j, i, f"{A[i, 0]}", ha="center", va="center", fontsize=9, fontweight="bold")
    axes[0].set_xticks(range(4))
    axes[0].set_yticks(range(3))

    # Plot B
    im2 = axes[1].imshow(np.repeat(B, 3, axis=0), cmap="Oranges", alpha=0.6)
    axes[1].set_title(f"Array B: shape (1, 4)\nBroadcast along rows", fontsize=10, fontweight="bold")
    for i in range(3):
        for j in range(4):
            axes[1].text(j, i, f"{B[0, j]}", ha="center", va="center", fontsize=9, fontweight="bold")
    axes[1].set_xticks(range(4))
    axes[1].set_yticks(range(3))

    # Plot Result C
    im3 = axes[2].imshow(C, cmap="Greens", alpha=0.7)
    axes[2].set_title(f"Result A + B: shape (3, 4)\nElement-wise Sum", fontsize=10, fontweight="bold")
    for i in range(3):
        for j in range(4):
            axes[2].text(j, i, f"{C[i, j]}", ha="center", va="center", fontsize=9, fontweight="bold")
    axes[2].set_xticks(range(4))
    axes[2].set_yticks(range(3))

    plt.suptitle("NumPy Tensor Broadcasting Mechanics: (3, 1) + (1, 4) → (3, 4)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file
