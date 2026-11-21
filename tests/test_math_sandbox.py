"""Unit Tests for Mathematical Sandbox (REQ-01).

Tests:
1. Tensor creation, inspection, reshaping, and matrix operations.
2. Broadcasting compatibility rules and error throwing on incompatible shapes.
3. SVD decomposition, rank-k approximation properties, and energy retention.
4. Numerical gradient checker: forward vs central difference and convergence order.

Author: Crop Disease Detection Project Team
"""

import numpy as np
import pytest

from src.math_sandbox.gradient_checker import (
    check_gradient_matrix,
    check_gradient_scalar,
    check_gradient_vector,
)
from src.math_sandbox.svd_compression import compress_matrix_svd, compress_rgb_svd
from src.math_sandbox.tensor_ops import (
    check_broadcasting_compatibility,
    compute_frobenius_norm,
    compute_inner_products,
    inspect_tensor,
    reshape_and_flatten,
)


# ---------------------------------------------------------------------------
# Tensor Operations Tests
# ---------------------------------------------------------------------------


def test_tensor_inspection():
    """Verify tensor metadata inspection correctly reports shape, size, dtype, and strides."""
    arr = np.zeros((3, 4, 5), dtype=np.float32)
    meta = inspect_tensor(arr, name="TestTensor3D")

    assert meta["name"] == "TestTensor3D"
    assert meta["shape"] == (3, 4, 5)
    assert meta["ndim"] == 3
    assert meta["size"] == 60
    assert "float32" in meta["dtype"]
    assert meta["is_c_contiguous"] is True


def test_reshape_and_flatten():
    """Verify contiguous reshaping and flattening preserves total element count and data."""
    arr = np.arange(24).reshape((2, 3, 4))
    res = reshape_and_flatten(arr, target_shape=(6, 4))

    assert res["flattened_1d"].shape == (24,)
    assert res["reshaped_nd"].shape == (6, 4)
    np.testing.assert_array_equal(res["flattened_1d"], np.arange(24))


def test_inner_products():
    """Verify dot product, matrix multiplication, and element-wise Hadamard products."""
    u = np.array([1.0, 2.0, 3.0])
    v = np.array([4.0, 5.0, 6.0])
    M1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    M2 = np.array([[2.0, 0.0], [1.0, 3.0]])

    results = compute_inner_products(u, v, M1, M2)

    # 1*4 + 2*5 + 3*6 = 32
    assert results["vector_dot_product"] == pytest.approx(32.0)
    # Matrix product [[4, 6], [10, 12]]
    expected_matmul = np.array([[4.0, 6.0], [10.0, 12.0]])
    np.testing.assert_allclose(results["matrix_multiplication"], expected_matmul)
    # Element-wise [[2, 0], [3, 12]]
    expected_hadamard = np.array([[2.0, 0.0], [3.0, 12.0]])
    np.testing.assert_allclose(results["elementwise_product"], expected_hadamard)


def test_frobenius_norm():
    """Verify Frobenius norm computation matches sqrt of sum of squared elements."""
    M = np.array([[3.0, 4.0], [0.0, 0.0]])
    # sqrt(3^2 + 4^2) = 5.0
    assert compute_frobenius_norm(M) == pytest.approx(5.0)


def test_broadcasting_rules():
    """Verify NumPy broadcasting rules on valid and invalid shape pairs."""
    # Compatible: (3, 1) and (1, 4) -> (3, 4)
    compat, res_shape, _ = check_broadcasting_compatibility((3, 1), (1, 4))
    assert compat is True
    assert res_shape == (3, 4)

    # Compatible: (5, 3, 1) and (3, 4) -> (5, 3, 4)
    compat, res_shape, _ = check_broadcasting_compatibility((5, 3, 1), (3, 4))
    assert compat is True
    assert res_shape == (5, 3, 4)

    # Incompatible: (3, 2) and (3, 4) -> incompatible along dim 1
    compat, res_shape, err = check_broadcasting_compatibility((3, 2), (3, 4))
    assert compat is False
    assert res_shape is None
    assert "could not be broadcast together" in err.lower() or "mismatch" in err.lower()


# ---------------------------------------------------------------------------
# SVD Compression Tests
# ---------------------------------------------------------------------------


def test_svd_matrix_compression():
    """Verify SVD decomposition reconstruction error decreases monotonically with rank k."""
    rng = np.random.RandomState(42)
    # Construct full-rank matrix of size 40x30
    A = rng.randn(40, 30)

    ranks = [2, 5, 10, 20]
    errors = []
    energies = []

    for k in ranks:
        approx, diag = compress_matrix_svd(A, k=k)
        assert approx.shape == A.shape
        errors.append(diag["frobenius_error"])
        energies.append(diag["retained_energy_percentage"])

    # Error must decrease monotonically
    for i in range(len(errors) - 1):
        assert errors[i] > errors[i + 1]

    # Energy must increase monotonically
    for i in range(len(energies) - 1):
        assert energies[i] < energies[i + 1]

    # Exact rank-10 matrix should have ~0 error at k=10
    A_lowrank = rng.randn(40, 10) @ rng.randn(10, 30)
    approx_10, diag_10 = compress_matrix_svd(A_lowrank, k=10)
    assert diag_10["frobenius_error"] < 1e-10
    assert diag_10["retained_energy_percentage"] == pytest.approx(100.0, abs=1e-4)


def test_svd_rgb_compression():
    """Verify SVD compression on 3-channel RGB image produces valid bounded outputs."""
    img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    res = compress_rgb_svd(img, k=10)

    assert res["reconstructed_rgb"].shape == (64, 64, 3)
    assert res["reconstructed_rgb"].dtype == np.uint8
    assert 0.0 < res["mean_energy_retention"] <= 1.0
    assert 0.0 < res["compression_ratio"] < 1.0


# ---------------------------------------------------------------------------
# Numerical Gradient Checking Tests
# ---------------------------------------------------------------------------


def test_scalar_gradient_check_order_of_accuracy():
    """Verify central finite difference achieves O(h^2) while forward difference is O(h)."""
    # f(x) = x^3 - 2x + 1, f'(x) = 3x^2 - 2
    f = lambda x: x**3 - 2.0 * x + 1.0
    df = lambda x: 3.0 * x**2 - 2.0
    x0 = 2.0

    # Test at h = 1e-5
    res = check_gradient_scalar(f, df, x0, h=1e-5)
    # Central difference error is O(h^2) ~ 1e-10
    assert res["central_diff_error"] < 1e-9
    # Forward difference error is O(h) ~ 1e-5
    assert res["forward_diff_error"] > res["central_diff_error"]
    assert res["central_diff_passed"] is True


def test_vector_gradient_check():
    """Verify gradient checking on multivariable vector function."""
    # f(x) = x1^2 + 3*x2^2 + 2*x1*x2
    f = lambda x: x[0] ** 2 + 3.0 * x[1] ** 2 + 2.0 * x[0] * x[1]
    # grad f = [2*x1 + 2*x2, 6*x2 + 2*x1]
    grad_f = lambda x: np.array([2.0 * x[0] + 2.0 * x[1], 6.0 * x[1] + 2.0 * x[0]])

    x0 = np.array([1.5, -2.0])
    res = check_gradient_vector(f, grad_f, x0, h=1e-7)

    assert res["passed"] is True
    assert res["relative_error"] < 1e-6


def test_matrix_gradient_check():
    """Verify gradient checking on matrix parameter tensor."""
    # f(W) = 0.5 * ||W||_F^2, grad_W = W
    f = lambda W: 0.5 * np.sum(W**2)
    grad_f = lambda W: W.copy()

    W0 = np.array([[1.0, -2.0], [3.0, 0.5]])
    res = check_gradient_matrix(f, grad_f, W0, h=1e-7)

    assert res["passed"] is True
    assert res["relative_error"] < 1e-6
