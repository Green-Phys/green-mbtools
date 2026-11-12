import numpy as np
import pytest

from green_mbtools.mint.integral_symmetry_transform import (
    CholeskyGauge,
    auxiliary_transform,
    reconstruct_factor,
    stored_basis_operation,
    validate_overlap_covariance,
)


def complex_random(rng, shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


@pytest.mark.parametrize(
    "conjugate,transpose", [(False, False), (True, False), (False, True), (True, True)]
)
def test_three_legs_and_all_source_rows_for_slice(conjugate, transpose):
    rng = np.random.default_rng(9107)
    c = np.linalg.cholesky((m := complex_random(rng, (5, 5))) @ m.conj().T + np.eye(5))
    a = np.linalg.qr(complex_random(rng, (5, 5)))[0]
    ck = c.conj() if conjugate else c
    ct = np.linalg.cholesky(a @ ck @ ck.conj().T @ a.conj().T)
    d = auxiliary_transform(
        CholeskyGauge(c, "source", "hf"),
        CholeskyGauge(ct, "target", "hf"),
        a,
        conjugate=conjugate,
    )
    raw = complex_random(rng, (5, 3, 3))
    v = np.linalg.solve(c, raw.reshape(5, -1)).reshape(raw.shape)
    left, right = complex_random(rng, (3, 3)), complex_random(rng, (3, 3))
    mapped_raw = raw.conj() if conjugate else raw
    if transpose:
        mapped_raw = mapped_raw.swapaxes(1, 2)
    mapped_raw = np.stack(
        [left @ b @ right.conj().T for b in np.einsum("ab,bij->aij", a, mapped_raw)]
    )
    expected = np.linalg.solve(ct, mapped_raw.reshape(5, -1)).reshape(raw.shape)
    actual = reconstruct_factor(
        v, d, left, right, conjugate=conjugate, transpose=transpose
    )
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    sliced = reconstruct_factor(
        v, d, left, right, conjugate=conjugate, transpose=transpose, output_slice=(2, 2)
    )
    np.testing.assert_allclose(sliced, actual[2:4], atol=1e-12, rtol=1e-12)
    assert not np.shares_memory(actual, v)


def test_inverse_and_nontrivial_little_group_action():
    rng = np.random.default_rng(91)
    v = complex_random(rng, (4, 3, 3))
    d = np.diag([1, -1, 1, -1])
    u = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    mapped = reconstruct_factor(v, d, u, u)
    assert np.linalg.norm(mapped - v) > 1
    np.testing.assert_allclose(
        reconstruct_factor(mapped, d, u, u), v, atol=1e-13, rtol=1e-13
    )


def test_cross_pair_eris_detect_independent_wrong_gauge():
    rng = np.random.default_rng(810)
    v1, v2 = complex_random(rng, (5, 2, 2)), complex_random(rng, (5, 2, 2))
    rotation = np.linalg.qr(complex_random(rng, (5, 5)))[0]
    wrong_v2 = np.einsum("ab,bij->aij", rotation, v2)
    gram = lambda x, y: np.einsum("aij,akl->ijkl", x.conj(), y)
    np.testing.assert_allclose(gram(v2, v2), gram(wrong_v2, wrong_v2), atol=1e-12)
    assert np.linalg.norm(gram(v1, v2) - gram(v1, wrong_v2)) > 1
    r1 = reconstruct_factor(v1, rotation, np.eye(2), np.eye(2))
    r2 = reconstruct_factor(v2, rotation, np.eye(2), np.eye(2))
    np.testing.assert_allclose(gram(r1, r2), gram(v1, v2), atol=1e-12)


@pytest.mark.parametrize("conjugate", [False, True])
def test_stored_basis_covariance_does_not_require_euclidean_unitarity(conjugate):
    rng = np.random.default_rng(914)
    u = np.linalg.qr(complex_random(rng, (3, 3)))[0]
    xs, xt = np.diag([1, 2, 3]).astype(complex), np.diag([0.5, 2, 4]).astype(complex)
    stored_u = stored_basis_operation(u, xt, np.linalg.inv(xs), conjugate=conjugate)
    assert not np.allclose(stored_u.conj().T @ stored_u, np.eye(3))
    s = (b := complex_random(rng, (3, 3))) @ b.conj().T
    st = u @ (s.conj() if conjugate else s) @ u.conj().T
    validate_overlap_covariance(
        xs @ s @ xs.conj().T, xt @ st @ xt.conj().T, stored_u, conjugate=conjugate
    )
    raw = complex_random(rng, (4, 3, 3))
    source = reconstruct_factor(raw, np.eye(4), xs, xs)
    actual = reconstruct_factor(
        source, np.eye(4), stored_u, stored_u, conjugate=conjugate
    )
    expected = reconstruct_factor(raw, np.eye(4), xt @ u, xt @ u, conjugate=conjugate)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_fail_closed_for_wrong_gauge_kernel_rank_and_slice():
    source = CholeskyGauge(np.eye(3), "source", "hf")
    with pytest.raises(ValueError, match="kernels"):
        auxiliary_transform(
            source, CholeskyGauge(np.eye(3), "target", "corrected"), np.eye(3)
        )
    with pytest.raises(ValueError, match="covariance"):
        auxiliary_transform(
            source, CholeskyGauge(2 * np.eye(3), "target", "hf"), np.eye(3)
        )
    with pytest.raises(ValueError, match="full-rank"):
        CholeskyGauge(np.ones((3, 2)), "source", "hf")
    with pytest.raises(ValueError, match="positive"):
        CholeskyGauge(np.diag([1, -1]), "source", "hf")
    with pytest.raises(ValueError, match="slice"):
        reconstruct_factor(
            np.ones((3, 2, 2)), np.eye(3), np.eye(2), np.eye(2), output_slice=(2, 2)
        )
