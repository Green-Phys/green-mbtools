"""Tests for the per-k orthogonalization helpers in ``green_mbtools.mint.ortho_utils``.

These cover the small building blocks used by ``common_utils.orthogonalize``:
    - ``lowdin_per_k``  : symmetric S^{-1/2} orthogonalization
    - ``mo_per_k``      : canonical-MO basis from supplied C(k)
    - ``natural_per_k`` : natural-orbital basis from a density matrix
All helpers return ``(X, X_inv)`` in the ``X Z X†`` convention used by
``common_utils.transform``.
"""

import numpy as np
import pytest
import scipy.linalg as LA

from green_mbtools.mint import ortho_utils


_TOL = 1e-10


def _hermitian_pd(rng, n, shift=None):
    """Build a random Hermitian positive-definite matrix of size n."""
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    M = A.conj().T @ A
    if shift is None:
        shift = n
    M = M + shift * np.eye(n)
    return 0.5 * (M + M.conj().T)


def _hermitian(rng, n):
    """Build a random Hermitian matrix of size n."""
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    return 0.5 * (A + A.conj().T)


@pytest.fixture
def rng():
    return np.random.default_rng(20260609)


@pytest.mark.parametrize("n", [1, 4, 7])
def test_lowdin_per_k_orthogonalizes_S(rng, n):
    S = _hermitian_pd(rng, n)
    X, X_inv = ortho_utils.lowdin_per_k(S)

    # X X_inv = I  (left-inverse)
    assert np.allclose(X @ X_inv, np.eye(n), atol=_TOL)
    # X S X† = I  (Löwdin orthogonalizes S)
    assert np.allclose(X @ S @ X.conj().T, np.eye(n), atol=_TOL)
    # X_inv = S^{1/2} reconstructs S:  X_inv X_inv† = S
    assert np.allclose(X_inv @ X_inv.conj().T, S, atol=_TOL)


def test_lowdin_per_k_drops_small_eigenvalues(rng):
    # Build S with one near-zero eigenvalue; expect rectangular output.
    n = 5
    U, _ = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    evals = np.array([1e-12, 0.5, 1.0, 2.0, 3.0])
    S = U @ np.diag(evals) @ U.conj().T
    S = 0.5 * (S + S.conj().T)

    X, X_inv = ortho_utils.lowdin_per_k(S, tol=1e-9)
    # one mode dropped → X has shape (n-1, n)
    assert X.shape == (n - 1, n)
    assert X_inv.shape == (n, n - 1)
    assert np.allclose(X @ X_inv, np.eye(n - 1), atol=_TOL)


def test_mo_per_k_diagonalizes_fock(rng):
    n = 6
    S = _hermitian_pd(rng, n)
    F = _hermitian(rng, n)
    eps, C = LA.eigh(F, S)  # C† S C = I by construction

    X, X_inv = ortho_utils.mo_per_k(S, C)

    # left-inverse
    assert np.allclose(X @ X_inv, np.eye(n), atol=_TOL)
    # X F X† is diagonal with mo_energy
    F_MO = X @ F @ X.conj().T
    assert np.allclose(F_MO, np.diag(np.diag(F_MO)), atol=_TOL)
    assert np.allclose(np.sort(np.diag(F_MO).real), np.sort(eps), atol=_TOL)
    # X S X† = I
    S_MO = X @ S @ X.conj().T
    assert np.allclose(S_MO, np.eye(n), atol=_TOL)


def test_natural_per_k_diagonalizes_dm(rng):
    n = 6
    S = _hermitian_pd(rng, n)
    # construct a positive-semi-definite "density" matrix
    B = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    dm = B.conj().T @ B
    dm = 0.5 * (dm + dm.conj().T)

    X, X_inv = ortho_utils.natural_per_k(S, dm)

    # left-inverse and S-orthogonalisation
    assert np.allclose(X @ X_inv, np.eye(n), atol=_TOL)
    assert np.allclose(X @ S @ X.conj().T, np.eye(n), atol=_TOL)
    # X dm X†  is diagonal (NO occupations on diagonal in the operator
    # convention; mirrors mo_per_k's shape contract X = C_NO†, X_inv = S·C_NO)
    D = X @ dm @ X.conj().T
    assert np.allclose(D, np.diag(np.diag(D)), atol=_TOL)


def test_helpers_return_complex128(rng):
    n = 3
    S = _hermitian_pd(rng, n)
    F = _hermitian(rng, n)
    _, C = LA.eigh(F, S)
    dm = _hermitian_pd(rng, n, shift=0.0)

    for X, X_inv in (
        ortho_utils.lowdin_per_k(S),
        ortho_utils.mo_per_k(S, C),
        ortho_utils.natural_per_k(S, dm),
    ):
        assert X.dtype == np.complex128
        assert X_inv.dtype == np.complex128
