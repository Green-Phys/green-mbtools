"""Tests for the per-k orthogonalization helpers in ``green_mbtools.mint.ortho_utils``.

These cover the small building blocks used by ``common_utils.orthogonalize``:
    - ``lowdin_per_k``  : canonical Löwdin (s^{-1/2} U†) orthogonalization
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


def test_realify_strips_self_tr_noise_and_fixes_lowdin_gauge():
    # Regression for the self-TR (e.g. Γ) failure that broke canonical Löwdin on
    # Silicon. A real symmetric overlap with a *degenerate* eigenvalue plus tiny
    # imaginary Hermitian noise (~1e-13, the level S picks up at self-TR
    # k-points): on the degenerate block that noise rotates the eigenvectors by
    # O(1) complex amounts, so lowdin_per_k on the raw matrix returns a
    # complex-gauge X (max|imag X| ~ 0.64) that violates X(-k) = X(k)*.
    # _build_X_ibz calls _realify first, stripping the noise so the gauge is
    # real. This pins that seam.
    rng = np.random.default_rng(0)
    n = 4
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    S_real = Q @ np.diag([1.0, 1.0, 2.0, 3.0]) @ Q.T   # eigenvalue 1 is 2-fold
    S_real = 0.5 * (S_real + S_real.T)
    noise = np.zeros((n, n), dtype=complex)
    noise[0, 1] = 1e-13j
    noise[1, 0] = -1e-13j                               # Hermitian imaginary
    S = S_real.astype(complex) + noise

    # Without realify the gauge is genuinely complex here (else nothing proven).
    X_raw, _ = ortho_utils.lowdin_per_k(S)
    assert np.max(np.abs(X_raw.imag)) > 1e-3

    # _realify strips the sub-threshold noise, returning a real matrix ...
    Sr = ortho_utils._realify(S)
    assert not np.iscomplexobj(Sr)
    assert np.allclose(Sr, S_real, atol=_TOL)
    # ... so the composition _build_X_ibz uses yields a real gauge.
    X, _ = ortho_utils.lowdin_per_k(Sr)
    assert np.max(np.abs(X.imag)) < _TOL, "realify failed: X not real at self-TR"
    assert np.allclose(X @ S @ X.conj().T, np.eye(n), atol=_TOL)

    # A genuinely complex matrix is left untouched (same object returned).
    Sc = _hermitian_pd(rng, n)
    assert ortho_utils._realify(Sc) is Sc

    # The threshold discriminates at the right scale: an imaginary part just
    # ABOVE _REAL_TOL is kept (never silently discarded), one just below is
    # treated as noise and realified. This bounds what _realify can drop.
    base = S_real.astype(complex)
    hi = base.copy(); hi[0, 1] += 1e-9j;  hi[1, 0] -= 1e-9j    # |imag| > _REAL_TOL
    lo = base.copy(); lo[0, 1] += 1e-11j; lo[1, 0] -= 1e-11j   # |imag| < _REAL_TOL
    assert ortho_utils._realify(hi) is hi                      # kept, unchanged
    assert not np.iscomplexobj(ortho_utils._realify(lo))       # realified


def test_build_X_ibz_rejects_rank_reduction():
    # Canonical Löwdin drops a near-singular overlap eigenvalue, making X
    # rectangular (rank reduction). This is not supported end-to-end, so the
    # build must fail fast with an actionable error rather than a later
    # broadcasting error downstream.
    from green_mbtools.mint.ortho_utils import _build_X_ibz
    n = 4
    U, _ = np.linalg.qr(np.random.default_rng(0).standard_normal((n, n)))
    S = U @ np.diag([1e-12, 0.5, 1.0, 2.0]) @ U.T   # one eigenvalue below tol
    S = 0.5 * (S + S.T)
    S_ibz = S.astype(complex)[None]                 # (1, n, n)
    with pytest.raises(ValueError, match="rank"):
        _build_X_ibz("lowdin", S_ibz, None, None, None,
                     tol_sing=1e-9, tol_degen=1e-8)


@pytest.mark.parametrize("n", [1, 4, 7])
def test_symmetric_lowdin_per_k_hermitian_full_rank(rng, n):
    S = _hermitian_pd(rng, n)
    X, X_inv = ortho_utils.symmetric_lowdin_per_k(S)

    # Both factors are Hermitian and full square.
    assert X.shape == (n, n)
    assert X_inv.shape == (n, n)
    assert np.allclose(X, X.conj().T, atol=_TOL)
    assert np.allclose(X_inv, X_inv.conj().T, atol=_TOL)

    # X X_inv = I (left-inverse) and X S X† = I (orthogonalises S).
    assert np.allclose(X @ X_inv, np.eye(n), atol=_TOL)
    assert np.allclose(X @ S @ X.conj().T, np.eye(n), atol=_TOL)

    # X_inv X_inv† = S (X_inv = S^{1/2}).
    assert np.allclose(X_inv @ X_inv.conj().T, S, atol=_TOL)


def test_symmetric_lowdin_per_k_rank_deficient_pseudoinverse(rng):
    # S with one eigenvalue below tol: symmetric Löwdin should treat the
    # mode pseudo-inversely (zero it out) so the output stays Hermitian and
    # finite. X @ X_inv reduces to the projector onto the kept subspace.
    n = 5
    U, _ = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    evals = np.array([1e-12, 0.5, 1.0, 2.0, 3.0])
    S = U @ np.diag(evals) @ U.conj().T
    S = 0.5 * (S + S.conj().T)

    X, X_inv = ortho_utils.symmetric_lowdin_per_k(S, tol=1e-9)
    # Outputs are still square Hermitian and finite.
    assert X.shape == (n, n)
    assert X_inv.shape == (n, n)
    assert np.isfinite(X).all() and np.isfinite(X_inv).all()
    assert np.allclose(X, X.conj().T, atol=_TOL)
    assert np.allclose(X_inv, X_inv.conj().T, atol=_TOL)
    # X @ X_inv is the projector onto the kept (n-1)-dim subspace.
    P = X @ X_inv
    # Idempotent (P @ P = P) and rank n-1.
    assert np.allclose(P @ P, P, atol=_TOL)
    assert np.linalg.matrix_rank(P, tol=1e-7) == n - 1


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
