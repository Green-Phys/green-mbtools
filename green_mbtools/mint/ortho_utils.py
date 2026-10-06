import numpy as np
import scipy.linalg as LA
from .symmetry_utils import get_representation, get_spinor_representation

import logging

# Below this max|imag|, a k-point matrix is treated as numerically real and its
# eigenproblem is solved on the real part. This matters at self-TR k-points
# (e.g. Γ): tiny imaginary noise on a degenerate block would otherwise rotate
# the canonical-Löwdin/mo/natural eigenvectors into a complex gauge that
# violates X(-k) = X(k)* and breaks the conjugate df-pair reduction downstream.
_REAL_TOL = 1e-10


def lowdin_per_k(Sk, tol=1e-9):
    '''
    Canonical (Löwdin) orthogonalization for a single k-point.

    Returns ``(X, X_inv)`` in the convention used by
    ``common_utils.transform`` (i.e. transforms apply as ``X Z X†``)::

        X     = s^{-1/2} U†   (shape (n_ortho, nao))
        X_inv = U s^{+1/2}    (shape (nao, n_ortho))

    Eigenvalues of ``Sk`` below ``tol`` are discarded.

    This keeps the overlap eigenvectors ``U``, so unlike Hermitian
    symmetric Löwdin (whose ``S^{-1/2}`` is gauge-free) it carries a
    phase/degenerate-block gauge freedom. At self-TR k-points a real
    ``Sk`` is required for a real (hence ``X(-k) = X(k)*``-consistent)
    result; ``_build_X_ibz`` realifies numerically-real inputs before
    calling this, so that gauge policy lives there, not here.
    '''
    s_ev, s_eb = np.linalg.eigh(Sk)
    istart = s_ev.searchsorted(tol)
    s_sqrtev = np.sqrt(s_ev[istart:])
    x_pinv = s_eb[:, istart:] * s_sqrtev
    x = (s_eb[:, istart:].conj() * (1.0 / s_sqrtev)).T
    return x, x_pinv


def mo_per_k(Sk, C_k):
    '''
    Canonical-MO basis for a single k-point from MO coefficients ``C_k``
    satisfying ``C† S C = I``.

    Returns ``(X = C†, X_inv = S C)`` in the ``X Z X†`` convention.
    '''
    C_k = np.asarray(C_k, dtype=np.complex128)
    return C_k.conj().T, Sk @ C_k


def symmetric_lowdin_per_k(Sk, tol=1e-9):
    '''
    Symmetric (Hermitian) Löwdin orthogonalization for a single k-point.

    Returns ``(X = S^{-1/2}, X_inv = S^{+1/2})`` — both Hermitian
    ``(nao, nao)`` matrices — in the ``X Z X†`` convention.
    Distinguished from ``lowdin_per_k`` (canonical Löwdin) by being
    Hermitian rather than rectangular.

    Eigenvalues of ``Sk`` below ``tol`` are treated pseudo-inversely:
    their contribution is zeroed in both ``X`` and ``X_inv``, the same
    convention ``LA.pinv`` applies (see ``pesto/orth.py``). The output
    stays Hermitian and square, but in the rank-deficient case
    ``X @ X_inv`` reduces to the projector onto the kept subspace
    rather than the identity (Hermitian symmetric Löwdin cannot
    simultaneously be a strict left inverse on a rank-deficient
    basis). When linear dependencies are present and a strict
    ``X @ X_inv = I`` contract is required, use ``lowdin_per_k``
    (canonical, rectangular).
    '''
    s_ev, s_eb = np.linalg.eigh(Sk)
    kept = s_ev >= tol
    s_sqrt = np.zeros_like(s_ev)
    s_inv_sqrt = np.zeros_like(s_ev)
    s_sqrt[kept] = np.sqrt(s_ev[kept])
    s_inv_sqrt[kept] = 1.0 / np.sqrt(s_ev[kept])
    X = (s_eb * s_inv_sqrt) @ s_eb.conj().T
    X_inv = (s_eb * s_sqrt) @ s_eb.conj().T
    return X.astype(np.complex128), X_inv.astype(np.complex128)


def _S_inv_half(Sk):
    '''
    Return ``S^{-1/2}`` via Hermitian eigendecomposition of S.
    '''
    s_ev, s_eb = np.linalg.eigh(Sk)
    return (s_eb / np.sqrt(s_ev)) @ s_eb.conj().T


def natural_per_k(Sk, dmk):
    '''
    Natural-orbital basis at one k-point from density matrix ``dmk``.

    Diagonalises ``S^{-1/2} dm S^{-1/2}`` to obtain S-orthonormal
    natural orbitals ``C_NO = S^{-1/2} u`` (columns) with
    ``C_NO† S C_NO = I`` and ``C_NO† dm C_NO = diag(occ)``. Returns
    ``(X, X_inv)`` in the same ``X Z X†`` convention as ``mo_per_k``:

        X     = C_NO†      (shape (n_ortho, nao))
        X_inv = S @ C_NO   (shape (nao, n_ortho))
    '''
    S_inv_half = _S_inv_half(Sk)
    M = S_inv_half @ dmk @ S_inv_half
    M = 0.5 * (M + M.conj().T)
    _, u = np.linalg.eigh(M)
    C_NO = (S_inv_half @ u).astype(np.complex128)
    return C_NO.conj().T, Sk @ C_NO


def _natural_per_k_with_fock_tiebreak(Sk, dmk, Fk, tol_degen=1e-8):
    '''
    Natural orbitals at one k-point with Fock-within-block tie-breaking.

    Same S-orthonormal convention as ``natural_per_k``: diagonalises
    ``S^{-1/2} dm S^{-1/2}`` to obtain occupations and orbitals in the
    ``S^{-1/2}`` frame, then for each block of columns whose occupations
    agree to ``tol_degen`` additionally diagonalises the block-projected
    AO Fock ``C_NO_B† F C_NO_B`` and rotates the block accordingly.
    Returns ``(X = C_NO†, X_inv = S @ C_NO)``.
    '''
    S_inv_half = _S_inv_half(Sk)
    M = S_inv_half @ dmk @ S_inv_half
    M = 0.5 * (M + M.conj().T)
    n_occ, u = np.linalg.eigh(M)
    u = u.astype(np.complex128)
    nbf = u.shape[1]

    i = 0
    while i < nbf:
        j = i + 1
        while j < nbf and abs(n_occ[j] - n_occ[i]) < tol_degen:
            j += 1
        if j - i > 1:
            uB = u[:, i:j]
            C_NO_B = S_inv_half @ uB
            FB = C_NO_B.conj().T @ Fk @ C_NO_B
            FB = 0.5 * (FB + FB.conj().T)
            _, W = np.linalg.eigh(FB)
            u[:, i:j] = uB @ W
        i = j

    C_NO = (S_inv_half @ u).astype(np.complex128)
    return C_NO.conj().T, Sk @ C_NO

# Thresholds reported in the FNO truncation summary
_FNO_OCC_THRESHOLDS = (1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2)
_FNO_PCT_THRESHOLDS = (0.75, 0.90, 0.95, 0.99, 0.995, 0.999, 0.9995)


def fno_per_k(Sk, dmk):
    '''
    Frozen-natural-orbital transformation for a single k-point.

    Returns
    -------
    X, X_inv : ndarray
        Transformation in the ``X Z X†`` convention.
    nat_occ_vir : ndarray
        Virtual natural occupations sorted in descending order, used for the
        truncation summary built in ``_build_X_ibz``.
    '''
    s_ev, s_eb = np.linalg.eigh(Sk)
    S_half = (s_eb * np.sqrt(s_ev)) @ s_eb.conj().T
    S_inv_half = (s_eb / np.sqrt(s_ev)) @ s_eb.conj().T
    M = S_half @ dmk @ S_half
    M = 0.5 * (M + M.conj().T)
    nat_occ, Ck = np.linalg.eigh(M)
    nocc = int(nat_occ.sum().round() / 2)
    idx = np.argsort(nat_occ)[::-1]
    nat_occ, Ck = nat_occ[idx], Ck[:, idx]
    nat_occ_vir = nat_occ[nocc:]

    Ck_NO = (S_inv_half @ Ck).astype(np.complex128)
    return Ck_NO.conj().T, Sk @ Ck_NO, nat_occ_vir


def _pct_occ_fno(nat_occ_vir, blocks, thresh):
    '''Number of virtual orbitals to delete so that the kept ones carry at least
    ``thresh`` of the total virtual occupation, without splitting degenerate blocks.'''
    total = nat_occ_vir.sum()
    # No virtual orbitals, or a (numerically) idempotent density: the virtual
    # space carries no occupation, so every virtual orbital can be deleted.
    if len(nat_occ_vir) == 0 or total <= 1e-14:
        return len(nat_occ_vir)
    cum = 0.0
    nkeep = 0
    for b in blocks:
        cum += nat_occ_vir[b].sum()
        nkeep += len(nat_occ_vir[b])
        if cum / total > thresh:
            break
    return len(nat_occ_vir) - nkeep


def _fno_truncation_counts(nat_occ_vir):
    '''
    Number of virtual orbitals that could be deleted at one k-point, for each
    threshold in ``_FNO_OCC_THRESHOLDS`` (by occupation) and
    ``_FNO_PCT_THRESHOLDS`` (by fraction of total virtual occupation kept).
    Degenerate blocks are never split in the percentage criterion.
    '''
    blocks = []
    start = 0
    for i in range(1, len(nat_occ_vir)):
        if not np.isclose(nat_occ_vir[i], nat_occ_vir[i - 1], atol=1e-8):
            blocks.append(slice(start, i))
            start = i
    blocks.append(slice(start, len(nat_occ_vir)))

    by_occ = [np.count_nonzero(nat_occ_vir < thr) for thr in _FNO_OCC_THRESHOLDS]
    by_pct = [_pct_occ_fno(nat_occ_vir, blocks, pct) for pct in _FNO_PCT_THRESHOLDS]
    return np.array(by_occ), np.array(by_pct)


def _log_fno_summary(nvir, ndel_occ, ndel_pct):
    '''Log, once for all k-points, the minimum over k of the deletable orbitals.'''
    lines = [f"FNO truncation summary ({nvir} virtual orbitals, minimum over all k-points):"]
    for thr, n in zip(_FNO_OCC_THRESHOLDS, ndel_occ):
        lines.append(f"  occupation < {thr:.0e}        -> delete {n} orbitals")
    for pct, n in zip(_FNO_PCT_THRESHOLDS, ndel_pct):
        lines.append(f"  keep {pct:7.2%} of virtual occ. -> delete {n} orbitals")
    logging.info("\n".join(lines))

    
def _realify(M):
    '''
    Strip numerical imaginary noise from a k-point matrix.

    Returns ``M.real`` when ``max|imag(M)| < _REAL_TOL`` (the case at
    self-TR k-points, where ``S``/``F``/``dm`` are physically real), else
    ``M`` unchanged. Centralizing this in ``_build_X_ibz`` keeps the
    real-gauge policy in one place: the gauge-sensitive modes (lowdin, mo,
    natural) then diagonalize a real matrix at self-TR points and produce a
    real, ``X(-k) = X(k)*``-consistent X, while gauge-free
    symmetric_lowdin is unaffected. ``None`` passes through.

    The criterion is numerical realness, not an explicit self-TR test.
    Applying it to every IBZ representative is safe: a generic (non-self-TR)
    point has O(1) imaginary parts from Bloch phases, so this never fires
    there; where it does fire the discarded imaginary part is below
    ``_REAL_TOL`` and physically negligible; and the star stays internally
    consistent because it is propagated from this (realified) representative.
    '''
    if M is None:
        return M
    return M.real if np.max(np.abs(M.imag)) < _REAL_TOL else M


def _build_X_ibz(mode, S_ibz, F_ibz, dm_ibz, mo_coeff_ibz,
                 tol_sing, tol_degen):
    '''
    Per-IBZ orthogonalization step shared by ``build_X_kspace`` and
    ``build_X_kspace_from_ao_reps``.

    Returns
    -------
    X_per_irrep, X_inv_per_irrep : lists of length n_ibz
        Per-IBZ-point X and X_inv in the ``X Z X†`` convention.
    '''
    n_ibz = np.asarray(S_ibz).shape[0]
    X_per_irrep = [None] * n_ibz
    Xinv_per_irrep = [None] * n_ibz

    fno_ndel_occ = []   # per-k deletable counts, reduced to the minimum after the loop
    fno_ndel_pct = []
    fno_nvir = None
    
    for i_ir in range(n_ibz):
        # Centralized real-gauge policy: at self-TR k-points S/F/dm are
        # physically real, so strip the imaginary noise once, up front. Every
        # gauge-sensitive mode below then diagonalizes a real matrix there and
        # yields a real, X(-k) = X(k)*-consistent X; symmetric_lowdin is
        # unaffected. None (unused inputs for a given mode) passes through.
        Sk = _realify(S_ibz[i_ir])
        Fk = _realify(F_ibz[i_ir]) if F_ibz is not None else None
        dmk = _realify(dm_ibz[i_ir]) if dm_ibz is not None else None

        if mode == "lowdin":
            x, x_inv = lowdin_per_k(Sk, tol=tol_sing)
        elif mode == "symmetric_lowdin":
            x, x_inv = symmetric_lowdin_per_k(Sk, tol=tol_sing)
        elif mode == "mo":
            if mo_coeff_ibz is not None:
                Ck = mo_coeff_ibz[i_ir]
            else:
                if Fk.ndim == 3:
                    Fk = 0.5 * (Fk[0] + Fk[1])
                _, Ck = LA.eigh(Fk, Sk)
            x, x_inv = mo_per_k(Sk, Ck)
        elif mode == "natural":
            if dmk.ndim == 3:
                dmk = 0.5 * (dmk[0] + dmk[1])
            if Fk.ndim == 3:
                Fk = 0.5 * (Fk[0] + Fk[1])
            x, x_inv = _natural_per_k_with_fock_tiebreak(
                Sk, dmk, Fk, tol_degen=tol_degen
            )
        elif mode == "fno":
            x, x_inv, nat_occ_vir = fno_per_k(Sk, dmk)
            ndel_occ, ndel_pct = _fno_truncation_counts(nat_occ_vir)
            fno_ndel_occ.append(ndel_occ)
            fno_ndel_pct.append(ndel_pct)
            fno_nvir = len(nat_occ_vir)
        else:
            raise ValueError(
                f"build_X_kspace: unknown mode {mode!r} "
                "(expected 'lowdin', 'symmetric_lowdin', 'mo', 'natural' or 'fno')."
            )
        X_per_irrep[i_ir] = np.asarray(x, dtype=np.complex128)
        Xinv_per_irrep[i_ir] = np.asarray(x_inv, dtype=np.complex128)

    if mode == "fno":
        # The same number of orbitals must be kept at every k-point, so the
        # number that can safely be deleted is the minimum over k.
        _log_fno_summary(fno_nvir,
                         np.min(fno_ndel_occ, axis=0),
                         np.min(fno_ndel_pct, axis=0))
        
    # Reject rank-deficient orthogonalization. Near-singular overlap eigenvalues
    # (< tol_sing) get dropped, which makes X non-invertible: canonical Löwdin
    # returns a rectangular X, while symmetric Löwdin returns a square X whose
    # X_inv is only a pseudo-inverse (X_inv @ X is a projector, not I). Differing
    # retained ranks across IBZ points also break star propagation. None of this
    # is supported end-to-end (downstream transform/symmetry operators and the
    # Green file format assume a square, invertible, AO-sized X). Fail fast with
    # an actionable message rather than a later NumPy broadcasting error or a
    # generic "Orthogonal transformation failed" RuntimeError from transform().
    shapes = {x.shape for x in X_per_irrep}
    if len(shapes) != 1:
        raise ValueError(
            f"build_X_kspace: mode {mode!r} produced X of differing shapes "
            f"across IBZ points ({sorted(shapes)}) — different retained ranks "
            "from dropping near-singular overlap eigenvalues. Rank reduction is "
            "not supported; use a linear-dependence-free basis."
        )
    n_basis = X_per_irrep[0].shape[1]
    eye = np.eye(n_basis)
    for i_ir, (x, x_inv) in enumerate(zip(X_per_irrep, Xinv_per_irrep)):
        if x.shape[0] != n_basis or not np.allclose(x_inv @ x, eye, atol=1e-8):
            raise ValueError(
                f"build_X_kspace: mode {mode!r} produced a rank-deficient, "
                f"non-invertible X at IBZ point {i_ir} (shape {x.shape}, "
                "X_inv @ X != I). Near-singular overlap eigenvalues (< tol_sing) "
                "were dropped: canonical Löwdin becomes rectangular, symmetric "
                "Löwdin's X_inv becomes a pseudo-inverse. Rank-deficient "
                "orthogonalization is not supported end-to-end; use a "
                "linear-dependence-free basis."
            )

    return X_per_irrep, Xinv_per_irrep


def _propagate_with_reps(X_per_irrep, Xinv_per_irrep, ibz2bz, bz2ibz,
                         k_sym_transform_ao, tr_conj):
    '''
    Propagate per-IBZ ``(X, X_inv)`` to every BZ point given precomputed
    AO-space rotations ``k_sym_transform_ao`` and TR flags ``tr_conj``.

    Convention (matching ``common_utils.store_kstruct_ops_info``):

    - Non-TR (``tr_conj[k] == False``): the AO rotation ``U(k)`` is
      ``get_representation(k, stars_ops[k], ...)`` (or the spinor analog),
      and ``M(k) = U M(k_ir) U†``. Then

          X(k)     = X(k_ir) @ U†
          X_inv(k) = U @ X_inv(k_ir)

    - TR (``tr_conj[k] == True``): the stored rotation already
      incorporates the conjugation factor (e.g. for X2C double group it
      is ``(U_spinor @ Θ).conj()``), and the reconstruction is
      ``M(k) = (U M(k_ir) U†).conj()``. Then

          X(k)     = (X(k_ir) @ U†).conj()
          X_inv(k) = (U @ X_inv(k_ir)).conj()

    so that ``X(k) S(k) X(k)† = I`` at every BZ point.
    '''
    ibz2bz_arr = np.asarray(ibz2bz)
    bz2ibz_arr = np.asarray(bz2ibz)
    nk = bz2ibz_arr.shape[0]

    sample = X_per_irrep[0]
    n_ortho, n_basis = sample.shape

    X_k = np.zeros((nk, n_ortho, n_basis), dtype=np.complex128)
    X_inv_k = np.zeros((nk, n_basis, n_ortho), dtype=np.complex128)

    if tr_conj is None:
        tr_conj = np.zeros(nk, dtype=bool)

    for ik in range(nk):
        i_ir = int(bz2ibz_arr[ik])
        X_ir = X_per_irrep[i_ir]
        Xinv_ir = Xinv_per_irrep[i_ir]
        u = k_sym_transform_ao[ik]

        Xk = X_ir @ u.conj().T
        Xinvk = u @ Xinv_ir
        if tr_conj[ik]:
            Xk = Xk.conj()
            Xinvk = Xinvk.conj()
        X_k[ik] = Xk
        X_inv_k[ik] = Xinvk

    return X_k, X_inv_k


def _propagate_X_to_star(
    X_per_irrep,
    Xinv_per_irrep,
    kstruct,
    mycell,
    spinor=False,
):
    '''
    Build precomputed AO-rotation arrays from ``kstruct`` + ``mycell`` and
    delegate to ``_propagate_with_reps``.
    '''
    nk = kstruct.nkpts
    stars_ops = kstruct.stars_ops_bz
    sample = X_per_irrep[0]
    nbasis_out = sample.shape[1]

    k_sym_transform_ao = np.zeros((nk, nbasis_out, nbasis_out),
                                  dtype=np.complex128)
    tr_conj_bz = kstruct.time_reversal_symm_bz

    if spinor:
        nao = mycell.nao_nr()
        theta = np.kron(np.array([[0, 1], [-1, 0]], dtype=np.complex128),
                        np.eye(nao))

    for ik in range(nk):
        iop = stars_ops[ik]
        if spinor:
            u = get_spinor_representation(ik, iop, mycell, kstruct)
            if tr_conj_bz[ik]:
                u = (u @ theta).conj()
        else:
            u = get_representation(ik, iop, mycell, kstruct)
        k_sym_transform_ao[ik] = u

    return _propagate_with_reps(
        X_per_irrep, Xinv_per_irrep,
        kstruct.ibz2bz, kstruct.bz2ibz,
        k_sym_transform_ao, tr_conj_bz,
    )


def build_X_kspace(
    mode,
    kstruct,
    mycell,
    S_ibz,
    *,
    F_ibz=None,
    dm_ibz=None,
    mo_coeff_ibz=None,
    spinor=False,
    tol_sing=1e-9,
    tol_degen=1e-8,
):
    '''
    Build orthogonalization matrices ``(X_k, X_inv_k)`` over the full BZ.

    ``X`` is constructed only at IBZ k-points using one of the per-k
    primitives, then propagated to every star member via the space-group
    + time-reversal representations carried by ``kstruct``. See
    ``_propagate_X_to_star`` for the convention.

    Parameters
    ----------
    mode : {"lowdin", "symmetric_lowdin", "mo", "natural"}
        IBZ primitive to use.
    kstruct : pyscf.pbc.lib.kpts.KPoints
        Same object the rest of mbtools uses (e.g. from
        ``kpt_utils.build_q_struct(mycell, kmesh, space_symm=True,
        tr_symm=True)``).
    mycell : pyscf.pbc.gto.Cell
        Required for the AO-space representations.
    S_ibz : (n_ibz, n, n) ndarray
        Overlap at IBZ k-points, in the order
        ``kstruct.kpts_scaled[kstruct.ibz2bz]``.
    F_ibz : (n_ibz, n, n) ndarray, optional
        Fock at IBZ. Required for ``mode="natural"`` (degeneracy
        tie-breaking) and for ``mode="mo"`` when ``mo_coeff_ibz`` is not
        provided (spin-averaged Fock is used to derive MOs).
    dm_ibz : (n_ibz, n, n) ndarray, optional
        Spin-averaged / total density matrix at IBZ. Required for
        ``mode="natural"``.
    mo_coeff_ibz : (n_ibz, n, n_mo) ndarray, optional
        MO coefficients at IBZ. Preferred input for ``mode="mo"``.
    spinor : bool, default False
        If True, use the double-group spinor representation
        (``get_spinor_representation``). Currently only supported with
        ``mode="lowdin"``.
    tol_sing : float
        Threshold for discarding small eigenvalues of ``S`` in the
        Löwdin primitive.
    tol_degen : float
        Tolerance used by the natural-mode Fock tie-breaker to detect
        degenerate occupation blocks.

    Returns
    -------
    X_k     : (nk, n_ortho, n) complex128 ndarray
    X_inv_k : (nk, n,        n_ortho) complex128 ndarray
    '''
    if spinor and mode not in ("lowdin", "symmetric_lowdin"):
        raise NotImplementedError(
            f"build_X_kspace: spinor=True only supported for mode in "
            f"{{'lowdin', 'symmetric_lowdin'}}, got mode={mode!r}."
        )
    if mode == "natural" and (dm_ibz is None or F_ibz is None):
        raise ValueError(
            "build_X_kspace: mode='natural' requires dm_ibz and F_ibz."
        )
    if mode == "mo" and mo_coeff_ibz is None and F_ibz is None:
        raise ValueError(
            "build_X_kspace: mode='mo' requires mo_coeff_ibz or F_ibz."
        )
    if mode == "fno" and dm_ibz is None:
          raise ValueError(
            "build_X_kspace: mode='fno' requires dm_ibz."
        )

    ibz2bz = kstruct.ibz2bz
    n_ibz = len(ibz2bz)
    S_ibz = np.asarray(S_ibz)
    if S_ibz.shape[0] != n_ibz:
        raise ValueError(
            f"build_X_kspace: S_ibz has {S_ibz.shape[0]} k-points but "
            f"kstruct has {n_ibz} IBZ points."
        )

    X_per_irrep, Xinv_per_irrep = _build_X_ibz(
        mode, S_ibz, F_ibz, dm_ibz, mo_coeff_ibz, tol_sing, tol_degen
    )

    return _propagate_X_to_star(
        X_per_irrep, Xinv_per_irrep, kstruct, mycell, spinor=spinor
    )


def build_X_kspace_from_ao_reps(
    mode,
    S_ibz,
    ibz2bz,
    bz2ibz,
    k_sym_transform_ao,
    *,
    tr_conj=None,
    F_ibz=None,
    dm_ibz=None,
    mo_coeff_ibz=None,
    tol_sing=1e-9,
    tol_degen=1e-8,
):
    '''
    Build ``(X_k, X_inv_k)`` from precomputed AO-space rotations.

    Identical to ``build_X_kspace`` but consumes the symmetry information
    that ``common_utils.store_kstruct_ops_info`` already writes into
    ``input.h5`` under ``/symmetry/k`` —

      - ``ibz2bz``            (n_ibz,)        BZ indices of IBZ reps
      - ``bz2ibz``            (nk,)           Full-BZ index of the IBZ representative for each
                                              BZ point, matching `/symmetry/k/bz2ibz`.
                                              Compact IBZ positions are not accepted
      - ``k_sym_transform_ao`` (nk, n, n)     stored AO rotation U(k)
      - ``tr_conj``           (nk,)  bool     TR partner flags

    — instead of (kstruct, mycell). Designed for callers like the SEET
    pre-processor which read these arrays from h5 and do not carry a
    PySCF ``KPoints`` object.

    For non-TR points, ``M(k) = U M(k_ir) U†``; for TR points the stored
    ``U`` already incorporates the conjugation factor (e.g. for X2C
    double group, ``(U_spinor @ Θ).conj()``) and the reconstruction is
    ``M(k) = (U M(k_ir) U†).conj()``. The function applies the matching
    rule for X automatically — callers do not need to special-case TR.

    Parameters mirror ``build_X_kspace``; ``mode`` semantics are
    identical. ``spinor`` is implicit in the basis size of
    ``S_ibz`` / ``k_sym_transform_ao`` (both are ``nso × nso`` for X2C).

    Returns
    -------
    X_k     : (nk, n_ortho, n) complex128 ndarray
    X_inv_k : (nk, n,        n_ortho) complex128 ndarray
    '''
    if mode == "natural" and (dm_ibz is None or F_ibz is None):
        raise ValueError(
            "build_X_kspace_from_ao_reps: mode='natural' requires "
            "dm_ibz and F_ibz."
        )
    if mode == "mo" and mo_coeff_ibz is None and F_ibz is None:
        raise ValueError(
            "build_X_kspace_from_ao_reps: mode='mo' requires "
            "mo_coeff_ibz or F_ibz."
        )

    ibz2bz_arr = np.asarray(ibz2bz)
    n_ibz = ibz2bz_arr.shape[0]
    S_ibz = np.asarray(S_ibz)
    if S_ibz.shape[0] != n_ibz:
        raise ValueError(
            f"build_X_kspace_from_ao_reps: S_ibz has {S_ibz.shape[0]} "
            f"k-points but ibz2bz has {n_ibz} entries."
        )

    X_per_irrep, Xinv_per_irrep = _build_X_ibz(
        mode, S_ibz, F_ibz, dm_ibz, mo_coeff_ibz, tol_sing, tol_degen
    )

    # ``/symmetry/k/bz2ibz`` stores, for each BZ point, the *full-BZ index*
    # of its IBZ representative (common_utils.save_data does
    # ``ind = ir_list[ind]``), matching pesto.mb.to_full_bz. _propagate_with_reps
    # indexes the per-IBZ arrays directly, so convert those representative
    # BZ indices into compact IBZ positions [0, n_ibz) via ibz2bz.
    # NOTE -    in the function _propagate_X_to_star, we use kstruct.bz2ibz which is same as
    #           the bz2ib_compact defined below.
    bz2ibz_arr = np.asarray(bz2ibz)
    pos_in_ibz = {int(b): i for i, b in enumerate(ibz2bz_arr)}

    # Check bz2ibz contains full-bz indices, not the compact ibz ones
    invalid = np.unique(bz2ibz_arr[~np.isin(bz2ibz_arr, ibz2bz_arr)])
    if invalid.size:
        raise ValueError(
            "bz2ibz must contain representative BZ indices from ibz2bz; "
            f"found invalid values: {invalid.tolist()}"
        )

    bz2ibz_compact = np.array(
        [pos_in_ibz[int(b)] for b in bz2ibz_arr], dtype=int
    )

    return _propagate_with_reps(
        X_per_irrep, Xinv_per_irrep,
        ibz2bz_arr, bz2ibz_compact,
        np.asarray(k_sym_transform_ao, dtype=np.complex128),
        np.asarray(tr_conj, dtype=bool) if tr_conj is not None else None,
    )


def _build_naf(M):
    '''
    Diagonalize the aux-metric M and return the (unitary) aux-AO -> NAF rotation.

    Unlike lowdin_per_k/symmetric_lowdin_per_k, this is a pure basis
    rotation (no rescaling): Y is unitary, so Y_inv = Y^dagger exactly.
    Eigenvalues are sorted descending so that, if a caller later truncates
    (in the scGW reader, not here), the most important NAFs come first.

    Returns
    -------
    Y, Y_inv : (NQ, NQ) complex128 ndarrays
    '''
    M = np.asarray(M, dtype=np.complex128)
    M = _realify(M)
    M = 0.5 * (M + M.conj().T)
    eigval, eigvec = np.linalg.eigh(M)
    idx = np.argsort(eigval)[::-1]
    eigval, eigvec = eigval[idx], eigvec[:, idx]
    # Y Z Y^dagger convention
    Y = eigvec.conj().T.astype(np.complex128)
    Y_inv = eigvec.astype(np.complex128)
    
    _log_naf_summary(eigval)

    return Y, Y_inv


# Thresholds on the normalized NAF eigenvalues (eigval / eigval_max) reported in the NAF summary
_NAF_EIG_THRESHOLDS = (1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 1e-4, 1e-5, 1e-6)

def _log_naf_summary(eigval):
    '''
    Log the NAF truncation summary: for each threshold in ``_NAF_EIG_THRESHOLDS``,
    the number of auxiliary functions whose normalized eigenvalue falls below it.
    ``eigval`` must be sorted in descending order.
    '''
    norm_eigval = eigval / eigval[0]
    lines = [f"NAF truncation summary ({len(eigval)} auxiliary functions, "
             f"normalized eigenvalues in [{norm_eigval[-1]:.2e}, 1]):"]
    for thr in _NAF_EIG_THRESHOLDS:
        lines.append(f"  eigenvalue < {thr:.0e}          -> delete "
                     f"{np.count_nonzero(norm_eigval < thr)} auxiliary functions")
    logging.info("\n".join(lines))
    logging.debug(f"NAF normalized eigenvalues: {norm_eigval}")
