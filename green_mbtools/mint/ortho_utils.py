import numpy as np
import scipy.linalg as LA
from functools import reduce
from ..pesto import orth
from .symmetry_utils import get_representation, get_spinor_representation

def wrap_k(k):
    while k < 0 :
        k = 1 + k
    while (k - 9.9999999999e-1) > 0.0 :
        k = k - 1
    return k

def normalize(Sv):
    Svo = np.zeros(Sv.shape, dtype=Sv.dtype)
    for iv, v in enumerate(Sv.T):
        ss = np.sign(v[0])*(-1)**iv
        Svo[:,iv] = ss*v.T
    return Svo

def avg(kpts, X):
    '''
    Average momentum dependent quantity 
    '''
    xxx = (X[kpts[0]] + X[kpts[1 % len(kpts)]].conj())/2
    for ikk, kk in enumerate(kpts):
        X[kk] = (xxx if (ikk % 2) == 0 else xxx.conj())

def lowdin_per_k(Sk, tol=1e-9):
    '''
    Symmetric (Löwdin) S^{-1/2} orthogonalization for a single k-point.

    Returns ``(X, X_inv)`` in the convention used by
    ``common_utils.transform`` (i.e. transforms apply as ``X Z X†``):
        X     = S^{-1/2}   (shape (n_ortho, nao))
        X_inv = S^{+1/2}   (shape (nao, n_ortho))
    Eigenvalues of ``Sk`` below ``tol`` are discarded.
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

    for i_ir in range(n_ibz):
        Sk = S_ibz[i_ir]
        if mode == "lowdin":
            x, x_inv = lowdin_per_k(Sk, tol=tol_sing)
        elif mode == "mo":
            if mo_coeff_ibz is not None:
                Ck = mo_coeff_ibz[i_ir]
            else:
                Fk = F_ibz[i_ir]
                if Fk.ndim == 3:
                    Fk = 0.5 * (Fk[0] + Fk[1])
                _, Ck = LA.eigh(Fk, Sk)
            x, x_inv = mo_per_k(Sk, Ck)
        elif mode == "natural":
            dmk = dm_ibz[i_ir]
            Fk = F_ibz[i_ir]
            if dmk.ndim == 3:
                dmk = 0.5 * (dmk[0] + dmk[1])
            if Fk.ndim == 3:
                Fk = 0.5 * (Fk[0] + Fk[1])
            x, x_inv = _natural_per_k_with_fock_tiebreak(
                Sk, dmk, Fk, tol_degen=tol_degen
            )
        else:
            raise ValueError(
                f"build_X_kspace: unknown mode {mode!r} "
                "(expected 'lowdin', 'mo', or 'natural')."
            )
        X_per_irrep[i_ir] = np.asarray(x, dtype=np.complex128)
        Xinv_per_irrep[i_ir] = np.asarray(x_inv, dtype=np.complex128)

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
    mode : {"lowdin", "mo", "natural"}
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
    if spinor and mode != "lowdin":
        raise NotImplementedError(
            f"build_X_kspace: spinor=True only supported for mode='lowdin', "
            f"got mode={mode!r}."
        )
    if mode == "natural" and (dm_ibz is None or F_ibz is None):
        raise ValueError(
            "build_X_kspace: mode='natural' requires dm_ibz and F_ibz."
        )
    if mode == "mo" and mo_coeff_ibz is None and F_ibz is None:
        raise ValueError(
            "build_X_kspace: mode='mo' requires mo_coeff_ibz or F_ibz."
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
      - ``bz2ibz``            (nk,)           IBZ index for each BZ point
      - ``k_sym_transform_ao``(nk, n, n)      stored AO rotation U(k)
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

    return _propagate_with_reps(
        X_per_irrep, Xinv_per_irrep,
        ibz2bz_arr, np.asarray(bz2ibz),
        np.asarray(k_sym_transform_ao, dtype=np.complex128),
        np.asarray(tr_conj, dtype=bool) if tr_conj is not None else None,
    )


def symmetrical_orbitals(S, dm, F_up, F_dn, T_up, T_dn, kmesh, Debug_print=False):
    X_k = []
    X_inv_k = []
    pairing = {}
    unique_kpts = []
    # generate list of unique k-points by applying an inversion symmetry
    for ik, k in enumerate(kmesh):
        for ikp, kp in enumerate(kmesh):
            minus_k = [wrap_k(-kk) for kk in k]
            if np.allclose(minus_k, kp) and ikp not in pairing.keys():
                pairing[ikp] = ik
                unique_kpts.append(ik)
        if ik not in pairing.keys():
            pairing[ik] = ik
    # find the list of k-points that are the same by inversion symmetry and apply the k-symmetrization
    for ik in unique_kpts:
        kpts = []
        for ikp in pairing.keys():
            if pairing[ikp] == ik:
                kpts.append(ikp)
        avg(kpts, dm)
        avg(kpts, F_up)
        avg(kpts, F_dn)
        avg(kpts, T_up)
        avg(kpts, T_dn)

    # Compute transformation matrices
    for ik, k in enumerate(kmesh):
        # check whether matrices should be real by inversion symmetry
        if np.allclose(k, [wrap_k(-kk) for kk in k]):
            if Debug_print:
                print(k)
            dmk = dm[ik].real
            Sk = S[ik].real
        else:
            dmk = dm[ik]
            Sk = S[ik]


        x, x_pinv = lowdin_per_k(Sk)

        n_ortho, n_nonortho = x.shape

        if Debug_print:
            print("S", np.allclose(np.dot(x_pinv, x_pinv.conj().T), S[ik]))
            print("X X^-1", np.allclose(np.dot(x, x_pinv), np.eye(x.shape[0])))
            print("X* X^-1*", np.allclose(np.dot(x.conj().T, x_pinv.conj().T), np.eye(x.shape[0])))

        # check that direct and inverse symmetries are consistent
        assert(np.allclose(np.dot(x, x_pinv), np.eye(x.shape[0])))

        # store transformation basis for current k-point
        xx_inv = x_pinv.conj().T.astype(np.complex128)
        xx = x.conj().T.astype(np.complex128)
        X_inv_k.append(xx_inv.copy())
        X_k.append(xx.copy())

        minus_k = [wrap_k(-kk) for kk in k]

        for ikk, kk in enumerate(kmesh):
            if np.allclose(kk, minus_k) and ikk < ik:
                # X_inv_k.append(np.copy(X_inv_k[ikk]))
                # X_k.append(np.copy(X_k[ikk]))
                if Debug_print:
                    print("============== reduced point ================")
                    print("S", np.allclose(S[ikk].conj(), S[ik]))
                    print("Sev", np.allclose(LA.eigvalsh(S[ikk]), LA.eigvalsh(S[ik])))
                    # print LA.eigvalsh(S[ikk]), "\n\n=====\n\n", LA.eigvalsh(S[ik])
                    print("S", np.allclose(np.dot(X_inv_k[ikk].T, X_inv_k[ikk].conj()), S[ik]))
                    print("S2", np.allclose(X_inv_k[ikk], xx_inv.conj().T.conj()))
                assert(np.allclose(np.dot(X_inv_k[ik].conj().T, X_inv_k[ik]), S[ik]))
                assert(np.allclose(np.dot(X_inv_k[ikk].T, X_inv_k[ikk].conj()), S[ik]))
                X_inv_k[ik] = np.copy(X_inv_k[ikk].conj())
                X_k[ik] = np.copy(X_k[ikk].conj())
                continue
    return X_inv_k, X_k

def canonical_orbitals(S, dm, F_up, F_dn, T_up, T_dn, kmesh, Debug_print=False):
    X_k = []
    X_inv_k = []
    pairing = {}
    unique_kpts = []
    # generate list of unique k-points by applying an inversion symmetry
    for ik, k in enumerate(kmesh):
        for ikp, kp in enumerate(kmesh):
            minus_k = [wrap_k(-kk) for kk in k]
            if np.allclose(minus_k, kp) and ikp not in pairing.keys():
                pairing[ikp] = ik
                unique_kpts.append(ik)
        if ik not in pairing.keys():
            pairing[ik] = ik
    # find the list of k-points that are the same by inversion symmetry and apply the k-symmetrization
    for ik in unique_kpts:
        kpts = []
        for ikp in pairing.keys():
            if pairing[ikp] == ik:
                kpts.append(ikp)
        avg(kpts, dm)
        #avg(kpts, S)
        avg(kpts, F_up)
        avg(kpts, F_dn)
        avg(kpts, T_up)
        avg(kpts, T_dn)

    # Compute transformation matrices
    for ik, k in enumerate(kmesh):
        # check whether matrices should be real by inversion symmetry
        if np.allclose(k, [wrap_k(-kk) for kk in k]):
            if Debug_print:
                print(k)
            dmk = dm[ik].real
            Sk = S[ik].real
        else:
            dmk = dm[ik]
            Sk = S[ik]


        x, x_pinv = lowdin_per_k(Sk)

        n_ortho, n_nonortho = x.shape
        if Debug_print:
            print("S", np.allclose(np.dot(x_pinv, x_pinv.conj().T), S[ik]))
            print("X X^-1", np.allclose(np.dot(x, x_pinv), np.eye(x.shape[0])))
            print("X* X^-1*", np.allclose(np.dot(x.conj().T, x_pinv.conj().T), np.eye(x.shape[0])))

        # check that direct and inverse symmetries are consistent
        assert(np.allclose(np.dot(x, x_pinv), np.eye(x.shape[0])))

        # store transformation basis for current k-point
        xx_inv = x_pinv.conj().T
        xx = x.conj().T
        X_inv_k.append(xx_inv.copy())
        X_k.append(xx.copy())

        minus_k = [wrap_k(-kk) for kk in k]
        for ikk, kk in enumerate(kmesh):
            if np.allclose(kk, minus_k) and ikk < ik:
                # X_inv_k.append(np.copy(X_inv_k[ikk]))
                # X_k.append(np.copy(X_k[ikk]))
                if Debug_print:
                    print("============== reduced point ================")
                    print("S", np.allclose(S[ikk].conj(), S[ik]))
                    print("Sev", np.allclose(LA.eigvalsh(S[ikk]), LA.eigvalsh(S[ik])))
                    # print LA.eigvalsh(S[ikk]), "\n\n=====\n\n", LA.eigvalsh(S[ik])
                    print("S", np.allclose(np.dot(X_inv_k[ikk].T, X_inv_k[ikk].conj()), S[ik]))
                    print("S2", np.allclose(X_inv_k[ikk], xx_inv.conj().T.conj()))
                assert(np.allclose(np.dot(X_inv_k[ik].conj().T, X_inv_k[ik]), S[ik]))
                assert(np.allclose(np.dot(X_inv_k[ikk].T, X_inv_k[ikk].conj()), S[ik]))
                X_inv_k[ik] = np.copy(X_inv_k[ikk].conj())
                X_k[ik] = np.copy(X_k[ikk].conj())
                continue

    return X_inv_k, X_k

def natural_orbitals(S, dm, F_up, F_dn, T_up, T_dn, kmesh, Debug_print=False):
    '''
    Compute transformation matricies for natural orbtial basis from spin-averaged density matrix
    ---
    
    '''
    X_k = []
    X_inv_k = []
    pairing = {}
    unique_kpts = []
    # generate list of unique k-points by applying an inversion symmetry
    for ik, k in enumerate(kmesh):
        for ikp, kp in enumerate(kmesh):
            minus_k = [wrap_k(-kk) for kk in k]
            if np.allclose(minus_k, kp) and ikp not in pairing.keys():
                pairing[ikp] = ik
                unique_kpts.append(ik)
        if ik not in pairing.keys():
            pairing[ik] = ik
    # find the list of k-points that are the same by inversion symmetry and apply the k-symmetrization
    for ik in unique_kpts:
        kpts = []
        for ikp in pairing.keys():
            if pairing[ikp] == ik:
                kpts.append(ikp)
        avg(kpts, dm)
        avg(kpts, F_up)
        avg(kpts, F_dn)
        avg(kpts, T_up)
        avg(kpts, T_dn)

    # Compute transformation matrices
    for ik, k in enumerate(kmesh):
        # check whether matrices should be real by inversion symmetry
        if np.allclose(k, [wrap_k(-kk) for kk in k]):
            dmk = dm[ik].real
            Sk = S[ik].real
        else:
            dmk = dm[ik]
            Sk = S[ik]

        # compute natural orbital basis
        Svi, Sv = natural_per_k(Sk, dmk)

        # Check that transformations for the inversion symmetries are consistent
        if ik != pairing[ik] and not np.allclose(reduce(np.dot, (Sv, dm[ik], Sv.conj().T)), reduce(np.dot, (Sv.conj(), dm[pairing[ik]], Sv.T)),
                          atol=1e-7):
            print(False, np.max(np.abs(reduce(np.dot, (Sv, dm[ik], Sv.conj().T)) - reduce(np.dot, (Sv.conj(), dm[pairing[ik]], Sv.T))) ))
            print(ik, k, pairing[ik], kmesh[pairing[ik]])
            print("A", reduce(np.dot, (Sv, dm[ik], Sv.conj().T)))
            print("B", reduce(np.dot, (Sv.conj(), dm[pairing[ik]], Sv.T)))
        else:
            pass

        # here we assume that the direct transformation is for quantities like Self-energy and Fock matrix
        # and the inverse transformation if for quantities like density matrix and Green's function
        x_pinv = Sv
        x  = Svi

        # check that direct and inverse symmetries are consistent
        assert(np.allclose(np.dot(x, x_pinv), np.eye(x.shape[0])))

        dmkd = orth.transform_per_k(dmk, x_pinv.conj().T, x.conj().T)
        assert(np.allclose(dmkd, np.diag(np.diag(dmkd))))

        # store transformation basis for current k-point
        X_inv_k.append(x_pinv.copy())
        X_k.append(x.copy())

    return X_inv_k, X_k
