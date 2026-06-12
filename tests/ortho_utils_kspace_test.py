"""Tests for the kstruct-driven full-BZ orthogonalization layer in
``green_mbtools.mint.ortho_utils``.

These cover the new ``build_X_kspace`` and ``_natural_per_k_with_fock_tiebreak``
helpers introduced by the orthofix work. The per-k primitives
(``lowdin_per_k``, ``mo_per_k``, ``natural_per_k``) are tested separately in
``ortho_utils_test.py``.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest
import scipy.linalg as LA

from green_mbtools.mint import ortho_utils
from green_mbtools.mint.ortho_utils import (
    build_X_kspace,
    build_X_kspace_from_ao_reps,
    _natural_per_k_with_fock_tiebreak,
)


_TOL = 1e-9


def _build_h2_cell_and_kstruct(nk=3, space_symm=True, tr_symm=True):
    """Build a small H2 PBC cell and a kstruct on an nk×nk×nk mesh."""
    from pyscf.pbc import gto as pbc_gto
    from green_mbtools.mint.kpt_utils import build_q_struct

    cell = pbc_gto.Cell()
    cell.atom = "H -0.25 -0.25 -0.25\nH  0.25  0.25  0.25"
    cell.a = np.array([[4.0655, 0.0, 0.0],
                       [0.0, 4.0655, 0.0],
                       [0.0, 0.0, 4.0655]])
    cell.basis = "gth-dzvp-molopt-sr"
    cell.pseudo = "gth-pbe"
    cell.unit = "Angstrom"
    cell.verbose = 0
    cell.build()

    kpts = cell.make_kpts([nk, nk, nk])
    kstruct = build_q_struct(cell, kpts, space_symm=space_symm, tr_symm=tr_symm)
    return cell, kstruct


def _per_k_overlap(cell, kstruct):
    """Return S(k) at every BZ k-point of kstruct."""
    return cell.pbc_intor("int1e_ovlp", kpts=kstruct.kpts)


@pytest.fixture(scope="module")
def h2_setup():
    cell, kstruct = _build_h2_cell_and_kstruct(nk=3, space_symm=True, tr_symm=True)
    S_bz = np.asarray(_per_k_overlap(cell, kstruct))
    S_ibz = S_bz[kstruct.ibz2bz]
    return {
        "cell": cell,
        "kstruct": kstruct,
        "S_bz": S_bz,
        "S_ibz": S_ibz,
    }


def test_lowdin_kspace_representation_contract(h2_setup):
    """X(k) S(k) X(k)† = I at every BZ point, X built only at IBZ."""
    cell = h2_setup["cell"]
    kstruct = h2_setup["kstruct"]
    S_bz = h2_setup["S_bz"]
    S_ibz = h2_setup["S_ibz"]

    X_k, X_inv_k = build_X_kspace(
        "lowdin", kstruct, cell, S_ibz
    )

    nk = kstruct.nkpts
    assert X_k.shape[0] == nk
    assert X_inv_k.shape[0] == nk

    n_ortho = X_k.shape[1]
    eye = np.eye(n_ortho, dtype=np.complex128)

    for ik in range(nk):
        prod = X_k[ik] @ S_bz[ik] @ X_k[ik].conj().T
        np.testing.assert_allclose(
            prod, eye, atol=1e-9, rtol=0,
            err_msg=f"Löwdin orthogonality fails at BZ k={ik}",
        )
        prod2 = X_k[ik] @ X_inv_k[ik]
        np.testing.assert_allclose(
            prod2, eye, atol=1e-9, rtol=0,
            err_msg=f"X @ X_inv != I at BZ k={ik}",
        )


def test_kspace_propagation_uses_kstruct_ordering(h2_setup):
    """build_X_kspace output ordering matches kstruct.kpts order one-to-one."""
    cell = h2_setup["cell"]
    kstruct = h2_setup["kstruct"]
    S_ibz = h2_setup["S_ibz"]

    X_k, _ = build_X_kspace("lowdin", kstruct, cell, S_ibz)

    # At each IBZ representative position, X should equal the per-k Löwdin
    # result on S_ibz directly (no rotation, because the IBZ point is in
    # its own little group's identity slot).
    for i_ir, ik_ir in enumerate(kstruct.ibz2bz):
        x_ref, _ = ortho_utils.lowdin_per_k(S_ibz[i_ir])
        # Gauge: at IBZ rep, stars_ops should map the rep to itself via U=I
        # (modulo phase). Compare X X† to identify with the reference up to
        # a unitary on the right — but for the identity op specifically
        # they must match bitwise via the formula X(k_ir) = X_ir @ I.
        np.testing.assert_allclose(
            X_k[ik_ir], x_ref, atol=1e-9, rtol=0,
            err_msg=(
                f"At IBZ rep BZ-index {ik_ir} (ibz idx {i_ir}), "
                "build_X_kspace does not return the per-k Löwdin result"
            ),
        )


def test_lowdin_kspace_orthogonalizes_fock(h2_setup):
    """X(k) F(k) X_inv(k) is gauge-invariant across the star and Hermitian."""
    cell = h2_setup["cell"]
    kstruct = h2_setup["kstruct"]
    S_bz = h2_setup["S_bz"]
    S_ibz = h2_setup["S_ibz"]
    rng = np.random.default_rng(20260610)

    # Build a symmetry-respecting Fock by propagating a random IBZ Fock to
    # the full BZ via the same representations build_X_kspace uses.
    from green_mbtools.mint.symmetry_utils import get_representation

    nao = cell.nao_nr()
    F_ibz = np.zeros((len(kstruct.ibz2bz), nao, nao), dtype=np.complex128)
    for i_ir in range(len(kstruct.ibz2bz)):
        A = rng.standard_normal((nao, nao)) + 1j * rng.standard_normal((nao, nao))
        F_ibz[i_ir] = 0.5 * (A + A.conj().T)

    F_bz = np.zeros((kstruct.nkpts, nao, nao), dtype=np.complex128)
    bz2ibz_bz = kstruct.ibz2bz[kstruct.bz2ibz]
    ir_pos = {int(k): i for i, k in enumerate(kstruct.ibz2bz)}
    for ik in range(kstruct.nkpts):
        ik_ir = bz2ibz_bz[ik]
        u = get_representation(ik, kstruct.stars_ops_bz[ik], cell, kstruct)
        F_bz[ik] = u @ F_ibz[ir_pos[int(ik_ir)]] @ u.conj().T

    X_k, X_inv_k = build_X_kspace("lowdin", kstruct, cell, S_ibz)

    # X F X_inv must equal the IBZ-frame quantity X_ir F_ir X_inv_ir for all
    # star members.
    F_ortho_ref = {}
    for i_ir, ik_ir in enumerate(kstruct.ibz2bz):
        F_ortho_ref[int(ik_ir)] = X_k[ik_ir] @ F_bz[ik_ir] @ X_inv_k[ik_ir]

    for ik in range(kstruct.nkpts):
        ik_ir = int(bz2ibz_bz[ik])
        prod = X_k[ik] @ F_bz[ik] @ X_inv_k[ik]
        np.testing.assert_allclose(
            prod, F_ortho_ref[ik_ir], atol=1e-9, rtol=0,
            err_msg=f"Gauge invariance broken at BZ k={ik} (ibz rep {ik_ir})",
        )


def test_natural_per_k_fock_tiebreak_deterministic_in_degenerate_block():
    """NO tie-breaker yields a deterministic basis when occupations degenerate."""
    n = 6
    rng = np.random.default_rng(42)

    # S = I keeps the generalized eigenproblem in standard form so the
    # degeneracy structure of dm is straightforward to control.
    S = np.eye(n, dtype=np.complex128)

    n_occ = np.array([2.0, 2.0, 1.5, 1.0, 0.5, 0.1])
    V0 = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    V0, _ = np.linalg.qr(V0)
    dm = V0 @ np.diag(n_occ) @ V0.conj().T
    dm = 0.5 * (dm + dm.conj().T)

    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    F = 0.5 * (A + A.conj().T)

    X1, X_inv1 = _natural_per_k_with_fock_tiebreak(S, dm, F)
    X2, X_inv2 = _natural_per_k_with_fock_tiebreak(S, dm, F)

    np.testing.assert_allclose(X1, X2, atol=0, rtol=0,
        err_msg="NO tie-breaker is not bitwise deterministic")
    np.testing.assert_allclose(X_inv1, X_inv2, atol=0, rtol=0)

    # X X_inv = I
    np.testing.assert_allclose(X1 @ X_inv1, np.eye(n), atol=1e-10)

    # Columns of X_inv are S-orthonormal: X_inv† S X_inv = I (with S=I)
    np.testing.assert_allclose(
        X_inv1.conj().T @ S @ X_inv1, np.eye(n), atol=1e-9,
    )

    # Within the originally-degenerate block (occupations both = 2.0), the
    # Fock projected onto the block must be diagonal in the new basis.
    # LA.eigh returns eigenvalues in ascending order, so the two
    # max-occupation columns sit at the end of V; X_inv = V† so the
    # corresponding *rows* of X_inv are the conjugated natural orbitals.
    deg_rows = [n - 2, n - 1]
    V_B = X_inv1[deg_rows, :].conj().T   # (n, 2) natural orbitals
    FB = V_B.conj().T @ F @ V_B
    off = FB - np.diag(np.diag(FB))
    assert np.max(np.abs(off)) < 1e-8, (
        f"Fock not diagonalized within degenerate block: |off|={np.max(np.abs(off))}"
    )


def test_natural_mode_requires_dm_and_F(h2_setup):
    """build_X_kspace mode='natural' raises without dm_ibz or F_ibz."""
    cell = h2_setup["cell"]
    kstruct = h2_setup["kstruct"]
    S_ibz = h2_setup["S_ibz"]

    with pytest.raises(ValueError):
        build_X_kspace("natural", kstruct, cell, S_ibz)
    with pytest.raises(ValueError):
        build_X_kspace("natural", kstruct, cell, S_ibz, dm_ibz=S_ibz)


def test_mo_mode_requires_C_or_F(h2_setup):
    """build_X_kspace mode='mo' raises without mo_coeff_ibz or F_ibz."""
    cell = h2_setup["cell"]
    kstruct = h2_setup["kstruct"]
    S_ibz = h2_setup["S_ibz"]

    with pytest.raises(ValueError):
        build_X_kspace("mo", kstruct, cell, S_ibz)


def test_unknown_mode_raises(h2_setup):
    cell = h2_setup["cell"]
    kstruct = h2_setup["kstruct"]
    S_ibz = h2_setup["S_ibz"]
    with pytest.raises(ValueError):
        build_X_kspace("bogus", kstruct, cell, S_ibz)


def test_spinor_only_with_lowdin(h2_setup):
    cell = h2_setup["cell"]
    kstruct = h2_setup["kstruct"]
    S_ibz = h2_setup["S_ibz"]
    with pytest.raises(NotImplementedError):
        build_X_kspace("mo", kstruct, cell, S_ibz, spinor=True,
                       mo_coeff_ibz=np.zeros_like(S_ibz))


def test_from_ao_reps_matches_kstruct_path(h2_setup):
    """build_X_kspace_from_ao_reps with reps extracted from kstruct
    reproduces build_X_kspace output bit-for-bit."""
    from green_mbtools.mint.symmetry_utils import get_representation

    cell = h2_setup["cell"]
    kstruct = h2_setup["kstruct"]
    S_ibz = h2_setup["S_ibz"]

    # Reference: kstruct-driven path
    X_ref, X_inv_ref = build_X_kspace("lowdin", kstruct, cell, S_ibz)

    # Precompute AO reps the way store_kstruct_ops_info does for non-X2C.
    nk = kstruct.nkpts
    nao = cell.nao_nr()
    k_sym = np.zeros((nk, nao, nao), dtype=np.complex128)
    for ik in range(nk):
        k_sym[ik] = get_representation(
            ik, kstruct.stars_ops_bz[ik], cell, kstruct
        )
    tr_conj = np.asarray(kstruct.time_reversal_symm_bz, dtype=bool)

    X_alt, X_inv_alt = build_X_kspace_from_ao_reps(
        "lowdin", S_ibz, kstruct.ibz2bz, kstruct.bz2ibz, k_sym,
        tr_conj=tr_conj,
    )

    np.testing.assert_allclose(X_alt, X_ref, atol=1e-12, rtol=0)
    np.testing.assert_allclose(X_inv_alt, X_inv_ref, atol=1e-12, rtol=0)


def test_from_ao_reps_orthogonalizes_S(h2_setup):
    """build_X_kspace_from_ao_reps output satisfies X(k) S(k) X(k)† = I."""
    from green_mbtools.mint.symmetry_utils import get_representation

    cell = h2_setup["cell"]
    kstruct = h2_setup["kstruct"]
    S_ibz = h2_setup["S_ibz"]
    S_bz = h2_setup["S_bz"]

    nk = kstruct.nkpts
    nao = cell.nao_nr()
    k_sym = np.zeros((nk, nao, nao), dtype=np.complex128)
    for ik in range(nk):
        k_sym[ik] = get_representation(
            ik, kstruct.stars_ops_bz[ik], cell, kstruct
        )
    tr_conj = np.asarray(kstruct.time_reversal_symm_bz, dtype=bool)

    X_k, _ = build_X_kspace_from_ao_reps(
        "lowdin", S_ibz, kstruct.ibz2bz, kstruct.bz2ibz, k_sym,
        tr_conj=tr_conj,
    )

    n_ortho = X_k.shape[1]
    eye = np.eye(n_ortho, dtype=np.complex128)
    for ik in range(nk):
        np.testing.assert_allclose(
            X_k[ik] @ S_bz[ik] @ X_k[ik].conj().T, eye, atol=1e-9,
            err_msg=f"X S X† != I at BZ k={ik} (from_ao_reps path)",
        )


def test_from_ao_reps_natural_requires_inputs(h2_setup):
    kstruct = h2_setup["kstruct"]
    S_ibz = h2_setup["S_ibz"]
    nk = kstruct.nkpts
    n = S_ibz.shape[1]
    fake_reps = np.tile(np.eye(n, dtype=np.complex128), (nk, 1, 1))
    with pytest.raises(ValueError):
        build_X_kspace_from_ao_reps(
            "natural", S_ibz, kstruct.ibz2bz, kstruct.bz2ibz, fake_reps
        )


@pytest.fixture(scope="module")
def h2_tr_only_setup():
    """H2 cubic cell with TR-only k-reduction (space_symm=False).

    Built specifically so that BZ stars are pure TR pairs ``(k, -k)`` and
    every TR partner has ``time_reversal_symm_bz[ik] = True``. The
    centrosymmetric default fixture sweeps TR into spatial inversion and
    never exercises the TR-conjugation branch of build_X_kspace_from_ao_reps.
    """
    cell, kstruct = _build_h2_cell_and_kstruct(
        nk=3, space_symm=False, tr_symm=True
    )
    S_bz = np.asarray(_per_k_overlap(cell, kstruct))
    S_ibz = S_bz[kstruct.ibz2bz]
    return {"cell": cell, "kstruct": kstruct, "S_bz": S_bz, "S_ibz": S_ibz}


def _tr_partner_indices(cell, kstruct):
    """Return ``tr_of[ik]`` = BZ index of the TR partner of BZ point ``ik``."""
    scaled = cell.get_scaled_kpts(kstruct.kpts)
    nk = scaled.shape[0]
    tr_of = np.full(nk, -1, dtype=int)
    for ik in range(nk):
        target = -scaled[ik]
        for jk in range(nk):
            diff = scaled[jk] - target
            diff -= np.round(diff)
            if np.max(np.abs(diff)) < 1e-8:
                tr_of[ik] = jk
                break
    assert np.all(tr_of >= 0), "Could not locate TR partner for every k-point"
    return tr_of


def test_itransform_tr_identity_with_new_X(h2_tr_only_setup):
    """Regression: the V-rotation TR identity that itransform.cpp's
    ``conj_kpair_list`` shortcut relies on holds under the new
    full-BZ X built by build_X_kspace_from_ao_reps + legacy-storage shim.

    For every TR-paired kpair, with V_AO satisfying
    ``V_AO(-ki, -kj) = V_AO(ki, kj).conj()`` (the real-AO inversion rule),
    requires the orthogonalized V to satisfy
    ``V_ortho(-ki, -kj) = V_ortho(ki, kj).conj()`` after applying
    ``X_legacy(ki).adjoint() @ V_AO @ X_legacy(kj)``.

    Without this property, itransform.cpp would silently produce the
    wrong V_ortho at TR-paired kpairs.
    """
    from green_mbtools.mint.symmetry_utils import get_representation
    from green_mbtools.mint.ortho_utils import build_X_kspace_from_ao_reps

    cell = h2_tr_only_setup["cell"]
    kstruct = h2_tr_only_setup["kstruct"]
    S_ibz = h2_tr_only_setup["S_ibz"]

    # Sanity check that the fixture actually exercises TR: at least one
    # BZ point must have time_reversal_symm_bz=True.
    tr_conj_bz = np.asarray(kstruct.time_reversal_symm_bz, dtype=bool)
    assert tr_conj_bz.any(), (
        "h2_tr_only fixture didn't produce any TR-paired k-points — "
        "test would not exercise the TR branch"
    )

    # Precompute AO reps and run the new path.
    nk = kstruct.nkpts
    nao = cell.nao_nr()
    k_sym_ao = np.zeros((nk, nao, nao), dtype=np.complex128)
    for ik in range(nk):
        k_sym_ao[ik] = get_representation(
            ik, kstruct.stars_ops_bz[ik], cell, kstruct
        )
    X_k_new, X_inv_k_new = build_X_kspace_from_ao_reps(
        "lowdin", S_ibz, kstruct.ibz2bz, kstruct.bz2ibz, k_sym_ao,
        tr_conj=tr_conj_bz,
    )

    # Apply the same legacy-storage shim init_seet.py uses before
    # writing to transform.h5.
    X_legacy = X_k_new.conj().swapaxes(1, 2)

    # Build a synthetic V_AO(k, k') over the full BZ that respects the
    # real-AO inversion identity V_AO(-k, -k') = V_AO(k, k').conj().
    # The simplest construction: pick V_AO independently at each
    # (ki, kj) where ki <= kj in some ordering that contains no TR
    # partners, then fill the conjugate-partner entries.
    rng = np.random.default_rng(20260611)
    V_AO = np.zeros((nk, nk, nao, nao), dtype=np.complex128)
    tr_of = _tr_partner_indices(cell, kstruct)
    assigned = np.zeros((nk, nk), dtype=bool)
    for ki in range(nk):
        for kj in range(nk):
            if assigned[ki, kj]:
                continue
            kti = tr_of[ki]
            ktj = tr_of[kj]
            if (kti, ktj) == (ki, kj):
                # Self-TR-paired kpair: identity forces V to be real.
                A = rng.standard_normal((nao, nao)).astype(np.complex128)
            else:
                A = (rng.standard_normal((nao, nao))
                     + 1j * rng.standard_normal((nao, nao)))
                V_AO[kti, ktj] = A.conj()
                assigned[kti, ktj] = True
            V_AO[ki, kj] = A
            assigned[ki, kj] = True

    # Verify the V_AO identity we constructed.
    for ki in range(nk):
        for kj in range(nk):
            np.testing.assert_allclose(
                V_AO[tr_of[ki], tr_of[kj]], V_AO[ki, kj].conj(),
                atol=1e-12,
                err_msg=f"V_AO TR identity broken at constructed ({ki},{kj})",
            )

    # The actual check: V_ortho satisfies the same TR identity, which
    # is exactly what itransform.cpp's conj_kpair_list reduction
    # assumes.
    for ki in range(nk):
        for kj in range(nk):
            kti = tr_of[ki]
            ktj = tr_of[kj]
            V_ortho_ij = (X_legacy[ki].conj().T @ V_AO[ki, kj]
                          @ X_legacy[kj])
            V_ortho_tr = (X_legacy[kti].conj().T @ V_AO[kti, ktj]
                          @ X_legacy[ktj])
            np.testing.assert_allclose(
                V_ortho_tr, V_ortho_ij.conj(), atol=1e-9, rtol=0,
                err_msg=(
                    f"V_ortho TR identity broken at kpair ({ki},{kj}) "
                    f"vs TR partner ({kti},{ktj}). "
                    "itransform.cpp's conj_kpair_list shortcut would "
                    "produce wrong V_ortho here."
                ),
            )


def test_ar_x2c_spinor_orthogonality():
    """Lowdin in the X2C double group orthogonalizes S(k) at every BZ point."""
    data_file = Path(__file__).parent / "test_data" / "Ar_x2c" / "input_full_symm.h5"
    if not data_file.exists():
        pytest.skip("Ar X2C reference data not available")

    with h5py.File(data_file, "r") as f:
        S_raw = f["HF/S-k"][()]
        bz2ibz = f["symmetry/k/bz2ibz"][()]
        nk = int(f["symmetry/k/nk"][()])
    S_bz = S_raw.view(complex).reshape(S_raw.shape[:-1])
    # (ns, nk, nso, nso) -> use spin 0
    if S_bz.ndim == 4:
        S_bz = S_bz[0]
    assert S_bz.shape[0] == nk

    # Build kstruct + cell for Ar X2C from the input file's params (the cell
    # must match the one used to generate the reference); if those params are
    # not in the file we skip the spinor end-to-end test.
    pytest.skip(
        "Spinor end-to-end test needs the original Ar cell parameters; "
        "covered by the AO-rep test in symmetry_test.py for now."
    )
