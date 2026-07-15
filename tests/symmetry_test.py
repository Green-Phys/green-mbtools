import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from green_mbtools.mint import common_utils as comm
from green_mbtools.mint import integral_utils as int_utils
from green_mbtools.mint import pyscf_pbc_init


_ATOM = "H -0.25 -0.25 -0.25\nH  0.25  0.25  0.25"
_LATTICE = "4.0655, 0.0,    0.0\n0.0,    4.0655, 0.0\n0.0,    0.0,    4.0655\n"
_BASIS = "gth-dzvp-molopt-sr"
_PSEUDO = "gth-pbe"


class _DummyCell:
    """Minimal cell-like object required by decomposition helpers."""

    def __init__(self, dimension=3, low_dim_ft_type="inf_vacuum"):
        self.dimension = dimension
        self.low_dim_ft_type = low_dim_ft_type


def _load_reference_hf_data():
    """Load reference HF matrices for the H2 periodic test case."""
    ref_file = Path(__file__).parent / "test_data" / "H2_pbc" / "UHF" / "input.h5"
    with h5py.File(ref_file, "r") as fref:
        fock = fref["HF/Fock-k"][()].view(complex)
        overlap = fref["HF/S-k"][()].view(complex)
        hcore = fref["HF/H-k"][()].view(complex)
    return (
        fock.reshape(fock.shape[:-1]),
        overlap.reshape(overlap.shape[:-1]),
        hcore.reshape(hcore.shape[:-1]),
    )


def _run_grid_only_case(run_dir: Path, space_symm: bool, tr_symm: bool, nk: int = 3, x2c: int = 0):
    """Run one grid-only generation and return output and cderi paths."""
    run_dir.mkdir(parents=True, exist_ok=True)
    old_cwd = Path.cwd()
    output_h5 = run_dir / "input.h5"
    os.chdir(run_dir)
    try:
        params = [
            "--atom", _ATOM,
            "--a", _LATTICE,
            "--basis", _BASIS,
            "--pseudo", _PSEUDO,
            "--output_path", str(output_h5),
            "--df_int", "0",
            "--nk", str(nk),
            "--grid_only", "true",
            "--keep_cderi", "true",
            "--space_symm", "true" if space_symm else "false",
            "--tr_symm", "true" if tr_symm else "false",
            "--x2c", str(x2c),
        ]
        args = comm.init_pbc_params(params=params)
        pyscf_init = pyscf_pbc_init(args)
        pyscf_init.mean_field_input()
    finally:
        os.chdir(old_cwd)
    return output_h5, run_dir / "cderi.h5"


def _read_j2c_by_numeric_key(cderi_path: Path):
    """Read j2c matrices keyed by their integer-like dataset name."""
    matrices = {}
    with h5py.File(cderi_path, "r") as f:
        j2c_grp = f["j2c"]
        for key in j2c_grp.keys():
            if key.isdigit():
                matrices[int(key)] = j2c_grp[key][...]
    return matrices


@pytest.fixture(scope="module")
def generated_cases(tmp_path_factory):
    """Generate one symmetric and one full-BZ case for reuse across tests."""
    base = tmp_path_factory.mktemp("symmetry_cases")
    symm_output, symm_cderi = _run_grid_only_case(base / "space_and_tr_symm_true", space_symm=True, tr_symm=True, nk=3)
    trs_output, trs_cderi = _run_grid_only_case(base / "tr_symm_true", space_symm=False, tr_symm=True, nk=3)
    full_output, full_cderi = _run_grid_only_case(base / "symm_false", space_symm=False, tr_symm=False, nk=3)
    return {
        "symm_output": symm_output,
        "symm_cderi": symm_cderi,
        "trs_output": trs_output,
        "trs_cderi": trs_cderi,
        "full_output" : full_output,
        "full_cderi" : full_cderi
    }


def test_j2c_cholesky_and_eigh_decomposition():
    """Validate algebraic consistency of Cholesky and eigenvalue j2c decompositions."""
    rng = np.random.default_rng(7)
    a = rng.standard_normal((6, 6)) + 1j * rng.standard_normal((6, 6))
    j2c = a @ a.conj().T + 1e-3 * np.eye(6)
    cell = _DummyCell()

    # Cholesky decomposition and inverse
    lmat, neg = int_utils.cholesky_decomposed_metric(j2c, cell, inv=False)
    assert neg is None
    np.testing.assert_allclose(lmat @ lmat.conj().T, j2c, atol=1e-10, rtol=1e-10)

    lmat_inv, neg_inv = int_utils.cholesky_decomposed_metric(j2c, cell, inv=True)
    assert neg_inv is None
    np.testing.assert_allclose(lmat_inv @ lmat, np.eye(lmat.shape[0]), atol=1e-10, rtol=1e-10)

    # Eigenvalue decomposition and inverse
    emat, neg_e = int_utils.eigenvalue_decomposed_metric(j2c, cell, inv=False)
    assert neg_e is None
    np.testing.assert_allclose(emat @ emat.conj().T, j2c, atol=1e-8, rtol=1e-8)

    emat_inv, neg_e_inv = int_utils.eigenvalue_decomposed_metric(j2c, cell, inv=True)
    assert neg_e_inv is None
    np.testing.assert_allclose(emat_inv @ j2c @ emat_inv.conj().T, np.eye(emat_inv.shape[0]), atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize(
    "case_key",
    ["symm_output", "trs_output", "full_output"],
)
def test_symmetry_on_ao_basis(generated_cases, case_key):
    """Check AO-space transformation against reference HF matrices for all generated cases."""
    output_h5 = generated_cases[case_key]
    fock, overlap, hcore = _load_reference_hf_data()

    with h5py.File(output_h5, "r") as fout:
        nk = fout["symmetry/k/nk"][()]
        ink = fout["symmetry/k/ink"][()]
        bz_to_ibz_index = fout["symmetry/k/bz2ibz"][()]
        kspace_orep = fout["symmetry/k/k_sym_transform_ao"][()]
        conj_list = fout["symmetry/k/tr_conj"][()]
    assert nk == overlap.shape[1]
    assert len(bz_to_ibz_index) == nk

    if case_key == "symm_output":
        assert ink == 6  # known/expected value
    elif case_key == "trs_output":
        assert ink == 14  # known/expected value
    else:
        assert ink == nk

    for ik in range(nk):
        ibz = bz_to_ibz_index[ik]
        uop = kspace_orep[ik]
        do_conj = int(conj_list[ik]) != 0

        overlap_recon = uop @ overlap[0, ibz] @ uop.conj().T
        if do_conj:
            overlap_recon = overlap_recon.conjugate()
        np.testing.assert_allclose(overlap_recon, overlap[0, ik], atol=1e-8, rtol=1e-8)

        hcore_recon = uop @ hcore[0, ibz] @ uop.conj().T
        if do_conj:
            hcore_recon = hcore_recon.conjugate()
        np.testing.assert_allclose(hcore_recon, hcore[0, ik], atol=1e-8, rtol=1e-8)

        # Vxc is built on a real-space grid, so Fock symmetrization is looser than H/S.
        fock_recon = uop @ fock[0, ibz] @ uop.conj().T
        if do_conj:
            fock_recon = fock_recon.conjugate()
        np.testing.assert_allclose(fock_recon, fock[0, ik], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "symm_case_key,symm_output_key",
    [
        ("symm_cderi", "symm_output"),
        ("trs_cderi", "trs_output"),
    ],
)
def test_j2c_ibz_to_full_bz_transformation(generated_cases, symm_case_key, symm_output_key):
    """Validate j2c transfer from each reduced case to full BZ via stored operators."""
    symm_j2c = _read_j2c_by_numeric_key(generated_cases[symm_case_key])
    full_j2c = _read_j2c_by_numeric_key(generated_cases["full_cderi"])

    with h5py.File(generated_cases[symm_output_key], "r") as fs:
        index = fs["symmetry/q/bz2ibz"][()]
        conj_list = fs["symmetry/q/tr_conj"][()]
        kspace_orep_j2c = fs["symmetry/q/k_sym_transform_j2c"][()]

    ncomp = 0
    for ik, ir_k_ibz in enumerate(index):
        ik = int(ik)
        ir_k_ibz = int(ir_k_ibz)
        if ik not in full_j2c or ir_k_ibz not in symm_j2c:
            continue
        uop = kspace_orep_j2c[ik]
        j2c_recon = uop @ symm_j2c[ir_k_ibz] @ uop.conj().T
        if conj_list[ik] != 0:
            j2c_recon = j2c_recon.conj()
        np.testing.assert_allclose(j2c_recon, full_j2c[ik], atol=1e-6, rtol=1e-6)
        ncomp += 1

    assert ncomp > 0, "No overlapping j2c keys found for IBZ->BZ transformation check"


def test_nk_list_stored_in_hdf5(generated_cases):
    """Verify params/nk_list = [nkx, nky, nkz] is written and consistent with params/nk."""
    for key in ("symm_output", "trs_output", "full_output"):
        with h5py.File(generated_cases[key], "r") as f:
            assert "symmetry/k/nk_list" in f, f"{key}: symmetry/k/nk_list missing from HDF5"
            nk_list = f["symmetry/k/nk_list"][()]
            nk      = int(f["params/nk"][()])

        assert nk_list.shape == (3,), f"{key}: nk_list shape {nk_list.shape} != (3,)"
        assert np.prod(nk_list) == nk, (
            f"{key}: prod(nk_list)={np.prod(nk_list)} != params/nk={nk}"
        )
        # All three generated cases use nk=3, so the grid must be 3x3x3.
        np.testing.assert_array_equal(nk_list, [3, 3, 3])


def test_x2c_tr_sym_transforms(tmp_path):
    """X2C1e k-space symmetry operators for TR-only and full double-group cases.

    TR-only (space_symm=False):
      - Non-TR k-points store I_nso; TR k-points store Theta = kron([[0,1],[-1,0]], I_nao).

    Double-group (space_symm=True):
      - Produces a strictly smaller IBZ than TR-only (space-group reduction active).
      - Every k_sym_transform_ao matrix is unitary (spinor representation of a symmetry op).
    """
    out_space_true, _ = _run_grid_only_case(
        tmp_path / "x2c_space_true",
        space_symm=True,
        tr_symm=True,
        nk=3,
        x2c=2,
    )
    out_space_false, _ = _run_grid_only_case(
        tmp_path / "x2c_space_false",
        space_symm=False,
        tr_symm=True,
        nk=3,
        x2c=2,
    )

    with h5py.File(out_space_true, "r") as f_true, h5py.File(out_space_false, "r") as f_false:
        nk_true = int(f_true["symmetry/k/nk"][()])
        nk_false = int(f_false["symmetry/k/nk"][()])
        ink_true = int(f_true["symmetry/k/ink"][()])
        ink_false = int(f_false["symmetry/k/ink"][()])
        conj_false = f_false["symmetry/k/tr_conj"][()]
        kops_true = f_true["symmetry/k/k_sym_transform_ao"][()]
        kops_false = f_false["symmetry/k/k_sym_transform_ao"][()]

    # Both runs cover the same full BZ.
    assert nk_true == nk_false

    nso = kops_true.shape[1]
    nao = nso // 2
    nso_eye = np.eye(nso, dtype=np.complex128)
    theta = np.kron(np.array([[0, 1], [-1, 0]], dtype=np.complex128), np.eye(nao))

    # --- TR-only case --------------------------------------------------------
    # Non-TR k-points get I_nso; TR k-points get Theta.
    expected_false = np.where(conj_false[:, None, None], theta, nso_eye)
    np.testing.assert_allclose(kops_false, expected_false, atol=1e-12, rtol=0.0)

    # --- Double-group case ---------------------------------------------------
    # Space-group reduction must give a strictly smaller IBZ than TR-only.
    assert ink_true < ink_false

    # Every stored operator must be unitary: U U† = I.
    for ik in range(nk_true):
        u = kops_true[ik]
        np.testing.assert_allclose(
            u @ u.conj().T, nso_eye, atol=1e-10, rtol=0.0,
            err_msg=f"k_sym_transform_ao[{ik}] is not unitary"
        )


def test_x2c_fock_ibz_to_full_bz():
    """Verify that X2C Fock at IBZ k-points reconstructs full BZ via k_sym_transform_ao.

    Uses a precomputed cubic Ar (def2-svp, --x2c 2) calculation with full
    space-group + time-reversal symmetry (3×3×1 k-mesh: nk=9, ink=3,
    nao=14, nso=28).  Checks the general reconstruction for all k-points:

        F(k) = U_k @ F(k_ibz) @ U_k†          [non-TR k-points]
        F(k) = (U_k @ F(k_ibz) @ U_k†).conj() [TR k-points]

    where U_k is the full nso×nso spinor rotation from k_sym_transform_ao.
    atol=1e-6 is tight enough to catch a wrong k_sym_transform_ao (all-identity
    produces O(1) errors) while accommodating floating-point residuals.
    """
    data_file = Path(__file__).parent / "test_data" / "Ar_x2c" / "input_full_symm.h5"

    with h5py.File(data_file, "r") as f:
        fock_raw = f["HF/Fock-k"][()]
        bz2ibz   = f["symmetry/k/bz2ibz"][()]
        k_sym_op = f["symmetry/k/k_sym_transform_ao"][()]
        tr_conj  = f["symmetry/k/tr_conj"][()]
        nk       = int(f["symmetry/k/nk"][()])

    # (ns, nk, nao, nao, 2) float64 → (ns, nk, nao, nao) complex128
    fock = fock_raw.view(complex).reshape(fock_raw.shape[:-1])

    assert fock.shape[1] == nk
    assert len(bz2ibz) == nk

    for ik in range(nk):
        ibz_k  = bz2ibz[ik]           # full-BZ index of the IBZ representative for k
        U      = k_sym_op[ik]         # nso x nso unitary rotation
        F_ibz  = fock[0, ibz_k]

        F_recon = U @ F_ibz @ U.conj().T
        if tr_conj[ik]:
            F_recon = F_recon.conj()

        np.testing.assert_allclose(
            F_recon, fock[0, ik], atol=1e-6, rtol=0,
            err_msg=f"Fock reconstruction failed at full-BZ k={ik} (IBZ representative={ibz_k})"
        )


@pytest.mark.skip(reason="TODO: validate k_sym_transform_p0 transformation against an independent real-data reference")
def test_k_sym_transform_p0_matches_metric_basis_transform(generated_cases):
    """TODO: replace implementation-coupled check with a real-data validation."""
    pass


def test_ao_rep_bloch_phase_on_supercell():
    """AO symmetry operators must obey S(k)=U_k S(k_ir) U_k† for a supercell.

    Regression for the Bloch-phase bug in ``get_representation``: the phase
    factor was built from atom coordinates folded into [-0.5, 0.5) instead of
    the as-input positions the Bloch integrals use, so any atom with input
    fractional coordinate >= 0.5 acquired a spurious e^{i2*pi*k.L} factor.

    A primitive cell (all atoms in [0, 0.5), e.g. the H2 case above) never
    exercises the fold and always passed. This uses a rock-salt LiH cell
    doubled along [111] (2 formula units): atoms sit at fractional 1/2 and 3/4,
    with inequivalent Li/H sublattices, which is what exposes the wrong
    relative phase between atoms. Broken operators give O(1) residuals here.
    """
    from pyscf.pbc import gto
    from pyscf.pbc.lib import kpts as libkpts

    from green_mbtools.mint.symmetry_utils import get_representation

    a = 4.0
    amat = np.array([[a, a / 2, a / 2], [a / 2, a, a / 2], [a / 2, a / 2, a]])
    # rock-salt LiH, doubled along [111]; shift 0.03 off the exact-0.5 boundary
    # so the origin never lands an atom on the fold discontinuity.
    frac = np.array([[0, 0, 0], [0.25] * 3, [0.5] * 3, [0.75] * 3]) + 0.03
    cart = frac @ amat

    cell = gto.Cell()
    cell.a = amat.tolist()
    cell.atom = [["Li", cart[0]], ["H", cart[1]], ["Li", cart[2]], ["H", cart[3]]]
    cell.basis = "gth-szv"
    cell.pseudo = "gth-pbe"
    cell.verbose = 0
    cell.space_group_symmetry = True
    cell.symmorphic = False
    cell.build()

    ks = libkpts.make_kpts(
        cell, cell.make_kpts([4, 4, 4]),
        space_group_symmetry=True, time_reversal_symmetry=True,
    )
    # The supercell must actually reduce, else the round-trip is vacuous.
    assert ks.nkpts_ibz < ks.nkpts

    overlap = np.asarray(cell.pbc_intor("int1e_ovlp", kpts=ks.kpts))
    ibz_of = ks.ibz2bz[ks.bz2ibz]          # full-BZ index of each k's IBZ rep
    tr = ks.time_reversal_symm_bz

    for ik in range(ks.nkpts):
        uop = get_representation(ik, ks.stars_ops_bz[ik], cell, ks)
        recon = uop @ overlap[ibz_of[ik]] @ uop.conj().T
        if tr[ik]:
            recon = recon.conj()
        np.testing.assert_allclose(
            recon, overlap[ik], atol=1e-9, rtol=0,
            err_msg=f"S(k) not reconstructed from IBZ at k={ik} "
                    f"(IBZ rep={ibz_of[ik]}); k_sym_transform_ao Bloch phase is wrong",
        )


def test_ao_rep_time_reversal_phase():
    """AO symmetry operators must obey S(k)=U_k S(k_ir) U_k† at time-reversal-
    paired k-points that carry a *complex* Bloch phase.

    Regression for the time-reversal phase bug in ``get_representation``. For a
    TR-paired k, the spatial operation lands on -k and the reconstruction applies
    a conjugation, so the Bloch phase must be evaluated at -k (not +k). Cubic
    supercells (NiO/LiH, test above) only have *real* phases at their TR points,
    where +k == -k, so they cannot catch this. Hexagonal hBN on a Gamma-centered
    6x6x1 mesh folds heavily via time reversal with complex phases (e^{±i pi/3}),
    which exposes it: the buggy code gives O(1) residuals on the inter-atom
    blocks.
    """
    from pyscf.pbc import gto
    from pyscf.pbc.lib import kpts as libkpts

    from green_mbtools.mint.symmetry_utils import get_representation

    a = 2.5
    amat = np.array([[a, 0.0, 0.0],
                     [-a / 2, a * np.sqrt(3) / 2, 0.0],
                     [0.0, 0.0, 15.0]])
    cell = gto.Cell()
    cell.a = amat
    cell.atom = [["B", (0.0, a / np.sqrt(3), 7.5)],
                 ["N", (a / 2, a / (2 * np.sqrt(3)), 7.5)]]
    cell.basis = "gth-szv"
    cell.pseudo = "gth-pbe"
    cell.verbose = 0
    cell.space_group_symmetry = True
    cell.symmorphic = False
    cell.build()

    ks = libkpts.make_kpts(
        cell, cell.make_kpts([6, 6, 1]),
        space_group_symmetry=True, time_reversal_symmetry=True,
    )
    # The test is only meaningful if the BZ reduction actually uses time reversal.
    assert ks.time_reversal_symm_bz.any()
    assert ks.nkpts_ibz < ks.nkpts

    overlap = np.asarray(cell.pbc_intor("int1e_ovlp", kpts=ks.kpts))
    ibz_of = ks.ibz2bz[ks.bz2ibz]
    tr = ks.time_reversal_symm_bz

    for ik in range(ks.nkpts):
        uop = get_representation(ik, ks.stars_ops_bz[ik], cell, ks)
        recon = uop @ overlap[ibz_of[ik]] @ uop.conj().T
        if tr[ik]:
            recon = recon.conj()
        np.testing.assert_allclose(
            recon, overlap[ik], atol=1e-8, rtol=0,
            err_msg=f"S(k) not reconstructed at time-reversal k={ik} "
                    f"(IBZ rep={ibz_of[ik]}); get_representation must evaluate the "
                    f"Bloch phase at -k for time-reversal-paired points",
        )
