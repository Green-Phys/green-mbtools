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

# AFM pinned H2 dimer geometry (all-electron sto-3g, no pseudo)
_AFM_ATOM = "H -1 -1 -1\nH 1 1 1"
_AFM_LATTICE = "10.0,0.0,0.0\n0.0,10.0,0.0\n0.0,0.0,10.0"
_AFM_BASIS = "sto-3g"


def _write_afm_dm0(run_dir: Path) -> Path:
    """Write an AFM initial-guess density matrix to run_dir/dm0.h5 and return the path.

    For a 2-atom sto-3g cell (nao=2), the AFM guess puts alpha density on H0
    and beta density on H1:
      dm_gamma[0, 0, 0] = 1.0   # alpha on H0
      dm_gamma[1, 1, 1] = 1.0   # beta  on H1
    """
    dm_gamma = np.zeros((2, 2, 2), dtype=complex)
    dm_gamma[0, 0, 0] = 1.0  # alpha density on H0
    dm_gamma[1, 1, 1] = 1.0  # beta  density on H1
    dm0_path = run_dir / "dm0.h5"
    with h5py.File(dm0_path, "w") as f:
        f.create_dataset("dm_gamma", data=dm_gamma)
    return dm0_path


def _run_afm_case(run_dir: Path, *, symm_ops=None, space_symm: bool = True, nk: int = 3) -> Path:
    """Run a full KUHF mean-field for the pinned AFM H2 dimer and return the output input.h5 path.

    Runs a complete (not grid_only) mean-field initialization with an AFM initial
    guess (up on H0, down on H1) so the SCF converges to a spin-polarized solution.
    TR symmetry is disabled because collinear AFM breaks pure time reversal.

    Parameters
    ----------
    run_dir:
        Directory to create and run the calculation in.
    symm_ops:
        If not None, passed as --symm_ops (comma-separated index string, e.g. "0,7,8,9,10,11").
    space_symm:
        Whether to use space-group symmetry (default True).
    nk:
        k-mesh size along each dimension (default 3 -> 3x3x3).

    Returns
    -------
    Path
        Path to the produced input.h5 file.
    """
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    dm0_path = _write_afm_dm0(run_dir)
    old_cwd = Path.cwd()
    output_h5 = run_dir / "input.h5"
    os.chdir(run_dir)
    try:
        params = [
            "--atom", _AFM_ATOM,
            "--a", _AFM_LATTICE,
            "--basis", _AFM_BASIS,
            "--output_path", str(output_h5),
            "--df_int", "0",
            "--nk", str(nk),
            "--grid_only", "false",
            "--keep_cderi", "true",
            "--space_symm", "true" if space_symm else "false",
            "--tr_symm", "false",
            "--x2c", "0",
            "--dm0", str(dm0_path),
        ]
        if symm_ops is not None:
            params += ["--symm_ops", str(symm_ops)]
        args = comm.init_pbc_params(params=params)
        pyscf_init = pyscf_pbc_init(args)
        pyscf_init.mean_field_input()
    finally:
        os.chdir(old_cwd)
    return output_h5


class _DummyCell:
    """Minimal cell-like object required by decomposition helpers."""

    def __init__(self, dimension=3, low_dim_ft_type="inf_vacuum"):
        self.dimension = dimension
        self.low_dim_ft_type = low_dim_ft_type


def _load_reference_hf_data():
    """Load reference HF matrices for the H2 periodic test case."""
    ref_file = Path(__file__).parent / "test_data" / "H2_pbc" / "UHF" / "input.h5"
    with h5py.File(ref_file, "r") as fref:
        fock = fref["HF/Fock-k"][()]
        overlap = fref["HF/S-k"][()]
        hcore = fref["HF/H-k"][()]
    return (
        fock,
        overlap,
        hcore,
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
        fock     = f["HF/Fock-k"][()]
        bz2ibz   = f["symmetry/k/bz2ibz"][()]
        k_sym_op = f["symmetry/k/k_sym_transform_ao"][()]
        tr_conj  = f["symmetry/k/tr_conj"][()]
        nk       = int(f["symmetry/k/nk"][()])

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

    The ``get_representation`` Bloch phase must be built from the as-input atom
    positions the integrals use. This exercises the case where that matters:
    atoms with fractional coordinate >= 0.5, where the folded ([-0.5, 0.5)) and
    as-input positions differ by a lattice vector.

    A primitive cell (all atoms in [0, 0.5), e.g. the H2 case above) does not
    reach that case. This uses a rock-salt LiH cell doubled along [111] (2
    formula units): atoms sit at fractional 1/2 and 3/4, with inequivalent Li/H
    sublattices, so a wrong relative phase between atoms would show up as an
    O(1) residual on the inter-atom blocks.
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


def _afm_cell():
    """Pinned AFM H2 dimer cell for detect_space_group tests.

    10 A cubic box, H at (-1,-1,-1) and (1,1,1) Angstrom, sto-3g all-electron
    basis (no pseudopotential), spin=0, space_group_symmetry=True.
    """
    from pyscf.pbc import gto
    cell = gto.Cell()
    cell.a = np.eye(3) * 10.0
    cell.atom = "H -1 -1 -1\nH 1 1 1"
    cell.basis = "sto-3g"
    cell.spin = 0
    cell.space_group_symmetry = True
    cell.verbose = 0
    cell.build()
    return cell


def _afm_cell_symmorphic():
    """Same geometry as _afm_cell() but built with symmorphic=True.

    Used to obtain PySCF's own symmorphic-only operator list via
    cell.lattice_symmetry.ops for comparison against select_ops "symmorphic" mode.
    """
    from pyscf.pbc import gto
    cell = gto.Cell()
    cell.a = np.eye(3) * 10.0
    cell.atom = "H -1 -1 -1\nH 1 1 1"
    cell.basis = "sto-3g"
    cell.spin = 0
    cell.space_group_symmetry = True
    cell.symmorphic = True
    cell.verbose = 0
    cell.build()
    return cell


def _native_symmorphic_ops(cell):
    """Return PySCF's own symmorphic-only ops from a cell built with symmorphic=True."""
    return cell.lattice_symmetry.ops


def _sublattice_preserving_indices(cell, ops):
    """Return indices of ops whose atom permutation is the identity (sublattice-preserving).

    Uses generate_permutation_info(cell, op)[0] to determine the partner-index
    array; an op is sublattice-preserving if partner[i] == i for every atom i.
    """
    from green_mbtools.mint.symmetry_utils import generate_permutation_info
    indices = []
    for i, op in enumerate(ops):
        partner, _ = generate_permutation_info(cell, op)
        if all(partner[j] == j for j in range(len(partner))):
            indices.append(i)
    return indices


def test_detect_space_group_h2_dimer():
    from green_mbtools.mint import symmetry_utils as su
    cell = _afm_cell()
    info = su.detect_space_group(cell)
    assert len(info.ops) == 12                     # D3d
    assert info.point_group_symbol == "-3m"
    assert info.ops[0].is_eye                       # identity sorts first
    assert any(op.is_inversion for op in info.ops)  # inversion present (swaps sublattices)


def test_detect_space_group_spglib_backend_labels():
    import pytest
    pytest.importorskip("spglib")
    from green_mbtools.mint import symmetry_utils as su
    info = su.detect_space_group(_afm_cell(), backend="spglib")
    assert info.international_number is not None   # spglib populates international labels


def test_ao_rep_time_reversal_phase():
    """AO symmetry operators must obey S(k)=U_k S(k_ir) U_k† at time-reversal-
    paired k-points that carry a *complex* Bloch phase.

    For a TR-paired k the spatial operation lands on -k and the reconstruction
    applies a conjugation, so the Bloch phase must be evaluated at -k. Cubic
    supercells (LiH, test above) only have *real* phases at their TR points,
    where +k == -k, so they do not exercise this. Hexagonal hBN on a
    Gamma-centered 6x6x1 mesh folds heavily via time reversal with complex
    phases (e^{±i pi/3}); a wrong choice of +k there gives an O(1) residual on
    the inter-atom blocks.
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


# ---------------------------------------------------------------------------
# select_ops tests
# ---------------------------------------------------------------------------

def test_select_ops_none_is_identity_passthrough():
    """mode='none' returns the original ops list unchanged."""
    from green_mbtools.mint import symmetry_utils as su
    cell = _afm_cell()
    ops = su.detect_space_group(cell).ops
    assert su.select_ops(ops, "none", None, cell) == ops


def test_select_ops_symmorphic_matches_native():
    """symmorphic subset equals PySCF's own cell.symmorphic=True ops (rotation lists match)."""
    from green_mbtools.mint import symmetry_utils as su
    cell = _afm_cell()
    ops = su.detect_space_group(cell).ops
    got = su.select_ops(ops, "symmorphic", None, cell)
    assert all(op.trans_is_zero for op in got)
    native = _native_symmorphic_ops(_afm_cell_symmorphic())
    assert sorted(repr(o.rot.tolist()) for o in got) == sorted(repr(o.rot.tolist()) for o in native)


def test_select_ops_indices_returns_requested_and_adds_identity():
    """mode='indices' returns a subgroup containing identity even when not requested."""
    from green_mbtools.mint import symmetry_utils as su
    cell = _afm_cell()
    ops = su.detect_space_group(cell).ops
    sub = su.select_ops(ops, "indices", _sublattice_preserving_indices(cell, ops), cell)
    assert any(op.is_eye for op in sub)


def test_select_ops_rejects_non_closed_set():
    """mode='indices' raises ValueError for a non-closed set.

    ops[8] is an order-3 rotation; {E, ops[8]} is not closed because
    ops[8]*ops[8] = ops[9] which is absent.  select_ops must detect this
    and raise with a message containing 'closed' or 'closure'.
    """
    import pytest
    from green_mbtools.mint import symmetry_utils as su
    cell = _afm_cell()
    ops = su.detect_space_group(cell).ops
    # [0, 8] = E + one order-3 rotation: not closed, closure needs its square (ops[9])
    with pytest.raises(ValueError, match="closed|closure"):
        su.select_ops(ops, "indices", [0, 8], cell)


def test_select_ops_rejects_out_of_range():
    """mode='indices' raises ValueError when an index is out of range."""
    import pytest
    from green_mbtools.mint import symmetry_utils as su
    cell = _afm_cell()
    ops = su.detect_space_group(cell).ops
    with pytest.raises(ValueError, match="range|print_symmetry_ops"):
        su.select_ops(ops, "indices", [999], cell)


def test_select_ops_indices_none_raises_valueerror():
    """mode='indices' with spec=None raises ValueError mentioning 'indices'."""
    import pytest
    from green_mbtools.mint import symmetry_utils as su
    cell = _afm_cell()
    ops = su.detect_space_group(cell).ops
    with pytest.raises(ValueError, match="indices"):
        su.select_ops(ops, "indices", None, cell)


# ---------------------------------------------------------------------------
# build_symmetry tests
# ---------------------------------------------------------------------------

def test_build_symmetry_injection_changes_ibz():
    """build_symmetry with identity-only ops gives a larger IBZ than the full group.

    Injects a {E}-only Symmetry into cell.lattice_symmetry; make_kpts must use
    the injected ops without re-detecting (so nkpts_ibz grows to 27 == 3^3).
    Also verifies the .cell back-reference is deleted so KPoints.cell is intact.
    """
    from green_mbtools.mint import symmetry_utils as su
    full = _afm_cell()
    n_full = full.make_kpts([3, 3, 3], space_group_symmetry=True,
                            time_reversal_symmetry=False).nkpts_ibz
    low = _afm_cell()
    ops = su.detect_space_group(low).ops
    low.lattice_symmetry = su.build_symmetry(low, su.select_ops(ops, "indices", [0], low))  # {E}
    n_low = low.make_kpts([3, 3, 3], space_group_symmetry=True,
                          time_reversal_symmetry=False).nkpts_ibz
    assert n_low > n_full                        # fewer ops -> larger IBZ, no re-detection
    assert not hasattr(low.lattice_symmetry, "cell")


def test_build_symmetry_spacegroup_and_symmorphic_attrs():
    """build_symmetry sets spacegroup and symmorphic on the returned Symmetry.

    Uses the identity-only subgroup ({E}), which is trivially symmorphic.
    Checks:
    - spacegroup is not None and its ops match the injected ops.
    - symmorphic reflects whether all injected ops have zero fractional translation.
    - neither result nor result.spacegroup carries a live .cell reference.
    """
    from green_mbtools.mint import symmetry_utils as su
    cell = _afm_cell()
    ops = su.detect_space_group(cell).ops
    identity_ops = su.select_ops(ops, "indices", [0], cell)  # {E}
    result = su.build_symmetry(cell, identity_ops)

    # spacegroup is attached and its op list matches the injected ops.
    assert result.spacegroup is not None
    assert len(result.spacegroup.ops) == len(result.ops)

    # symmorphic: all identity ops have zero translation, so must be True.
    expected_symmorphic = all(op.trans_is_zero for op in result.ops)
    assert result.symmorphic == expected_symmorphic

    # No live .cell back-references on result or its spacegroup.
    assert not hasattr(result, "cell")
    assert not hasattr(result.spacegroup, "cell")


# ---------------------------------------------------------------------------
# apply_symmetry_restriction / mode_from_args tests (Task 4)
# ---------------------------------------------------------------------------

def test_apply_restriction_noop_when_space_symm_false():
    from argparse import Namespace
    from green_mbtools.mint import symmetry_utils as su
    cell = _afm_cell()
    before = cell.lattice_symmetry
    su.apply_symmetry_restriction(cell, Namespace(space_symm=False, symmorphic_only=False,
                                                  symm_ops=None, symm_backend="pyscf"))
    assert cell.lattice_symmetry is before


def test_mode_from_args_mutual_exclusivity():
    import pytest
    from argparse import Namespace
    from green_mbtools.mint import symmetry_utils as su
    with pytest.raises(ValueError, match="mutual|both"):
        su.mode_from_args(Namespace(symmorphic_only=True, symm_ops="0,1"))


def test_apply_restriction_indices_shrinks_group():
    from argparse import Namespace
    from green_mbtools.mint import symmetry_utils as su
    cell = _afm_cell()
    args = Namespace(space_symm=True, symmorphic_only=False, symm_ops="0",
                     symm_backend="pyscf")
    su.apply_symmetry_restriction(cell, args)
    assert cell.lattice_symmetry.nop == 1


def test_apply_restriction_noop_in_none_mode():
    """mode='none' must leave cell.lattice_symmetry untouched (identity check).

    When no reduction flag is set, apply_symmetry_restriction should return
    early without calling build_symmetry, so PySCF's own lattice_symmetry
    object is retained by identity (``is`` check).
    """
    from argparse import Namespace
    from green_mbtools.mint import symmetry_utils as su
    cell = _afm_cell()
    before = cell.lattice_symmetry
    su.apply_symmetry_restriction(
        cell,
        Namespace(space_symm=True, symmorphic_only=False, symm_ops=None, symm_backend="pyscf"),
    )
    assert cell.lattice_symmetry is before
    assert cell.lattice_symmetry.nop == 12


def test_apply_restriction_indices_emits_qspace_warning():
    """mode='indices' must emit a UserWarning mentioning q-space/auxcell.

    Uses the sublattice-preserving C3v subgroup [0,7,8,9,10,11] for _afm_cell()
    (verified closed). The warning should alert users that auxcell q-space
    datasets are still derived from the full group.
    """
    import warnings
    from argparse import Namespace
    from green_mbtools.mint import symmetry_utils as su

    cell = _afm_cell()
    args = Namespace(
        space_symm=True,
        symmorphic_only=False,
        symm_ops="0,7,8,9,10,11",
        symm_backend="pyscf",
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        su.apply_symmetry_restriction(cell, args)

    qspace_warnings = [x for x in w if issubclass(x.category, UserWarning)
                       and ("q-space" in str(x.message) or "auxcell" in str(x.message))]
    assert qspace_warnings, (
        f"Expected a UserWarning mentioning 'q-space' or 'auxcell' but got: "
        f"{[str(x.message) for x in w]}"
    )


def test_apply_restriction_no_qspace_warning_for_none_and_symmorphic():
    """mode='none' and mode='symmorphic' must NOT emit the q-space warning.

    The auxcell/q-space warning is specific to the 'indices' mode (--symm_ops).
    Other modes should be silent.
    """
    import warnings
    from argparse import Namespace
    from green_mbtools.mint import symmetry_utils as su

    # mode='none': no reduction flags
    cell_none = _afm_cell()
    args_none = Namespace(
        space_symm=True,
        symmorphic_only=False,
        symm_ops=None,
        symm_backend="pyscf",
    )
    with warnings.catch_warnings(record=True) as w_none:
        warnings.simplefilter("always")
        su.apply_symmetry_restriction(cell_none, args_none)

    qspace_none = [x for x in w_none if issubclass(x.category, UserWarning)
                   and ("q-space" in str(x.message) or "auxcell" in str(x.message))]
    assert not qspace_none, (
        f"Unexpected q-space warning for mode='none': {[str(x.message) for x in qspace_none]}"
    )

    # mode='symmorphic': symmorphic_only=True
    cell_symm = _afm_cell()
    args_symm = Namespace(
        space_symm=True,
        symmorphic_only=True,
        symm_ops=None,
        symm_backend="pyscf",
    )
    with warnings.catch_warnings(record=True) as w_symm:
        warnings.simplefilter("always")
        su.apply_symmetry_restriction(cell_symm, args_symm)

    qspace_symm = [x for x in w_symm if issubclass(x.category, UserWarning)
                   and ("q-space" in str(x.message) or "auxcell" in str(x.message))]
    assert not qspace_symm, (
        f"Unexpected q-space warning for mode='symmorphic': {[str(x.message) for x in qspace_symm]}"
    )


# ---------------------------------------------------------------------------
# format_symmetry_ops tests (Task 5)
# ---------------------------------------------------------------------------

def test_format_symmetry_ops_lists_all_and_flags_sublattice_swap():
    from green_mbtools.mint import symmetry_utils as su
    cell = _afm_cell(); info = su.detect_space_group(cell)
    text = su.format_symmetry_ops(cell, info)
    assert text.count("\n") >= len(info.ops)          # one line per op (plus header)
    assert "-3m" in text                               # group label present
    assert "perm" in text.lower()                      # atom-permutation column present


# ---------------------------------------------------------------------------
# AFM round-trip validation tests (Task 7)
# ---------------------------------------------------------------------------

_TRIPWIRE_SUBSTR = "stored k-point symmetry"


def test_afm_full_symmetry_trips_tripwire(tmp_path):
    """Full D3d group breaks collinear AFM -> check_kspace_symmetry_breaking warns.

    The AFM solution has unequal spin blocks; the stored full-group operators
    map H0<->H1 (sublattice swap), so reconstruction fails with residual ~0.774.
    The tripwire warning must fire.
    """
    import warnings
    from green_mbtools.mint import symmetry_utils as su

    out = _run_afm_case(tmp_path / "full", symm_ops=None)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        su.check_kspace_symmetry_breaking(str(out), ["HF/Fock-k"])
    assert any(_TRIPWIRE_SUBSTR in str(x.message) for x in w), (
        f"Expected a warning containing {_TRIPWIRE_SUBSTR!r} but got: {[str(x.message) for x in w]}"
    )


def test_afm_subgroup_roundtrips(tmp_path):
    """Sublattice-preserving subgroup keeps AFM intact and passes the tripwire silently.

    Derives the subgroup indices dynamically via _sublattice_preserving_indices
    (for this cell these are [0, 7, 8, 9, 10, 11], 6 ops of the C3v stabiliser
    of both H0 and H1 simultaneously).  After running with that restricted
    symmetry the check must be silent, and the two spin blocks of HF/Fock-k must
    differ by more than 1e-4 (AFM survived).
    """
    import warnings
    import h5py
    from green_mbtools.mint import symmetry_utils as su

    cell = _afm_cell()
    ops = su.detect_space_group(cell).ops
    # [0, 7, 8, 9, 10, 11] for this cell — derived dynamically so the test
    # stays correct if operator ordering changes.
    keep = ",".join(str(i) for i in _sublattice_preserving_indices(cell, ops))

    out = _run_afm_case(tmp_path / "low", symm_ops=keep)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        su.check_kspace_symmetry_breaking(str(out), ["HF/Fock-k"])
    assert not any(_TRIPWIRE_SUBSTR in str(x.message) for x in w), (
        f"Unexpected tripwire warning(s): {[str(x.message) for x in w if _TRIPWIRE_SUBSTR in str(x.message)]}"
    )

    with h5py.File(out, "r") as f:
        fock = f["HF/Fock-k"][()]
    assert np.max(np.abs(fock[0] - fock[1])) > 1e-4, (
        f"Spin blocks are nearly identical (diff={np.max(np.abs(fock[0]-fock[1])):.3e}); "
        "AFM did not survive the subgroup run."
    )
