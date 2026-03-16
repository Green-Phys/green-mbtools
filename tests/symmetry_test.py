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
    symm_output, symm_cderi = _run_grid_only_case(base / "spacce_and_tr_symm_true", space_symm=True, tr_symm=True, nk=3)
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


def test_x2c_space_symm_not_supported(tmp_path):
    """For X2C1e, space symmetry is ignored and AO k-space transforms are identity."""
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

        idx_true = f_true["symmetry/k/bz2ibz"][()]
        idx_false = f_false["symmetry/k/bz2ibz"][()]
        ir_true = f_true["symmetry/k/ibz2bz"][()]
        ir_false = f_false["symmetry/k/ibz2bz"][()]
        conj_true = f_true["symmetry/k/tr_conj"][()]
        conj_false = f_false["symmetry/k/tr_conj"][()]

        kops_true = f_true["symmetry/k/k_sym_transform_ao"][()]
        kops_false = f_false["symmetry/k/k_sym_transform_ao"][()]

    # X2C disables space-group symmetry, so both runs should match TR-only reduction.
    assert nk_true == nk_false
    assert ink_true == ink_false
    np.testing.assert_array_equal(idx_true, idx_false)
    np.testing.assert_array_equal(ir_true, ir_false)
    np.testing.assert_array_equal(conj_true, conj_false)

    # With TR enabled, we still expect a reduced IBZ for nk > 1.
    if nk_true > 1:
        assert ink_true < nk_true

    # In current X2C path, AO-space symmetry transforms are identity matrices.
    nso = kops_true.shape[1]
    eye = np.eye(nso, dtype=np.complex128)
    eye_stack = np.broadcast_to(eye, kops_true.shape)
    np.testing.assert_allclose(kops_true, eye_stack, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(kops_false, eye_stack, atol=1e-12, rtol=0.0)


@pytest.mark.skip(reason="TODO: validate k_sym_transform_p0 transformation against an independent real-data reference")
def test_k_sym_transform_p0_matches_metric_basis_transform(generated_cases):
    """TODO: replace implementation-coupled check with a real-data validation."""
    pass
