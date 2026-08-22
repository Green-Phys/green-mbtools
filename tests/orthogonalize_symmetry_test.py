import os
import types
from pathlib import Path

import h5py
import numpy as np
import pytest

import green_mbtools.mint as pymb
from green_mbtools.mint import common_utils as comm
from green_mbtools.mint.kpt_utils import build_q_struct


def _h2_krhf(nk=3):
    """Small H2 PBC RHF; returns (cell, kpts, mf)."""
    from pyscf.pbc import gto as pbc_gto, scf as pbc_scf
    cell = pbc_gto.Cell()
    cell.atom = "H -0.25 -0.25 -0.25\nH  0.25  0.25  0.25"
    cell.a = np.eye(3) * 4.0655
    cell.basis = "gth-dzvp-molopt-sr"
    cell.pseudo = "gth-pbe"
    cell.unit = "Angstrom"
    cell.verbose = 0
    cell.build()
    kpts = cell.make_kpts([nk, nk, nk])
    mf = pbc_scf.KRHF(cell, kpts).density_fit()
    mf.kernel()
    return cell, kpts, mf


def _minus_k_index(cell, kpts):
    """For each k, index j with kpts[j] == -kpts[k] (mod reciprocal lattice)."""
    sk = cell.get_scaled_kpts(np.asarray(kpts))
    out = np.empty(len(kpts), dtype=int)
    for k in range(len(kpts)):
        s = sk + sk[k]
        hit = np.all(np.abs(s - np.round(s)) < 1e-6, axis=1)
        out[k] = int(np.nonzero(hit)[0][0])
    return out


@pytest.fixture(scope="module")
def h2():
    cell, kpts, mf = _h2_krhf(nk=3)
    n = cell.nao_nr()
    nk = len(kpts)
    S = np.array([np.asarray(mf.get_ovlp())])          # (1, nk, n, n)
    T = np.array([np.asarray(mf.get_hcore())])
    F = np.asarray(mf.get_fock()).reshape(1, nk, n, n)
    dm = np.asarray(mf.make_rdm1()).reshape(1, nk, n, n)
    return dict(cell=cell, kpts=kpts, mf=mf, n=n, nk=nk,
                S=S.astype(complex), T=T.astype(complex),
                F=F.astype(complex), dm=dm.astype(complex),
                mk=_minus_k_index(cell, kpts))


def test_orthogonalize_mo_enforces_time_reversal(h2):
    sym_kstruct = build_q_struct(h2["cell"], h2["kpts"],
                                 space_symm=False, tr_symm=True)
    mydf = types.SimpleNamespace(kpts=h2["kpts"])
    X_k, X_inv_k, S2, F2, T2, dm2 = comm.orthogonalize(
        mydf, "mo", [], [], h2["F"], h2["T"], h2["dm"], h2["S"],
        mf=h2["mf"], sym_kstruct=sym_kstruct, mycell=h2["cell"])
    mk = h2["mk"]
    tr = max(np.max(np.abs(X_k[k] - X_k[mk[k]].conj())) for k in range(h2["nk"]))
    assert tr < 1e-10, f"X(-k) != X(k)* : {tr}"
    # still a valid orthonormalizer: X S X^dag = I
    orth = max(np.max(np.abs(X_k[k] @ h2["S"][0, k] @ X_k[k].conj().T
                             - np.eye(h2["n"]))) for k in range(h2["nk"]))
    assert orth < 1e-9, f"X S X^dag != I : {orth}"


def test_orthogonalize_natural_enforces_time_reversal(h2):
    sym_kstruct = build_q_struct(h2["cell"], h2["kpts"],
                                 space_symm=False, tr_symm=True)
    mydf = types.SimpleNamespace(kpts=h2["kpts"])
    X_k, *_ = comm.orthogonalize(
        mydf, "natural", [], [], h2["F"], h2["T"], h2["dm"], h2["S"],
        mf=h2["mf"], sym_kstruct=sym_kstruct, mycell=h2["cell"])
    mk = h2["mk"]
    tr = max(np.max(np.abs(X_k[k] - X_k[mk[k]].conj())) for k in range(h2["nk"]))
    assert tr < 1e-10, f"natural X(-k) != X(k)* : {tr}"
    # still a valid orthonormalizer: X S X^dag = I
    orthn = max(np.max(np.abs(X_k[k] @ h2["S"][0, k] @ X_k[k].conj().T
                             - np.eye(h2["n"]))) for k in range(h2["nk"]))
    assert orthn < 1e-9, f"natural X S X^dag != I : {orthn}"


def test_orthogonalize_without_kstruct_is_unchanged(h2):
    mydf = types.SimpleNamespace(kpts=h2["kpts"])
    X_k, X_inv_k, S2, *_ = comm.orthogonalize(
        mydf, "lowdin", [], [], h2["F"], h2["T"], h2["dm"], h2["S"])
    # lowdin is deterministic in S: X S X^dag = I everywhere
    orth = max(np.max(np.abs(X_k[k] @ h2["S"][0, k] @ X_k[k].conj().T
                             - np.eye(h2["n"]))) for k in range(h2["nk"]))
    assert orth < 1e-9


def _run_init(tmp_path, extra):
    params = [
        "--atom", "H -0.25 -0.25 -0.25\nH  0.25  0.25  0.25",
        "--a", "4.0655,0,0\n0,4.0655,0\n0,0,4.0655\n",
        "--basis", "gth-dzvp-molopt-sr", "--pseudo", "gth-pbe",
        "--output_path", str(tmp_path / "input.h5"),
        "--df_int", "0", "--nk", "3", "--restricted", "1",
        "--use_j2c_eig_decomposition", "false",
    ] + extra
    old = Path.cwd()
    os.chdir(tmp_path)
    try:
        init = pymb.pyscf_pbc_init(comm.init_pbc_params(params))
        init.mean_field_input()
    finally:
        os.chdir(old)
    return tmp_path / "input.h5"


def test_space_symm_orth_is_guarded(tmp_path):
    with pytest.raises(NotImplementedError):
        _run_init(tmp_path, ["--orth", "mo", "--space_symm", "true", "--tr_symm", "true"])


def _kspace_residual(input_h5):
    with h5py.File(input_h5, "r") as f:
        def cx(a):
            a = a[...]
            return a if np.iscomplexobj(a) else a[..., 0] + 1j * a[..., 1]
        bz2ibz = f["symmetry/k/bz2ibz"][()]
        tr_conj = f["symmetry/k/tr_conj"][()]
        U = cx(f["symmetry/k/k_sym_transform_ao"])
        worst = 0.0
        for dset in ("HF/H-k", "HF/S-k", "HF/Fock-k"):
            X = cx(f[dset])            # (ns, nk, n, n)
            for s in range(X.shape[0]):
                for k in range(X.shape[1]):
                    kir = int(bz2ibz[k])
                    rec = U[k] @ X[s, kir] @ U[k].conj().T
                    if int(tr_conj[k]) != 0:
                        rec = rec.conjugate()
                    worst = max(worst, float(np.max(np.abs(rec - X[s, k]))))
        return worst


def test_end_to_end_ksym_consistency(tmp_path):
    out = _run_init(tmp_path, ["--orth", "mo", "--space_symm", "false", "--tr_symm", "true"])
    assert _kspace_residual(out) < 1e-9


def test_end_to_end_ksym_consistency_natural(tmp_path):
    out = _run_init(tmp_path, ["--orth", "natural", "--space_symm", "false", "--tr_symm", "true"])
    assert _kspace_residual(out) < 1e-9
