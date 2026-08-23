import os
import types
from pathlib import Path

import h5py
import numpy as np
import pytest

from pyscf.pbc.lib import kpts as libkpts

import green_mbtools.mint as pymb
from green_mbtools.mint import common_utils as comm
from green_mbtools.mint import ortho_utils


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
    # Build the symmetry struct exactly as production does (make_kpts on the
    # actual k-mesh), so this test is sensitive to k-ordering / shifted-mesh
    # mistakes in the construction used at runtime.
    sym_kstruct = libkpts.make_kpts(h2["cell"], h2["kpts"],
                                    space_group_symmetry=False,
                                    time_reversal_symmetry=True)
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
    sym_kstruct = libkpts.make_kpts(h2["cell"], h2["kpts"],
                                    space_group_symmetry=False,
                                    time_reversal_symmetry=True)
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


def test_orthogonalize_mo_sym_kstruct_does_not_require_mf(h2):
    # The sym_kstruct 'mo' path builds X from F_ibz/S_ibz and never uses mf,
    # so it must work with mf=None (the mf-required guard applies only to the
    # legacy per-k path).
    sym_kstruct = libkpts.make_kpts(h2["cell"], h2["kpts"],
                                    space_group_symmetry=False,
                                    time_reversal_symmetry=True)
    mydf = types.SimpleNamespace(kpts=h2["kpts"])
    X_k, *_ = comm.orthogonalize(
        mydf, "mo", [], [], h2["F"], h2["T"], h2["dm"], h2["S"],
        mf=None, sym_kstruct=sym_kstruct, mycell=h2["cell"])
    mk = h2["mk"]
    tr = max(np.max(np.abs(X_k[k] - X_k[mk[k]].conj())) for k in range(h2["nk"]))
    assert tr < 1e-10, f"mo (mf=None) X(-k) != X(k)* : {tr}"
    orth = max(np.max(np.abs(X_k[k] @ h2["S"][0, k] @ X_k[k].conj().T
                             - np.eye(h2["n"]))) for k in range(h2["nk"]))
    assert orth < 1e-9, f"mo (mf=None) X S X^dag != I : {orth}"


def test_orthogonalize_mo_without_kstruct_still_requires_mf(h2):
    # The legacy per-k 'mo' path uses mf.mo_coeff, so mf is still required there.
    mydf = types.SimpleNamespace(kpts=h2["kpts"])
    with pytest.raises(ValueError):
        comm.orthogonalize(mydf, "mo", [], [], h2["F"], h2["T"], h2["dm"], h2["S"],
                           mf=None)


def test_orthogonalize_without_kstruct_is_unchanged(h2):
    # The no-kstruct path must produce exactly the per-k Lowdin result: assert
    # every returned array (X, X_inv, F, T, dm, S) matches a reference assembled
    # directly from lowdin_per_k + transform, not merely the X S X^dag = I
    # invariant (which a different gauge could also satisfy).
    n, nk = h2["n"], h2["nk"]
    mydf = types.SimpleNamespace(kpts=h2["kpts"])
    X_k, X_inv_k, S2, F2, T2, dm2 = comm.orthogonalize(
        mydf, "lowdin", [], [], h2["F"], h2["T"], h2["dm"], h2["S"])

    Xref = np.empty((nk, n, n), dtype=np.complex128)
    Xiref = np.empty((nk, n, n), dtype=np.complex128)
    for k in range(nk):
        x, x_inv = ortho_utils.lowdin_per_k(h2["S"][0, k])
        Xref[k], Xiref[k] = x, x_inv
    Fref = comm.transform(h2["F"], Xref, Xiref)
    Tref = comm.transform(h2["T"], Xref, Xiref)
    dmref = comm.transform(h2["dm"], Xiref, Xref)   # dm is contravariant
    Sref = np.array([[np.eye(n, dtype=np.complex128)] * nk])

    assert np.allclose(np.asarray(X_k), Xref, atol=1e-12)
    assert np.allclose(np.asarray(X_inv_k), Xiref, atol=1e-12)
    assert np.allclose(F2, Fref, atol=1e-12)
    assert np.allclose(T2, Tref, atol=1e-12)
    assert np.allclose(dm2, dmref, atol=1e-12)
    assert np.allclose(S2, Sref, atol=1e-12)


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


def test_end_to_end_ksym_consistency_shifted_mesh(tmp_path):
    # Coverage for the orth path on a non-Gamma-centered (half-shifted) mesh.
    # The sym_kstruct that builds X is the symmetry decomposition of self.kmesh
    # (via make_kpts), so its k-ordering matches the S/F arrays that orthogonalize
    # indexes with ibz2bz. (This does not discriminate the make_kpts vs
    # build_q_struct choice: on TR-valid meshes both give identical X. It guards
    # the shifted-mesh store/reconstruct pipeline generally.)
    out = _run_init(tmp_path, [
        "--orth", "mo", "--space_symm", "false", "--tr_symm", "true",
        "--center", "0.5", "0.5", "0.5",
    ])
    assert _kspace_residual(out) < 1e-9


def _minus_k_index_h5(mesh_scaled):
    nk = len(mesh_scaled)
    out = np.empty(nk, dtype=int)
    for k in range(nk):
        s = mesh_scaled + mesh_scaled[k]
        hit = np.all(np.abs(s - np.round(s)) < 1e-6, axis=1)
        out[k] = int(np.nonzero(hit)[0][0])
    return out


def _tr_conjugate_residual(input_h5, dset):
    with h5py.File(input_h5, "r") as f:
        def cx(a):
            a = a[...]
            return a if np.iscomplexobj(a) else a[..., 0] + 1j * a[..., 1]
        mk = _minus_k_index_h5(f["symmetry/k/mesh_scaled"][()])
        Z = cx(f[dset])            # (ns, nk, n, n)
        worst = 0.0
        for s in range(Z.shape[0]):
            for k in range(Z.shape[1]):
                worst = max(worst, float(np.max(np.abs(Z[s, mk[k]] - Z[s, k].conj()))))
        return worst


def test_orth_mo_enforces_tr_when_tr_symm_false(tmp_path):
    # The X-build enforces X(-k)=X(k)* independently of --tr_symm (the df pair
    # reduction always folds k->-k). With --tr_symm false the exported reduction
    # is trivial, so this is NOT a reconstruction assertion: we check the stored
    # orthogonal hcore obeys T_orth(-k) = conj(T_orth(k)) on the actual k-mesh.
    # Since T_ao(-k) = conj(T_ao(k)), that identity holds iff X(-k) = X(k)*.
    # HF/H-k (hcore) is off-diagonal in the MO basis, so it is gauge-sensitive
    # (unlike the diagonal Fock, which is TR-trivial regardless of gauge).
    out = _run_init(tmp_path, ["--orth", "mo", "--space_symm", "false", "--tr_symm", "false"])
    assert _tr_conjugate_residual(out, "HF/H-k") < 1e-9
