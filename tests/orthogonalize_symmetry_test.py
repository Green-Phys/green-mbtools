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
        sym_kstruct=sym_kstruct, mycell=h2["cell"])
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
        sym_kstruct=sym_kstruct, mycell=h2["cell"])
    mk = h2["mk"]
    tr = max(np.max(np.abs(X_k[k] - X_k[mk[k]].conj())) for k in range(h2["nk"]))
    assert tr < 1e-10, f"natural X(-k) != X(k)* : {tr}"
    # still a valid orthonormalizer: X S X^dag = I
    orthn = max(np.max(np.abs(X_k[k] @ h2["S"][0, k] @ X_k[k].conj().T
                             - np.eye(h2["n"]))) for k in range(h2["nk"]))
    assert orthn < 1e-9, f"natural X S X^dag != I : {orthn}"


def test_orthogonalize_lowdin_enforces_time_reversal(h2):
    # Canonical Löwdin keeps the overlap eigenvectors, so like mo/natural it
    # is gauge-sensitive. The fixture passes S as complex dtype, so without the
    # real-input guard in lowdin_per_k, eigh at self-TR points (e.g. Γ) returns
    # arbitrary complex phases and X(-k) != X(k)* breaks. This is the regression
    # guard for the 4x4x4-Silicon lowdin failure.
    sym_kstruct = libkpts.make_kpts(h2["cell"], h2["kpts"],
                                    space_group_symmetry=False,
                                    time_reversal_symmetry=True)
    mydf = types.SimpleNamespace(kpts=h2["kpts"])
    X_k, *_ = comm.orthogonalize(
        mydf, "lowdin", [], [], h2["F"], h2["T"], h2["dm"], h2["S"],
        sym_kstruct=sym_kstruct, mycell=h2["cell"])
    mk = h2["mk"]
    tr = max(np.max(np.abs(X_k[k] - X_k[mk[k]].conj())) for k in range(h2["nk"]))
    assert tr < 1e-10, f"lowdin X(-k) != X(k)* : {tr}"
    # X must be real at self-TR points (k == -k), the property the guard restores.
    selftr = max(np.max(np.abs(X_k[k].imag))
                 for k in range(h2["nk"]) if mk[k] == k)
    assert selftr < 1e-10, f"lowdin X not real at self-TR k : {selftr}"
    # still a valid orthonormalizer: X S X^dag = I
    orthl = max(np.max(np.abs(X_k[k] @ h2["S"][0, k] @ X_k[k].conj().T
                             - np.eye(h2["n"]))) for k in range(h2["nk"]))
    assert orthl < 1e-9, f"lowdin X S X^dag != I : {orthl}"


def test_orthogonalize_symmetric_lowdin_enforces_time_reversal(h2):
    # Symmetric Löwdin X = S^{-1/2} is gauge-free, so X(-k) = X(k)* holds
    # structurally with no guard; this locks that in against regressions in the
    # unified build path.
    sym_kstruct = libkpts.make_kpts(h2["cell"], h2["kpts"],
                                    space_group_symmetry=False,
                                    time_reversal_symmetry=True)
    mydf = types.SimpleNamespace(kpts=h2["kpts"])
    X_k, *_ = comm.orthogonalize(
        mydf, "symmetric_lowdin", [], [], h2["F"], h2["T"], h2["dm"], h2["S"],
        sym_kstruct=sym_kstruct, mycell=h2["cell"])
    mk = h2["mk"]
    tr = max(np.max(np.abs(X_k[k] - X_k[mk[k]].conj())) for k in range(h2["nk"]))
    assert tr < 1e-10, f"symmetric_lowdin X(-k) != X(k)* : {tr}"
    orths = max(np.max(np.abs(X_k[k] @ h2["S"][0, k] @ X_k[k].conj().T
                             - np.eye(h2["n"]))) for k in range(h2["nk"]))
    assert orths < 1e-9, f"symmetric_lowdin X S X^dag != I : {orths}"


def test_orthogonalize_requires_kstruct_except_for_none(h2):
    # Every nontrivial mode requires sym_kstruct/mycell (X is built on the
    # irreducible wedge); only 'none' works without them.
    mydf = types.SimpleNamespace(kpts=h2["kpts"])
    for mode in ("lowdin", "mo", "natural", "symmetric_lowdin"):
        with pytest.raises(ValueError):
            comm.orthogonalize(
                mydf, mode, [], [], h2["F"], h2["T"], h2["dm"], h2["S"])
    comm.orthogonalize(
        mydf, 'none', [], [], h2["F"], h2["T"], h2["dm"], h2["S"])


def test_orthogonalize_dm_preserves_electron_number(h2):
    # The density matrix is contravariant: dm_orth = X_inv^dag dm X_inv (not
    # X_inv dm X_inv^dag). Pin the transform with three properties:
    #   - Tr(dm_orth(k)) == Tr(S_AO(k) dm_AO(k)) = N_k   (electron count),
    #   - dm_orth Hermitian,
    #   - exact round-trip to AO: X(k)^dag dm_orth(k) X(k) == dm_AO(k).
    # The transposed ordering breaks the trace and round-trip for mo/natural.
    sym_kstruct = libkpts.make_kpts(h2["cell"], h2["kpts"],
                                    space_group_symmetry=False,
                                    time_reversal_symmetry=True)
    mydf = types.SimpleNamespace(kpts=h2["kpts"])
    for mode in ("lowdin", "symmetric_lowdin", "mo", "natural"):
        X_k, _, _, _, _, dm2 = comm.orthogonalize(
            mydf, mode, [], [], h2["F"], h2["T"], h2["dm"], h2["S"],
            sym_kstruct=sym_kstruct, mycell=h2["cell"])
        for k in range(h2["nk"]):
            n_ref = np.trace(h2["S"][0, k] @ h2["dm"][0, k]).real
            n_orth = np.trace(dm2[0, k]).real
            assert abs(n_orth - n_ref) < 1e-9, (
                f"{mode}: Tr(dm_orth)={n_orth} != N_k={n_ref} at k={k}")
            assert np.allclose(dm2[0, k], dm2[0, k].conj().T, atol=1e-10), (
                f"{mode}: dm_orth not Hermitian at k={k}")
            dm_ao = X_k[k].conj().T @ dm2[0, k] @ X_k[k]
            assert np.allclose(dm_ao, h2["dm"][0, k], atol=1e-9), (
                f"{mode}: dm round-trip to AO failed at k={k}")


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


def test_space_symm_orth_mo(tmp_path):
    # Full space-group symmetry with the gauge-sensitive mo basis. X is built on
    # the IBZ and propagated by the space-group AO reps; the stored one-body
    # matrices must reconstruct as U(k) X(k_ir) U(k)dag to ~machine precision.
    # (This path was previously guarded off as "Phase 2".)
    out = _run_init(tmp_path, ["--orth", "mo", "--space_symm", "true", "--tr_symm", "true"])
    assert _kspace_residual(out) < 1e-9


def test_space_symm_orth_natural(tmp_path):
    out = _run_init(tmp_path, ["--orth", "natural", "--space_symm", "true", "--tr_symm", "true"])
    assert _kspace_residual(out) < 1e-9


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
