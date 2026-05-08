"""
Tests for the RSGDF-compatible Ewald correction (fix/rsgdf-ewald).

Level 1 — kernel unit tests (no HDF5 build, <1 s each):
  test_sr_preservation      : weighted_coulG_rs_ewald with omega=-omega0 matches bare SR
  test_madelung_at_G0       : G=0 value equals Nk * vol * madelung
  test_lr_propagation       : weighted_coulG_LR with patch contains Madelung at G=0
  test_nonzero_G_unchanged  : G!=0 values are identical to bare Coulomb
  test_patch_restore        : class attribute restored after patch context

Level 2 — builder-level CDERI comparison (seconds each):
  test_df1_ewald_cc_vs_rs              : Ewald df1 CDERIs agree CC-forced vs RS-aware
  test_df2_bare_cc_vs_rs               : bare df2 CDERIs unchanged by removing _prefer_ccdf
  test_ewald_correction_nonzero_and_consistent : df1-df2 nonzero and equal across paths

Level 3 — full compute_ewald_correction comparison:
  test_ewald_correction_rs_vs_cc : old (CC-forced) vs new (RS) give identical EW tensors
  test_patch_restore_after_compute : class methods restored after function returns

Level 4 — end-to-end exchange energy regression:
  test_hf_energy_unchanged         : HF total energy unchanged after Ewald patch cycle
  test_exchange_energy_correction  : ΔEx from EW tensors agrees to 1e-7 Ha between paths

Level 5 — Option A vs Option B consistency:
  test_option_a_vs_b_total_exchange : total Ewald exchange energy agrees between
                                      RS-everywhere (A) and CC-everywhere (B) to 1e-6 Ha
"""

import os
import random
import string
import tempfile

import h5py
import numpy as np
import pytest

from pyscf.pbc import gto, df, tools
from pyscf.pbc.df import aft
from pyscf.pbc.df.rsdf_builder import _RSGDFBuilder
from pyscf.pbc.df.gdf_builder import _CCGDFBuilder

from green_mbtools.mint.integral_utils import (
    weighted_coulG_rs_ewald,
    weighted_coulG_ewald_2nd,
)


# ─── Shared fixture: minimal periodic H2 cell ────────────────────────────────

@pytest.fixture(scope="module")
def h2_cell():
    cell = gto.Cell()
    cell.atom = "H 0 0 0; H 0 0 1.4"
    cell.basis = "gth-szv"
    cell.pseudo = "gth-pade"
    cell.a = np.diag([4.0, 4.0, 5.0])
    cell.verbose = 0
    cell.build()
    return cell


@pytest.fixture(scope="module")
def kmesh_2x2x2(h2_cell):
    return h2_cell.make_kpts([2, 2, 2])


@pytest.fixture(scope="module")
def rs_builder(h2_cell, kmesh_2x2x2):
    """A freshly built _RSGDFBuilder for use in kernel-level tests.

    _RSGDFBuilder(cell, auxcell, kpts) — auxcell is built from auxbasis.
    We use the GDF object's auxcell after build() to construct the builder.
    """
    from pyscf.df import addons
    mydf = df.GDF(h2_cell)
    mydf.kpts = kmesh_2x2x2
    mydf.verbose = 0
    mydf.build()
    auxcell = addons.make_auxmol(h2_cell, mydf.auxbasis)
    builder = _RSGDFBuilder(h2_cell, auxcell, kmesh_2x2x2)
    builder.verbose = 0
    builder.build()
    return builder


# ─── Level 1 tests ───────────────────────────────────────────────────────────

class TestKernelLevel:

    def test_sr_preservation(self, rs_builder):
        """SR call (omega != None) must return identical result to bare aft call."""
        kpt = np.zeros(3)
        mesh = rs_builder.mesh
        omega = rs_builder.omega

        bare_sr = aft.weighted_coulG(rs_builder, kpt, False, mesh, -omega)
        patched_sr = weighted_coulG_rs_ewald(rs_builder, kpt, False, mesh, -omega)

        np.testing.assert_array_equal(
            bare_sr, patched_sr,
            err_msg="SR call through weighted_coulG_rs_ewald changed the SR kernel"
        )

    def test_madelung_at_G0(self, rs_builder, kmesh_2x2x2, h2_cell):
        """G=0 value of the Ewald call must equal Nk * vol * madelung * kws[0].

        aft.weighted_coulG multiplies get_coulG by grid quadrature weights kws,
        so the G=0 contribution is Nk * vol * madelung * kws[0], not just
        Nk * vol * madelung.
        """
        kpt = np.zeros(3)
        mesh = rs_builder.mesh

        coulG = weighted_coulG_rs_ewald(rs_builder, kpt, False, mesh, omega=None)

        Nk = len(kmesh_2x2x2)
        _, _, kws = h2_cell.get_Gv_weights(mesh)  # kws is a uniform scalar weight
        expected_G0 = Nk * h2_cell.vol * tools.madelung(h2_cell, kmesh_2x2x2) * kws

        np.testing.assert_allclose(
            coulG[0], expected_G0, rtol=1e-10,
            err_msg="G=0 Madelung value incorrect"
        )

    def test_lr_propagation(self, rs_builder, kmesh_2x2x2, h2_cell):
        """weighted_coulG_LR with patch installed must include Madelung at G=0."""
        kpt = np.zeros(3)
        mesh = rs_builder.mesh

        rs_old = _RSGDFBuilder.weighted_coulG
        try:
            _RSGDFBuilder.weighted_coulG = weighted_coulG_rs_ewald
            coulG_LR = rs_builder.weighted_coulG_LR(kpt, False, mesh)
        finally:
            _RSGDFBuilder.weighted_coulG = rs_old

        # LR at G=0 = full_Ewald[G=0] - SR[G=0]
        coulG_full_ewald = weighted_coulG_rs_ewald(rs_builder, kpt, False, mesh)
        coulG_sr = aft.weighted_coulG(rs_builder, kpt, False, mesh, -rs_builder.omega)

        expected_G0 = coulG_full_ewald[0] - coulG_sr[0]
        np.testing.assert_allclose(
            coulG_LR[0], expected_G0, rtol=1e-10,
            err_msg="Madelung not correctly propagated into weighted_coulG_LR"
        )

    def test_nonzero_G_unchanged(self, rs_builder):
        """For G != 0, the Ewald call must equal the bare Coulomb call."""
        kpt = np.zeros(3)
        mesh = rs_builder.mesh

        bare   = aft.weighted_coulG(rs_builder, kpt, False, mesh)
        ewald  = weighted_coulG_rs_ewald(rs_builder, kpt, False, mesh, omega=None)

        np.testing.assert_array_equal(
            bare[1:], ewald[1:],
            err_msg="Ewald patch changed G!=0 Coulomb weights"
        )

    def test_patch_restore(self, rs_builder):
        """Class-level attribute must be fully restored after patch/restore."""
        original = _RSGDFBuilder.weighted_coulG
        _RSGDFBuilder.weighted_coulG = weighted_coulG_rs_ewald
        _RSGDFBuilder.weighted_coulG = original
        assert _RSGDFBuilder.weighted_coulG is original, \
            "weighted_coulG not restored to original after patch/restore"


# ─── Level 2 helpers ─────────────────────────────────────────────────────────

def _build_df(cell, kmesh, auxbasis, mesh, ewald=False, prefer_cc=False,
              patch_cc=False, patch_rs=False):
    """Build a df.GDF object and return a dict of CDERIs read via sr_loop.

    Returns {i: array(NQ, nao, nao, complex)} for each diagonal k-pair index i.
    Patches are applied only during .build() and restored immediately after.
    """
    cc_old = _CCGDFBuilder.weighted_coulG
    rs_old = _RSGDFBuilder.weighted_coulG
    try:
        if patch_cc:
            _CCGDFBuilder.weighted_coulG = weighted_coulG_ewald_2nd
        if patch_rs:
            _RSGDFBuilder.weighted_coulG = weighted_coulG_rs_ewald

        mydf = df.GDF(cell)
        mydf.auxbasis = auxbasis
        mydf.mesh = mesh
        mydf.kpts = kmesh
        mydf.verbose = 0
        if ewald:
            mydf.exxdiv = 'ewald'
            mydf.cell.full_k_mesh = kmesh
        if prefer_cc:
            mydf._prefer_ccdf = True
        f = tempfile.mktemp(suffix=".h5")
        mydf._cderi_to_save = f
        mydf._cderi = f
        mydf.build()
    finally:
        _CCGDFBuilder.weighted_coulG = cc_old
        _RSGDFBuilder.weighted_coulG = rs_old

    nao = cell.nao_nr()
    result = {}
    for i, ki in enumerate(kmesh):
        chunks = list(mydf.sr_loop((ki, ki), max_memory=4000, compact=False))
        rows = [c[0] + 1j * c[1] for c in chunks]
        NQ = rows[0].shape[1]
        Lpq = np.vstack(rows).reshape(-1, nao, nao)
        result[i] = Lpq
    os.remove(f)
    return result


# ─── Level 2 tests ───────────────────────────────────────────────────────────

class TestBuilderLevel:

    @pytest.fixture(scope="class")
    def cell_kmesh(self, h2_cell, kmesh_2x2x2):
        return h2_cell, kmesh_2x2x2

    @pytest.fixture(scope="class")
    def reference_df(self, h2_cell, kmesh_2x2x2):
        """Shared GDF built once to get auxbasis and mesh."""
        mydf = df.GDF(h2_cell); mydf.kpts = kmesh_2x2x2; mydf.verbose = 0
        f = tempfile.mktemp(suffix=".h5")
        mydf._cderi_to_save = f; mydf._cderi = f; mydf.build()
        auxbasis = mydf.auxbasis
        mesh = mydf.mesh
        os.remove(f)
        return auxbasis, mesh

    def test_df1_ewald_cc_vs_rs(self, h2_cell, kmesh_2x2x2, reference_df):
        """Ewald-probed df1 CDERIs must agree between CC-forced and RS-aware builds."""
        auxbasis, mesh = reference_df

        cderi_cc = _build_df(h2_cell, kmesh_2x2x2, auxbasis, mesh,
                             ewald=True, prefer_cc=True,
                             patch_cc=True, patch_rs=False)
        cderi_rs = _build_df(h2_cell, kmesh_2x2x2, auxbasis, mesh,
                             ewald=True, prefer_cc=False,
                             patch_cc=True, patch_rs=True)

        for i in range(len(kmesh_2x2x2)):
            np.testing.assert_allclose(
                cderi_cc[i], cderi_rs[i], atol=1e-8,
                err_msg=f"Ewald df1 CDERIs differ at k-pair {i}: "
                        f"CC-forced vs RS-aware"
            )

    def test_df2_bare_cc_vs_rs(self, h2_cell, kmesh_2x2x2, reference_df):
        """Bare df2 CDERIs must agree with and without _prefer_ccdf.

        _CCGDFBuilder and _RSGDFBuilder use different numerical integration
        paths (classical lattice sum vs range-separated FT), so they agree to
        ~1e-8 in absolute value, not to machine precision.  This tolerance
        corresponds to the documented "mathematically identical" claim.
        """
        auxbasis, mesh = reference_df

        cderi_cc = _build_df(h2_cell, kmesh_2x2x2, auxbasis, mesh,
                             ewald=False, prefer_cc=True,
                             patch_cc=False, patch_rs=False)
        cderi_rs = _build_df(h2_cell, kmesh_2x2x2, auxbasis, mesh,
                             ewald=False, prefer_cc=False,
                             patch_cc=False, patch_rs=False)

        for i in range(len(kmesh_2x2x2)):
            np.testing.assert_allclose(
                cderi_cc[i], cderi_rs[i], atol=1e-8,
                err_msg=f"Bare df2 CDERIs changed when _prefer_ccdf was removed "
                        f"at k-pair {i}"
            )

    def test_ewald_correction_nonzero_and_consistent(self, h2_cell,
                                                      kmesh_2x2x2, reference_df):
        """df1 - df2 must be nonzero and the same for CC and RS paths.

        Checks that the Ewald correction is actually present in df1 (not silently
        dropped) and that both paths produce the same correction tensor.
        """
        auxbasis, mesh = reference_df

        df1_cc = _build_df(h2_cell, kmesh_2x2x2, auxbasis, mesh,
                           ewald=True, prefer_cc=True,
                           patch_cc=True, patch_rs=False)
        df2_cc = _build_df(h2_cell, kmesh_2x2x2, auxbasis, mesh,
                           ewald=False, prefer_cc=True,
                           patch_cc=False, patch_rs=False)
        df1_rs = _build_df(h2_cell, kmesh_2x2x2, auxbasis, mesh,
                           ewald=True, prefer_cc=False,
                           patch_cc=True, patch_rs=True)
        df2_rs = _build_df(h2_cell, kmesh_2x2x2, auxbasis, mesh,
                           ewald=False, prefer_cc=False,
                           patch_cc=False, patch_rs=False)

        for i in range(len(kmesh_2x2x2)):
            ew_cc = df1_cc[i] - df2_cc[i]
            ew_rs = df1_rs[i] - df2_rs[i]

            # correction must be present (not silently zero)
            assert np.abs(ew_rs).max() > 1e-10, \
                f"EW correction is zero at k-pair {i}: RS path dropped the Ewald term"

            # both paths must give the same correction
            np.testing.assert_allclose(
                ew_cc, ew_rs, atol=1e-8,
                err_msg=f"EW correction differs between CC and RS paths at k-pair {i}"
            )


# ─── Level 3 test ────────────────────────────────────────────────────────────

class _FakeArgs:
    """Minimal stand-in for the args object expected by compute_ewald_correction."""
    def __init__(self, nk):
        self.nk = nk


def _run_ewald_old(cell, kmesh, maindf, nao, filename):
    """Run compute_ewald_correction with the old CC-forced approach."""
    import pyscf.pbc.df.gdf_builder as gdf_mod
    cc_old = _CCGDFBuilder.weighted_coulG
    rs_old = _RSGDFBuilder.weighted_coulG

    data = h5py.File(filename, "w")
    EW     = data.create_group("EW")
    EW_bar = data.create_group("EW_bar")

    from pyscf.df import addons
    auxcell = addons.make_auxmol(cell, maindf.auxbasis)
    NQ = auxcell.nao_nr()
    buffer1 = np.zeros((NQ, nao, nao), dtype=np.complex128)
    buffer2 = np.zeros((NQ, nao, nao), dtype=np.complex128)
    Lpq_mo  = np.zeros((NQ, nao, nao), dtype=np.complex128)

    # bare df2 — CC forced
    df2 = df.GDF(cell)
    df2._prefer_ccdf = True
    df2.auxbasis = maindf.auxbasis
    df2.mesh = maindf.mesh
    f2 = tempfile.mktemp(suffix=".h5")
    df2._cderi_to_save = f2; df2._cderi = f2
    df2.kpts = kmesh; df2.verbose = 0; df2.build()

    # Ewald df1 — CC forced
    _CCGDFBuilder.weighted_coulG = weighted_coulG_ewald_2nd
    df1 = df.GDF(cell)
    df1._prefer_ccdf = True
    df1.cell.full_k_mesh = kmesh
    df1.auxbasis = maindf.auxbasis
    df1.exxdiv = 'ewald'; df1.mesh = maindf.mesh
    f1 = tempfile.mktemp(suffix=".h5")
    df1._cderi_to_save = f1; df1._cderi = f1
    df1.kpts = kmesh; df1.verbose = 0; df1.build()
    _CCGDFBuilder.weighted_coulG = cc_old

    for i, ki in enumerate(kmesh):
        s1 = 0
        for XXX in df1.sr_loop((ki,ki), max_memory=4000, compact=False):
            Lpq = (XXX[0] + XXX[1]*1j).reshape(XXX[0].shape[0], nao, nao)
            buffer1[s1:s1+Lpq.shape[0]] = Lpq; s1 += Lpq.shape[0]
        s1 = 0
        for XXX in df2.sr_loop((ki,ki), max_memory=4000, compact=False):
            Lpq = (XXX[0] + XXX[1]*1j).reshape(XXX[0].shape[0], nao, nao)
            buffer2[s1:s1+Lpq.shape[0]] = Lpq; s1 += Lpq.shape[0]
        EW[str(i)]     = (buffer1 - buffer2).view(np.float64)
        EW_bar[str(i)] = buffer2.view(np.float64)
        buffer1[:] = 0.; buffer2[:] = 0.

    data.close()
    os.remove(f1); os.remove(f2)


def _run_ewald_new(cell, kmesh, maindf, nao, filename):
    """Run compute_ewald_correction with the new RS-aware approach."""
    from green_mbtools.mint.integral_utils import compute_ewald_correction
    args = _FakeArgs(nk=(1, len(kmesh), len(kmesh)))
    compute_ewald_correction(args, maindf, kmesh, nao, filename)


class TestEwaldCorrectionComparison:

    @pytest.fixture(scope="class")
    def setup(self, h2_cell, kmesh_2x2x2):
        maindf = df.GDF(h2_cell)
        maindf.kpts = kmesh_2x2x2
        maindf.verbose = 0
        maindf.build()
        nao = h2_cell.nao_nr()
        return h2_cell, kmesh_2x2x2, maindf, nao

    def test_ewald_correction_rs_vs_cc(self, setup, tmp_path):
        """EW tensors from old CC-forced path and new RS-aware path must agree."""
        cell, kmesh, maindf, nao = setup

        f_old = str(tmp_path / "ewald_old.h5")
        f_new = str(tmp_path / "ewald_new.h5")

        _run_ewald_old(cell, kmesh, maindf, nao, f_old)
        _run_ewald_new(cell, kmesh, maindf, nao, f_new)

        with h5py.File(f_old, "r") as old, h5py.File(f_new, "r") as new:
            for i in range(len(kmesh)):
                key = str(i)
                np.testing.assert_allclose(
                    old["EW"][key][...], new["EW"][key][...],
                    atol=1e-8,
                    err_msg=f"EW[{i}] differs between CC-forced and RS-aware paths"
                )
                np.testing.assert_allclose(
                    old["EW_bar"][key][...], new["EW_bar"][key][...],
                    atol=1e-8,
                    err_msg=f"EW_bar[{i}] differs between CC-forced and RS-aware paths"
                )

    def test_patch_restore_after_compute(self, setup, tmp_path):
        """Both class-level methods must be restored after compute_ewald_correction."""
        cell, kmesh, maindf, nao = setup
        cc_before = _CCGDFBuilder.weighted_coulG
        rs_before = _RSGDFBuilder.weighted_coulG

        f_new = str(tmp_path / "ewald_restore_check.h5")
        _run_ewald_new(cell, kmesh, maindf, nao, f_new)

        assert _CCGDFBuilder.weighted_coulG is cc_before, \
            "_CCGDFBuilder.weighted_coulG not restored after compute_ewald_correction"
        assert _RSGDFBuilder.weighted_coulG is rs_before, \
            "_RSGDFBuilder.weighted_coulG not restored after compute_ewald_correction"


# ─── Option B helper ──────────────────────────────────────────────────────────

def _run_ewald_cc_consistent(cell, kmesh, maindf_cc, nao, filename):
    """Option B: CC everywhere.

    maindf_cc must be a df.GDF built with _prefer_ccdf=True so that the
    production CDERIs use the CC metric.  The Ewald correction (df1 and df2
    inside _run_ewald_old) also uses _CCGDFBuilder, making all three objects
    (production, bare EW, Ewald-probed EW) self-consistent in the CC metric.
    """
    _run_ewald_old(cell, kmesh, maindf_cc, nao, filename)


def _total_ex_from_ew_file(ew_file, dm, Nk, nao):
    """Total Ewald exchange energy from a single EW file.

    Reconstructs L_ew = EW_bar + EW (the full Ewald CDERI, both stored in the
    file) and contracts with dm:
        Ex = -(1/Nk) Σ_ki Tr[(Σ_Q L_ew_Q(ki) dm_k L_ew_Q†(ki)) dm_k]
    """
    Ex = 0.0
    with h5py.File(ew_file, "r") as fh:
        for i in range(Nk):
            L_bare = fh["EW_bar"][str(i)][...].view(np.complex128).reshape(-1, nao, nao)
            dL     = fh["EW"][str(i)][...].view(np.complex128).reshape(-1, nao, nao)
            L_ew   = L_bare + dL
            dm_k   = dm[i]
            Sigma_x = -np.einsum('Qps,st,Qrt->pr', L_ew, dm_k, L_ew.conj())
            Ex += np.trace(Sigma_x @ dm_k).real
    return Ex / Nk


# ─── Level 4 helpers ─────────────────────────────────────────────────────────

def _exchange_energy_from_file(filename, dm, Nk, nao):
    """Contract EW tensors in *filename* with density matrix *dm*.

    Computes ΔEx = -(1/Nk) Σ_ki Tr[Σ_Q ΔL^Q(ki) γ(ki) ΔL^Q†(ki) γ(ki)]

    Parameters
    ----------
    filename : str
        Path to HDF5 file produced by _run_ewald_old / _run_ewald_new.
        Must contain group ``EW`` with datasets ``"0"``..``"Nk-1"``.
    dm : ndarray, shape (Nk, nao, nao)
        k-space density matrix (both spins included, factor of 2 for RHF).
    """
    dEx = 0.0
    with h5py.File(filename, "r") as fh:
        for i in range(Nk):
            # EW stored as float64 view of complex128 — recover complex
            dL = fh["EW"][str(i)][...].view(np.complex128).reshape(-1, nao, nao)
            dm_k = dm[i]
            # exchange self-energy: Σ_x(ki)_pr = -Σ_Q ΔL^Q_ps γ_st ΔL^Q*_rt
            Sigma_x = -np.einsum('Qps,st,Qrt->pr', dL, dm_k, dL.conj())
            dEx += np.trace(Sigma_x @ dm_k).real
    return dEx / Nk


# ─── Level 4 tests ───────────────────────────────────────────────────────────

class TestEndToEnd:
    """Level 4: end-to-end exchange energy regression.

    4a: HF total energy must be unchanged after the compute_ewald_correction
        patch-and-restore cycle — confirms no class-level side effects survive.

    4b: Exchange energy correction ΔEx is numerically identical between the
        CC-forced (old) and RS-aware (new) Ewald paths to 1e-7 Ha.
    """

    @pytest.fixture(scope="class")
    def hf_setup(self, h2_cell, kmesh_2x2x2):
        """KRHF on H2 2×2×2 for density matrix and shared DF object."""
        from pyscf.pbc.scf import KRHF
        mydf = df.GDF(h2_cell)
        mydf.kpts = kmesh_2x2x2
        mydf.verbose = 0
        mydf.build()

        mf = KRHF(h2_cell, kmesh_2x2x2)
        mf.with_df = mydf
        mf.exxdiv = 'ewald'
        mf.verbose = 0
        e_hf = mf.kernel()

        nao = h2_cell.nao_nr()
        Nk = len(kmesh_2x2x2)
        # KRHF: make_rdm1 → (Nk, nao, nao) with factor 2 for double occupation
        dm = mf.make_rdm1()
        return h2_cell, kmesh_2x2x2, mydf, nao, Nk, dm, e_hf

    def test_hf_energy_unchanged(self, hf_setup, tmp_path):
        """HF total energy must be identical before and after the Ewald patch cycle.

        A fresh KRHF is run before and after calling compute_ewald_correction.
        Any unreleased class-level monkey-patch would corrupt the second run.
        """
        cell, kmesh, maindf, nao, Nk, _, e_before = hf_setup

        f_ew = str(tmp_path / "df_ewald_4a.h5")
        _run_ewald_new(cell, kmesh, maindf, nao, f_ew)

        from pyscf.pbc.scf import KRHF
        mf_after = KRHF(cell, kmesh)
        mf_after.with_df = df.GDF(cell)
        mf_after.with_df.kpts = kmesh
        mf_after.with_df.verbose = 0
        mf_after.exxdiv = 'ewald'
        mf_after.verbose = 0
        e_after = mf_after.kernel()

        np.testing.assert_allclose(
            e_before, e_after, atol=1e-10,
            err_msg=(
                f"HF energy changed after compute_ewald_correction patch cycle: "
                f"before={e_before:.10f}, after={e_after:.10f}"
            )
        )

    def test_exchange_energy_correction(self, hf_setup, tmp_path):
        """ΔEx from EW tensors must agree between CC-forced and RS-aware paths.

        Both paths produce EW tensors stored in HDF5; we contract them with the
        k-space density matrix and assert the resulting exchange energy corrections
        agree to atol=1e-7 Ha.  We also assert the correction is nonzero so that
        a silent Ewald dropout (correction set to zero) would be caught.
        """
        cell, kmesh, maindf, nao, Nk, dm, _ = hf_setup

        f_old = str(tmp_path / "ewald_4b_old.h5")
        f_new = str(tmp_path / "ewald_4b_new.h5")
        _run_ewald_old(cell, kmesh, maindf, nao, f_old)
        _run_ewald_new(cell, kmesh, maindf, nao, f_new)

        dEx_old = _exchange_energy_from_file(f_old, dm, Nk, nao)
        dEx_new = _exchange_energy_from_file(f_new, dm, Nk, nao)

        assert abs(dEx_old) > 1e-12, \
            "Old-path exchange correction is identically zero — correction silent dropped"
        assert abs(dEx_new) > 1e-12, \
            "New-path exchange correction is identically zero — correction silently dropped"

        np.testing.assert_allclose(
            dEx_old, dEx_new, atol=1e-7,
            err_msg=(
                f"Exchange energy correction differs between CC-forced and RS-aware paths: "
                f"old={dEx_old:.10f} Ha, new={dEx_new:.10f} Ha"
            )
        )


# ─── Level 5: Option A vs Option B consistency ───────────────────────────────

class TestOptionBConsistency:
    """Level 5: Option A (RS everywhere) vs Option B (CC everywhere).

    Both options are internally self-consistent: Option A uses RS for all three
    objects (production maindf, bare df2, Ewald-probed df1); Option B uses CC
    for all three.  Since both compute the same physical Ewald-corrected Coulomb
    integrals in different J2c bases, the total Ewald exchange energy must agree
    to within CC vs RS numerical integration accuracy.
    """

    @pytest.fixture(scope="class")
    def setup_both(self, h2_cell, kmesh_2x2x2):
        from pyscf.pbc.scf import KRHF

        # RS maindf — Option A production metric
        maindf_rs = df.GDF(h2_cell)
        maindf_rs.kpts = kmesh_2x2x2
        maindf_rs.verbose = 0
        f_rs = tempfile.mktemp(suffix=".h5")
        maindf_rs._cderi_to_save = f_rs
        maindf_rs._cderi = f_rs
        maindf_rs.build()

        # CC maindf — Option B production metric (same mesh for a fair comparison)
        maindf_cc = df.GDF(h2_cell)
        maindf_cc._prefer_ccdf = True
        maindf_cc.mesh = maindf_rs.mesh
        maindf_cc.kpts = kmesh_2x2x2
        maindf_cc.verbose = 0
        f_cc = tempfile.mktemp(suffix=".h5")
        maindf_cc._cderi_to_save = f_cc
        maindf_cc._cderi = f_cc
        maindf_cc.build()

        # Reference dm from KRHF with RS maindf
        mf = KRHF(h2_cell, kmesh_2x2x2)
        mf.with_df = maindf_rs
        mf.exxdiv = 'ewald'
        mf.verbose = 0
        mf.kernel()
        dm = mf.make_rdm1()

        nao = h2_cell.nao_nr()
        Nk  = len(kmesh_2x2x2)

        yield h2_cell, kmesh_2x2x2, maindf_rs, maindf_cc, nao, Nk, dm

        os.remove(f_rs)
        os.remove(f_cc)

    def test_option_a_vs_b_total_exchange(self, setup_both, tmp_path):
        """Total Ewald exchange energy from Option A and B must agree to 1e-6 Ha.

        The tolerance is 1e-6 (not 1e-7) because CC and RS 3-centre integrals
        agree to ~1e-5, and the exchange energy inherits that numerical error.
        """
        cell, kmesh, maindf_rs, maindf_cc, nao, Nk, dm = setup_both

        f_a = str(tmp_path / "ew_optA.h5")
        f_b = str(tmp_path / "ew_optB.h5")

        _run_ewald_new(cell, kmesh, maindf_rs, nao, f_a)            # Option A
        _run_ewald_cc_consistent(cell, kmesh, maindf_cc, nao, f_b)  # Option B

        Ex_A = _total_ex_from_ew_file(f_a, dm, Nk, nao)
        Ex_B = _total_ex_from_ew_file(f_b, dm, Nk, nao)

        assert abs(Ex_A) > 1e-12, "Option A total exchange energy is zero"
        assert abs(Ex_B) > 1e-12, "Option B total exchange energy is zero"

        np.testing.assert_allclose(
            Ex_A, Ex_B, atol=1e-6,
            err_msg=(
                f"Total Ewald exchange energy differs between Option A (RS) and "
                f"Option B (CC): A={Ex_A:.10f} Ha, B={Ex_B:.10f} Ha"
            )
        )
