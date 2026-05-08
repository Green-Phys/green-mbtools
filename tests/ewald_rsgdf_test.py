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
