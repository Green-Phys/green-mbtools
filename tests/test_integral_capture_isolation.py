"""The capture wrapper must not change state seen by unrelated threads."""

from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np
import pytest
import scipy
from pyscf.pbc import df, gto
import green_igen.df as legacy

from green_mbtools.mint import integral_symmetry_gauge as capture
from green_mbtools.mint.integral_symmetry_compat import require_ccgdf_compatibility
from green_mbtools.mint.integral_utils import weighted_coulG_ewald

# The injected routine below resolves these through its per-call globals copy.
GDF = legacy.GDF


def test_concurrent_capture_uses_private_classes_and_modules(monkeypatch, tmp_path):
    cell = gto.Cell(
        a=np.eye(3) * 4, atom="He 0 0 0", basis={"He": [[0, [1.0, 1.0]]]}, verbose=0
    ).build()
    old_pyscf, old_legacy, old_scipy = (
        df.GDF.weighted_coulG,
        legacy.GDF.weighted_coulG,
        legacy.scipy,
    )
    rendezvous = threading.Barrier(2)

    def injected(provider, cell, auxcell, pairs, path):
        # Reproduce the real legacy routine's class assignment and construction.
        GDF.weighted_coulG = df.GDF.weighted_coulG
        private = GDF(cell, provider.kpts)
        assert type(private) is not legacy.GDF
        assert private.weighted_coulG.__func__ is weighted_coulG_ewald
        rendezvous.wait(timeout=20)
        assert df.GDF.weighted_coulG is old_pyscf
        assert legacy.GDF.weighted_coulG is old_legacy
        assert legacy.scipy is old_scipy
        scipy.linalg.cholesky(np.eye(auxcell.nao_nr()), lower=True)

    monkeypatch.setattr(legacy, "_make_j3c", injected)
    monkeypatch.setattr(capture, "version", lambda package: "0.2.9")

    def run(index):
        return capture.build_captured_legacy_ewald(
            cell,
            cell.make_kpts([1, 1, 1]),
            cell.basis,
            [(0, 0)],
            tmp_path / f"{index}.h5",
        )[1]

    with ThreadPoolExecutor(max_workers=2) as pool:
        gauges = list(pool.map(run, (0, 1)))
    assert len(gauges) == 2
    assert (
        df.GDF.weighted_coulG is old_pyscf and legacy.GDF.weighted_coulG is old_legacy
    )
    assert legacy.scipy is old_scipy


def test_private_api_shape_is_checked_before_use(monkeypatch):
    from pyscf.pbc.df.gdf_builder import _CCGDFBuilder

    monkeypatch.setattr(_CCGDFBuilder, "gen_uniq_kpts_groups", lambda self: None)
    with pytest.raises(RuntimeError, match="private interface: gen_uniq_kpts_groups"):
        require_ccgdf_compatibility()


@pytest.mark.parametrize("release", ["0.2.7", "0.3.0"])
def test_unsupported_legacy_version_has_clear_failure(monkeypatch, tmp_path, release):
    monkeypatch.setattr(capture, "version", lambda package: release)
    with pytest.raises(RuntimeError, match=r"green-igen >=0\.2\.8,<0\.3"):
        capture.build_captured_legacy_ewald(
            None, None, None, None, tmp_path / "unused.h5"
        )
    assert not list(tmp_path.iterdir())
