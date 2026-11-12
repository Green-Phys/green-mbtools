import numpy as np
import pytest
from pyscf.pbc import gto, df
import green_igen.df as legacy
from green_mbtools.mint.integral_symmetry_gauge import build_captured_legacy_ewald


@pytest.mark.parametrize("failure_point", ["_make_j3c", "make_modrho_basis"])
def test_failed_capture_restores_class_ownership_and_module(
    monkeypatch, tmp_path, failure_point
):
    cell = gto.Cell(
        a=np.eye(3) * 4, atom="He 0 0 0", basis={"He": [[0, [1.0, 1.0]]]}, verbose=0
    ).build()
    classes = [df.GDF, legacy.GDF]
    original = [
        ("weighted_coulG" in cls.__dict__, getattr(cls, "weighted_coulG"))
        for cls in classes
    ]
    scipy_original = legacy.scipy

    def fail(*args, **kwargs):
        raise RuntimeError("injected corrected producer failure")

    monkeypatch.setattr(legacy, failure_point, fail)
    for attempt in range(2):
        with pytest.raises(RuntimeError, match="injected"):
            build_captured_legacy_ewald(
                cell,
                cell.make_kpts([1, 1, 1]),
                cell.basis,
                [(0, 0)],
                tmp_path / f"failure-{attempt}.h5",
            )
        assert legacy.scipy is scipy_original
        for cls, (owned, function) in zip(classes, original):
            assert ("weighted_coulG" in cls.__dict__) == owned
            assert getattr(cls, "weighted_coulG") is function
