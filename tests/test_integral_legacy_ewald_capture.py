import numpy as np
import pytest
from pyscf.pbc import gto, df
import green_igen.df as legacy
from green_mbtools.mint import integral_utils
from green_mbtools.mint.integral_symmetry_gauge import build_captured_legacy_ewald


@pytest.mark.parametrize("custom_mesh", [False, True], ids=["default", "custom"])
def test_ewald_kernel_uses_reported_mesh(monkeypatch, tmp_path, custom_mesh):
    cell = gto.Cell(
        a=np.eye(3) * 4, atom="He 0 0 0", basis={"He": [[0, [1.0, 1.0]]]}, verbose=0
    ).build()
    kpts = cell.make_kpts([1, 1, 1])
    default = np.asarray(legacy.GDF(cell, kpts).mesh)
    expected = default + [2, 4, 6] if custom_mesh else default
    seen = []
    kernel = integral_utils.weighted_coulG_ewald

    def record_mesh(mydf, kpt, exx, mesh, omega=None):
        seen.append(np.array(mesh, copy=True))
        return kernel(mydf, kpt, exx, mesh, omega)

    monkeypatch.setattr(integral_utils, "weighted_coulG_ewald", record_mesh)
    provider, _ = build_captured_legacy_ewald(
        cell,
        kpts,
        cell.basis,
        [(0, 0)],
        tmp_path / "q0.h5",
        mesh=expected if custom_mesh else None,
    )
    assert seen
    for mesh in seen:
        np.testing.assert_array_equal(mesh, expected)
    np.testing.assert_array_equal(provider.mesh, expected)


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
