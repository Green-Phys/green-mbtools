# tests/migrate_test.py
import os
import numpy as np
import h5py
import pytest

from green_mbtools.mint.migrate import detect_version, _to_native_complex, INPUT_DATASETS

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")


def test_detect_grid_legacy():
    assert detect_version(os.path.join(DATA, "H2_GW_legacy", "input.h5")) == "grid-legacy"


def test_detect_111():
    # H2_GW/input.h5 was migrated to native complex (1.1.0) by the earlier work
    assert detect_version(os.path.join(DATA, "H2_GW", "input.h5")) == "1.1.0"


def test_detect_unrecognized_raises(tmp_path):
    p = str(tmp_path / "junk.h5")
    with h5py.File(p, "w") as f:
        f["foo"] = np.zeros(3)
    with pytest.raises(ValueError):
        detect_version(p)


def _write_legacy(path, name, arr):
    with h5py.File(path, "w") as f:
        f[name] = arr.view(np.float64).reshape(arr.shape + (2,))
        f[name].attrs["__complex__"] = np.int8(1)


def test_to_native_complex_core(tmp_path):
    p = str(tmp_path / "x.h5")
    arr = (np.arange(24).reshape(2, 3, 2, 2) + 1j).astype(np.complex128)
    _write_legacy(p, "HF/S-k", arr)
    with h5py.File(p, "a") as f:
        out = _to_native_complex(f, ("HF/S-k",))
    assert out == ["HF/S-k"]
    with h5py.File(p, "r") as f:
        assert f["HF/S-k"].dtype == np.complex128
        assert "__complex__" not in f["HF/S-k"].attrs
        np.testing.assert_array_equal(f["HF/S-k"][()], arr)


def test_detect_100_from_git_fixture(tmp_path):
    # A real 1.0.0 file: symmetry/ datagroup + float+2 matrices.
    import subprocess
    p = str(tmp_path / "v100.h5")
    with open(p, "wb") as fh:
        subprocess.run(
            ["git", "show", "3492197~1:tests/test_data/H2_GW/input.h5"],
            stdout=fh, check=True, cwd=os.path.dirname(DATA),
        )
    assert detect_version(p) == "1.0.0"


def test_v100_to_110(tmp_path):
    import shutil
    from green_mbtools.mint.migrate import _v100_to_110, _bump_meta_version
    src = os.path.join(DATA, "migrate", "v100_input.h5")
    work = str(tmp_path / "input.h5")
    shutil.copy(src, work)
    # a stand-in integral meta.h5
    intdir = tmp_path / "df_int"
    intdir.mkdir()
    with h5py.File(intdir / "meta.h5", "w") as m:
        m.attrs["__green_version__"] = "1.0.0"

    # expected values via the float+2 decode of the source
    with h5py.File(src, "r") as f:
        exp_S = f["HF/S-k"][()].view(np.complex128).reshape(f["HF/S-k"].shape[:-1])

    _v100_to_110(work, dm_file=None, int_paths=(str(intdir),))

    with h5py.File(work, "r") as f:
        assert f.attrs["__green_version__"] == "1.1.0"
        for name in ("HF/Fock-k", "HF/S-k", "HF/H-k"):
            assert f[name].dtype == np.complex128
            assert "__complex__" not in f[name].attrs
        np.testing.assert_array_equal(f["HF/S-k"][()], exp_S)
    with h5py.File(intdir / "meta.h5", "r") as m:
        assert m.attrs["__green_version__"] == "1.1.0"
