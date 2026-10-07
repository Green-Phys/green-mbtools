import numpy as np
import h5py
import pytest

from convert_input_to_native_complex import convert_file, INPUT_DATASETS


def _write_legacy(path, name, arr):
    """Store a complex array the legacy float+2 way, with the marker attr."""
    with h5py.File(path, "w") as f:
        legacy = arr.view(np.float64).reshape(arr.shape + (2,))
        f[name] = legacy
        f[name].attrs["__complex__"] = np.int8(1)
        f.attrs["__green_version__"] = "1.0.0"


def test_convert_file_produces_native_complex(tmp_path):
    path = str(tmp_path / "input.h5")
    arr = (np.arange(2 * 3 * 2 * 2).reshape(2, 3, 2, 2)
           + 1j * np.arange(2 * 3 * 2 * 2).reshape(2, 3, 2, 2)).astype(np.complex128)
    _write_legacy(path, "HF/S-k", arr)

    converted = convert_file(path, ("HF/S-k",))

    assert converted == ["HF/S-k"]
    with h5py.File(path, "r") as f:
        ds = f["HF/S-k"]
        assert ds.dtype == np.complex128
        assert ds.shape == (2, 3, 2, 2)
        assert "__complex__" not in ds.attrs
        assert f.attrs["__green_version__"] == "1.1.0"
        np.testing.assert_array_equal(ds[()], arr)


def test_convert_file_is_idempotent(tmp_path):
    path = str(tmp_path / "input.h5")
    arr = (np.ones((2, 3, 2, 2)) + 1j * np.full((2, 3, 2, 2), 2.0)).astype(np.complex128)
    _write_legacy(path, "HF/H-k", arr)

    first = convert_file(path, ("HF/H-k",))
    second = convert_file(path, ("HF/H-k",))

    assert first == ["HF/H-k"]
    assert second == []  # already native, nothing to do
    with h5py.File(path, "r") as f:
        assert f["HF/H-k"].dtype == np.complex128
        np.testing.assert_array_equal(f["HF/H-k"][()], arr)
        assert f.attrs["__green_version__"] == "1.1.0"


def test_convert_file_handles_dm(tmp_path):
    path = str(tmp_path / "dm.h5")
    arr = (np.arange(2 * 3 * 2 * 2).reshape(2, 3, 2, 2)
           + 1j).astype(np.complex128)
    _write_legacy(path, "HF/dm-k", arr)

    converted = convert_file(path, ("HF/dm-k",))

    assert converted == ["HF/dm-k"]
    with h5py.File(path, "r") as f:
        assert f["HF/dm-k"].dtype == np.complex128
        np.testing.assert_array_equal(f["HF/dm-k"][()], arr)
        assert f.attrs["__green_version__"] == "1.1.0"


def test_convert_file_skips_absent_datasets(tmp_path):
    path = str(tmp_path / "input.h5")
    with h5py.File(path, "w") as f:
        f["HF/S-k"] = np.ones((2, 3, 2, 2), dtype=np.complex128)  # already native
    converted = convert_file(path, INPUT_DATASETS)
    assert converted == []  # Fock-k, H-k absent; S-k already native
    with h5py.File(path, "r") as f:
        assert f.attrs["__green_version__"] == "1.1.0"
