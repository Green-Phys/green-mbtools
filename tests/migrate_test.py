# tests/migrate_test.py
import os
import numpy as np
import h5py
import pytest

from green_mbtools.mint.migrate import detect_version, _to_native_complex, INPUT_DATASETS

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")


def test_detect_grid_legacy():
    assert detect_version(os.path.join(DATA, "H2_GW_legacy", "input.h5")) is None


def test_detect_111():
    # H2_GW/input.h5 was migrated to native complex (1.1.0) by the earlier work
    assert detect_version(os.path.join(DATA, "H2_GW", "input.h5")) == "1.1.0"


def test_detect_unversioned_returns_none(tmp_path):
    p = str(tmp_path / "junk.h5")
    with h5py.File(p, "w") as f:
        f["foo"] = np.zeros(3)
    assert detect_version(p) is None


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
        # idempotent: a second pass finds it already complex and skips it;
        # an absent name is skipped too.
        again = _to_native_complex(f, ("HF/S-k", "HF/Fock-k"))
    assert out == ["HF/S-k"]
    assert again == []
    with h5py.File(p, "r") as f:
        assert f["HF/S-k"].dtype == np.complex128
        assert "__complex__" not in f["HF/S-k"].attrs
        np.testing.assert_array_equal(f["HF/S-k"][()], arr)


def test_detect_100():
    # A real 1.0.0 file: symmetry/ datagroup + float+2 matrices.
    assert detect_version(os.path.join(DATA, "migrate", "v100_input.h5")) == "1.0.0"


@pytest.mark.parametrize("dtype,shape,marker", [
    (np.float32, (2, 2, 2), 1),
    (np.int64, (2, 2, 2), 1),
    (np.float64, (2, 2, 3), 1),
    (np.float64, (), 1),
    (np.float64, (2, 2, 2), None),
])
def test_invalid_legacy_storage_is_not_modified(tmp_path, dtype, shape, marker):
    from green_mbtools.mint.migrate import migrate

    src = tmp_path / "input.h5"
    valid = np.ones((1, 1, 2, 2), dtype=np.complex128)
    _write_legacy(src, "HF/Fock-k", valid)
    with h5py.File(src, "a") as f:
        f.attrs["__green_version__"] = "1.0.0"
        f["HF/S-k"] = np.zeros(shape, dtype=dtype)
        if marker is not None:
            f["HF/S-k"].attrs["__complex__"] = marker
        with pytest.raises(ValueError, match="HF/S-k.*expected float64"):
            _to_native_complex(f, INPUT_DATASETS)
        # A valid earlier dataset must not be converted before rejection.
        assert f["HF/Fock-k"].dtype == np.float64
        assert f["HF/Fock-k"].attrs["__complex__"] == 1
        assert f["HF/S-k"].dtype == np.dtype(dtype)
        assert f["HF/S-k"].shape == shape

    before = src.read_bytes()
    out = tmp_path / "output.h5"
    for in_place in (False, True):
        with pytest.raises(ValueError, match="HF/S-k"):
            migrate(src, output=out, in_place=in_place)
        assert src.read_bytes() == before
        assert not out.exists()


@pytest.mark.parametrize("version", ["0.2.4", "0.2.5", "0.3.0", "1.0.0", "1.1.0", "2.0.0"])
def test_detect_returns_stored_version(tmp_path, version):
    src = tmp_path / "input.h5"
    with h5py.File(src, "w") as f:
        f.attrs["__green_version__"] = version
    assert detect_version(src) == version


def test_migrate_rejects_unknown_version(tmp_path):
    from green_mbtools.mint.migrate import migrate

    src = tmp_path / "input.h5"
    with h5py.File(src, "w") as f:
        f.attrs["__green_version__"] = "2.0.0"
    out = tmp_path / "output.h5"
    with pytest.raises(ValueError, match="Detected version '2.0.0'"):
        migrate(src, output=out)
    assert not out.exists()


def test_migrate_rejects_unversioned_input(tmp_path):
    import shutil
    from green_mbtools.mint.migrate import migrate

    src = tmp_path / "input.h5"
    shutil.copy(os.path.join(DATA, "H2_GW_legacy", "input.h5"), src)
    before = src.read_bytes()
    out = tmp_path / "output.h5"
    for in_place in (False, True):
        with pytest.raises(ValueError, match="Unversioned.*migrate_legacy_to_green.py"):
            migrate(src, output=out, in_place=in_place)
        assert src.read_bytes() == before
        assert not out.exists()


@pytest.mark.parametrize("target", ["0.2.4", "0.3.0", "grid-legacy", "2.0.0"])
def test_migrate_rejects_unsupported_target(tmp_path, target):
    from green_mbtools.mint.migrate import migrate

    src = os.path.join(DATA, "migrate", "v100_input.h5")
    out = tmp_path / "output.h5"
    with pytest.raises(ValueError, match="Unknown target version"):
        migrate(src, output=out, target=target)
    assert not out.exists()


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


def test_grid_to_100(tmp_path):
    import shutil
    from green_mbtools.mint.migrate import _grid_to_100, detect_version
    from green_mbtools.mint.integral_utils import integrals_grid

    src = os.path.join(DATA, "H2_GW_legacy", "input.h5")
    work = str(tmp_path / "input.h5")
    shutil.copy(src, work)

    with h5py.File(src, "r") as f:
        g_index = f["grid/index"][()]
        g_irlist = f["grid/ir_list"][()]
        g_conj = f["grid/conj_list"][()]
        g_kmesh = f["grid/k_mesh"][()]
        nso = f["HF/S-k"].shape[2]
        # full-BZ k-count derived from grid/index (shape = (nk_full,)),
        # NOT from HF/nk which is the per-axis mesh dimension.
        nk_full = len(g_index)

    _grid_to_100(work)

    assert detect_version(work) == "1.0.0"
    with h5py.File(work, "r") as f:
        assert f.attrs["__green_version__"] == "1.0.0"
        assert "grid" not in f
        k = f["symmetry/k"]
        np.testing.assert_array_equal(k["bz2ibz"][()], g_index)
        np.testing.assert_array_equal(k["ibz2bz"][()], g_irlist)
        np.testing.assert_array_equal(k["tr_conj"][()], g_conj.astype(k["tr_conj"].dtype))
        assert int(k["n_stars"][()]) == len(g_irlist)
        # identity AO transforms, full shape (nk_full, nso, nso)
        ao = k["k_sym_transform_ao"][()]
        assert ao.shape == (nk_full, nso, nso) and ao.dtype == np.complex128
        np.testing.assert_array_equal(ao, np.broadcast_to(np.eye(nso), (nk_full, nso, nso)))
        # symmetry/k/nk must equal the full-BZ count, not the per-axis dim
        assert int(k["nk"][()]) == nk_full
        assert ao.shape[0] == len(k["bz2ibz"][()])
        # a sample star: full-BZ indices whose rep is ibz2bz[1]
        exp_star1 = np.sort(np.where(g_index == g_irlist[1])[0])
        np.testing.assert_array_equal(np.sort(k["stars"]["1"][()]), exp_star1)
        # Fock/S/H still float+2 at the 1.0.0 stage
        assert f["HF/S-k"].dtype == np.float64 and f["HF/S-k"].shape[-1] == 2
        # pairs match a fresh integrals_grid computation
        # legacy file stores cell under 'Cell' key, not 'mol' (pyscf load_cell expects 'mol')
        import pyscf.pbc.gto
        with h5py.File(src, "r") as fc:
            cell_str = fc["Cell"][()]
        cell = pyscf.pbc.gto.loads(cell_str)
        kptij_idx, kij_conj, kij_trans, kpair_irre, num_kpair, _, _ = integrals_grid(cell, g_kmesh)
        np.testing.assert_array_equal(f["symmetry/pairs/num_kpair_stored"][()], num_kpair)
        np.testing.assert_array_equal(f["symmetry/pairs/kpair_irre_list"][()], kpair_irre)
        # HF/nk must be the full-BZ count after migration, not the per-axis dim
        assert f["HF/nk"][()] == nk_full


def test_grid_to_100_missing_cell_raises(tmp_path):
    import shutil
    from green_mbtools.mint.migrate import _grid_to_100
    src = os.path.join(DATA, "H2_GW_legacy", "input.h5")
    work = str(tmp_path / "input.h5")
    shutil.copy(src, work)
    with h5py.File(work, "a") as f:
        if "Cell" in f:
            del f["Cell"]
    with pytest.raises(ValueError, match="Cell"):
        _grid_to_100(work)


@pytest.mark.parametrize("version", ["0.2.4", "0.2.5", "0.3.0"])
@pytest.mark.parametrize("target", ["1.0.0", "1.1.0"])
def test_migrate_grid_end_to_end(tmp_path, version, target):
    import shutil, types
    from green_mbtools.mint.migrate import migrate, detect_version
    from green_mbtools.mint.seet_init import seet_init
    legacy = os.path.join(DATA, "H2_GW_legacy")
    # Use the dedicated fixture that includes iter14/Sigma1; the shared
    # H2_GW_legacy/sim.h5 is kept pristine for ir_test.py and others.
    migrate_data = os.path.join(DATA, "migrate")
    work = tmp_path / "work"
    work.mkdir()
    shutil.copy(os.path.join(legacy, "input.h5"), work / "input.h5")
    shutil.copy(os.path.join(migrate_data, "legacy_sim.h5"), work / "sim.h5")
    out = str(work / "input_migrated.h5")

    with h5py.File(work / "input.h5", "a") as f:
        f.attrs["__green_version__"] = version

    returned = migrate(str(work / "input.h5"), output=out, target=target)

    assert returned == out
    assert detect_version(out) == target
    assert detect_version(work / "input.h5") == version
    with h5py.File(out, "r") as f:
        assert "grid" not in f
        assert "symmetry" in f
        assert f["HF/S-k"].dtype == (np.float64 if target == "1.0.0" else np.complex128)
    if target == "1.0.0":
        return
    # the migrated file is accepted by the 1.1.0 reader and expands to full BZ
    args = types.SimpleNamespace(input_file=out, gf2_input_file=str(work / "sim.h5"))
    F, S, T, dm, dm_s, kmesh, kmesh_sc = seet_init(args).get_input_data()
    assert S.shape[1] == 27


def test_migrate_already_111_is_noop(tmp_path):
    import shutil
    from green_mbtools.mint.migrate import migrate
    src = os.path.join(DATA, "H2_GW", "input.h5")  # already 1.1.0
    work = str(tmp_path / "input.h5")
    shutil.copy(src, work)
    before = open(work, "rb").read()
    migrate(work, in_place=True)
    assert open(work, "rb").read() == before


def test_migrate_newfile_refuses_clobber(tmp_path):
    import shutil
    from green_mbtools.mint.migrate import migrate
    src = os.path.join(DATA, "migrate", "v100_input.h5")
    work = str(tmp_path / "input.h5"); shutil.copy(src, work)
    existing = str(tmp_path / "out.h5")
    open(existing, "w").close()
    with pytest.raises(FileExistsError):
        migrate(work, output=existing)
    # original untouched
    with h5py.File(work, "r") as f:
        assert f.attrs["__green_version__"] == "1.0.0"


def test_cli_smoke(tmp_path):
    import shutil, subprocess, sys
    src = os.path.join(DATA, "H2_GW_legacy", "input.h5")
    work = str(tmp_path / "input.h5"); shutil.copy(src, work)
    with h5py.File(work, "a") as f:
        f.attrs["__green_version__"] = "0.2.4"
    out = str(tmp_path / "out.h5")
    subprocess.run([sys.executable, "-m", "green_mbtools.mint.migrate",
                    "--input", work, "--output", out], check=True)
    from green_mbtools.mint.migrate import detect_version
    assert detect_version(out) == "1.1.0"


def _write_v100_dm(path, shape=(2, 2, 2, 2)):
    """Write legacy (ns, nk, nao, nao, 2) storage for a 4D density matrix."""
    arr = np.ones(shape, dtype=np.complex128)
    with h5py.File(path, "w") as f:
        f["HF/dm-k"] = arr.view(np.float64).reshape(arr.shape + (2,))
        f["HF/dm-k"].attrs["__complex__"] = np.int8(1)
        f.attrs["__green_version__"] = "1.0.0"


def test_migrate_already_111_upgrades_companions(tmp_path):
    import shutil
    from green_mbtools.mint.migrate import migrate

    input_file = tmp_path / "input.h5"
    shutil.copy(os.path.join(DATA, "H2_GW", "input.h5"), input_file)
    with h5py.File(input_file, "r") as f:
        matrices = {name: f[name][()] for name in INPUT_DATASETS}
    dm_file = tmp_path / "dm.h5"
    _write_v100_dm(dm_file)
    int_dir = tmp_path / "df_int"
    int_dir.mkdir()
    with h5py.File(int_dir / "meta.h5", "w") as f:
        f.attrs["__green_version__"] = "1.0.0"

    migrate(input_file, in_place=True, dm_file=dm_file, int_paths=(int_dir,))

    with h5py.File(input_file, "r") as f:
        assert f.attrs["__green_version__"] == "1.1.0"
        for name, expected in matrices.items():
            np.testing.assert_array_equal(f[name][()], expected)
    with h5py.File(dm_file, "r") as f:
        assert f.attrs["__green_version__"] == "1.1.0"
        assert f["HF/dm-k"].dtype == np.complex128
        assert f["HF/dm-k"].shape == (2, 2, 2, 2)
        assert "__complex__" not in f["HF/dm-k"].attrs
        np.testing.assert_array_equal(f["HF/dm-k"][()], np.ones((2, 2, 2, 2)))
    with h5py.File(int_dir / "meta.h5", "r") as f:
        assert f.attrs["__green_version__"] == "1.1.0"


def test_newfile_migrate_dm_nondestructive(tmp_path):
    """new-file mode must not mutate the source dm.h5."""
    import shutil
    from green_mbtools.mint.migrate import migrate

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    src_input = str(src_dir / "input.h5")
    shutil.copy(os.path.join(DATA, "migrate", "v100_input.h5"), src_input)

    src_dm = str(src_dir / "dm.h5")
    _write_v100_dm(src_dm)

    # record the raw bytes of the source dm before migration
    src_dm_bytes_before = open(src_dm, "rb").read()

    out_input = str(out_dir / "input.h5")
    migrate(src_input, output=out_input, dm_file=src_dm)

    # source dm must be byte-for-byte unchanged
    assert open(src_dm, "rb").read() == src_dm_bytes_before, (
        "source dm.h5 was mutated by new-file migration"
    )
    # migrated dm must be beside the output
    out_dm = str(out_dir / "dm.h5")
    assert os.path.exists(out_dm), "migrated dm.h5 not created beside output"
    with h5py.File(out_dm, "r") as f:
        assert f["HF/dm-k"].dtype == np.complex128
        assert f["HF/dm-k"].shape == (2, 2, 2, 2)
        assert "__complex__" not in f["HF/dm-k"].attrs
        assert f.attrs["__green_version__"] == "1.1.0"
    # the source dm still has the float+2 layout and 1.0.0 version
    with h5py.File(src_dm, "r") as f:
        assert f["HF/dm-k"].dtype == np.float64
        assert f["HF/dm-k"].shape == (2, 2, 2, 2, 2)
        assert "__complex__" in f["HF/dm-k"].attrs
        assert f.attrs["__green_version__"] == "1.0.0"


@pytest.mark.parametrize("in_place", [False, True])
def test_migrate_invalid_companion_preserves_files(tmp_path, in_place):
    import shutil
    from green_mbtools.mint.migrate import migrate

    src_input = tmp_path / "input.h5"
    shutil.copy(os.path.join(DATA, "migrate", "v100_input.h5"), src_input)
    src_dm = tmp_path / "dm.h5"
    _write_v100_dm(src_dm)
    with h5py.File(src_dm, "a") as f:
        del f["HF/dm-k"].attrs["__complex__"]
    input_before = src_input.read_bytes()
    dm_before = src_dm.read_bytes()
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    with pytest.raises(ValueError, match="HF/dm-k.*expected float64"):
        migrate(src_input, output=out_dir / "input.h5", dm_file=src_dm,
                in_place=in_place)

    assert src_input.read_bytes() == input_before
    assert src_dm.read_bytes() == dm_before
    assert list(out_dir.iterdir()) == []


def test_newfile_migrate_dm_refuses_clobber_unless_forced(tmp_path):
    from pathlib import Path
    from green_mbtools.mint.migrate import migrate

    src_input = Path(DATA) / "migrate" / "v100_input.h5"
    src_dm = tmp_path / "dm.h5"
    _write_v100_dm(src_dm)
    input_before = src_input.read_bytes()
    dm_before = src_dm.read_bytes()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out_input = out_dir / "input.h5"
    out_dm = out_dir / "dm.h5"
    existing_dm = b"Existing destination must be preserved"
    out_dm.write_bytes(existing_dm)

    with pytest.raises(FileExistsError, match=r"dm\.h5.*force=True"):
        migrate(src_input, output=out_input, dm_file=src_dm)
    assert out_dm.read_bytes() == existing_dm
    assert not out_input.exists()
    assert src_input.read_bytes() == input_before
    assert src_dm.read_bytes() == dm_before

    migrate(src_input, output=out_input, dm_file=src_dm, force=True)
    assert detect_version(out_input) == "1.1.0"
    with h5py.File(out_dm, "r") as f:
        assert f.attrs["__green_version__"] == "1.1.0"
        assert f["HF/dm-k"].dtype == np.complex128
        assert f["HF/dm-k"].shape == (2, 2, 2, 2)
        assert "__complex__" not in f["HF/dm-k"].attrs
        np.testing.assert_array_equal(f["HF/dm-k"][()], np.ones((2, 2, 2, 2)))
    assert src_input.read_bytes() == input_before
    assert src_dm.read_bytes() == dm_before


def test_newfile_migrate_dm_same_dir_raises(tmp_path):
    """new-file mode with output in same dir as source dm must raise ValueError."""
    import shutil
    from green_mbtools.mint.migrate import migrate

    src_input = str(tmp_path / "input.h5")
    shutil.copy(os.path.join(DATA, "migrate", "v100_input.h5"), src_input)

    src_dm = str(tmp_path / "dm.h5")
    _write_v100_dm(src_dm)

    # output lives in the same directory as the source dm -> should raise
    out_input = str(tmp_path / "input_migrated.h5")
    with pytest.raises(ValueError, match="same as the source dm directory"):
        migrate(src_input, output=out_input, dm_file=src_dm)
