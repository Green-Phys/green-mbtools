import pytest
from green_mbtools.version import require_input_version


@pytest.mark.parametrize("version", ["1.1.0", "1.1.1", "2.0.0"])
def test_require_input_version_accepts_new(version):
    require_input_version(version)  # must not raise


@pytest.mark.parametrize("version", ["1.0.0", "1.0.9", "0.9"])
def test_require_input_version_rejects_old(version):
    with pytest.raises(ValueError):
        require_input_version(version)


def test_require_input_version_rejects_missing():
    with pytest.raises(ValueError):
        require_input_version(None)


import os
import types


def _legacy_args():
    legacy_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_data", "H2_GW_legacy",
    )
    return types.SimpleNamespace(
        input_file=os.path.join(legacy_dir, "input.h5"),
        gf2_input_file=os.path.join(legacy_dir, "sim.h5"),
    )


def test_seet_init_rejects_legacy_input():
    from green_mbtools.mint.seet_init import seet_init
    with pytest.raises(ValueError):
        seet_init(_legacy_args()).get_input_data()


def test_rejection_message_points_to_migrate():
    with pytest.raises(ValueError) as exc:
        require_input_version("1.0.0")
    assert "green_mbtools.mint.migrate" in str(exc.value)


@pytest.mark.parametrize("version", [None, "1.0.0"])
def test_initialize_mb_post_rejects_old_input_before_reading_datasets(tmp_path, version):
    import h5py
    from green_mbtools.pesto.mb import initialize_MB_post

    input_path = tmp_path / "input.h5"
    with h5py.File(input_path, "w") as f:
        if version is not None:
            f.attrs["__green_version__"] = version
    data = os.path.join(os.path.dirname(__file__), "test_data")
    with pytest.raises(ValueError, match="green_mbtools.mint.migrate"):
        initialize_MB_post(
            os.path.join(data, "H2_GW", "sim.h5"), input_path, ir_file=None
        )
