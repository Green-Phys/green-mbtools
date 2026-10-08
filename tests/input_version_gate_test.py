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
