"""Opt-in and unsupported profiles must be checked before expensive production."""

import pytest
from green_mbtools.mint import common_utils as comm
from green_mbtools.mint.pyscf_init import pyscf_pbc_init


def parameters(tmp_path):
    return [
        "--a",
        "4 0 0\n0 4 0\n0 0 4",
        "--atom",
        "H 0 0 0\nH 0 0 .74",
        "--nk",
        "1",
        "--basis",
        "sto3g",
        "--output_path",
        str(tmp_path / "input.h5"),
        "--hf_int_path",
        str(tmp_path / "hf"),
        "--int_path",
        str(tmp_path / "corr"),
        "--integral_symmetry_work",
        str(tmp_path / "work"),
    ]


def test_default_remains_legacy(tmp_path):
    assert comm.init_pbc_params(parameters(tmp_path)).integral_symmetry == "legacy"


@pytest.mark.parametrize(
    "option,value",
    [
        ("--x2c", "2"),
        ("--use_j2c_eig_decomposition", "true"),
        ("--finite_size_kind", "gf2"),
        ("--finite_size_kind", "gw"),
        ("--finite_size_kind", "gw_s"),
        ("--finite_size_kind", "coarse_grained"),
    ],
)
def test_unsupported_profile_rejected(tmp_path, option, value):
    args = comm.init_pbc_params(
        parameters(tmp_path) + ["--integral_symmetry", "space_group", option, value]
    )
    with pytest.raises(ValueError, match="Initial SG producer"):
        pyscf_pbc_init(args)
    assert not (tmp_path / "input.h5").exists()


def test_existing_output_is_preserved(tmp_path):
    (tmp_path / "hf").mkdir()
    args = comm.init_pbc_params(
        parameters(tmp_path) + ["--integral_symmetry", "space_group"]
    )
    with pytest.raises(ValueError, match="fresh"):
        pyscf_pbc_init(args)
    assert (tmp_path / "hf").is_dir()
