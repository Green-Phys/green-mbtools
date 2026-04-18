import shutil
from pathlib import Path

import h5py
import pytest
import numpy as np

import green_mbtools.mint as pymb
from green_mbtools.mint import common_utils as comm


#
# Test function for PySCF initialization
#


def compare_datasets(
    actual_path, expected_path, dataset_names, rtol=1e-6, atol=1e-7
) -> None:
    """Helper to assert that each named dataset in two HDF5 files is allclose."""
    with h5py.File(actual_path, "r") as actual, h5py.File(
        expected_path, "r"
    ) as expected:
        for name in dataset_names:
            a = actual[name][...]
            e = expected[name][...]
            np.testing.assert_allclose(
                a, e, rtol=rtol, atol=atol, err_msg=f"HDF5 dataset '{name}' mismatch"
            )


@pytest.mark.parametrize(
    "extra_flags, subdir",
    [
        (["--basis", "gth-dzvp-molopt-sr", "--pseudo", "gth-pbe"], "UHF"),
        (["--basis", "gth-dzvp-molopt-sr", "--pseudo", "gth-pbe", "--restricted", "1"], "RHF"),
        (["--basis", "gth-dzvp-molopt-sr", "--pseudo", "gth-pbe", "--xc", "PBE"], "UKS"),
        (["--basis", "gth-dzvp-molopt-sr", "--pseudo", "gth-pbe", "--xc", "PBE", "--restricted", "1"], "RKS"),
        (["--basis", "cc-pvdz", "--x2c", "1"], "UHF_sfx2c"),
        (["--basis", "cc-pvdz", "--xc", "PBE", "--x2c", "1"], "UKS_sfx2c"),
        (["--basis", "cc-pvdz", "--x2c", "2"], "GHF"),
        (["--basis", "cc-pvdz", "--xc", "PBE", "--x2c", "2"], "GKS"),
    ],
)
def test_meanfield_variants(data_path, extra_flags, subdir) -> None:
    import os

    test_data_dir = Path(pytest.test_data_dir) / "H2_pbc"
    tmp_dir = Path(__file__).parent / "tmp"

    # ensure a clean slate
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()
    old_cwd = Path.cwd()
    os.chdir(tmp_dir)
    # prepare scratch locations
    output_h5 = tmp_dir / "input.h5"

    try:
        # build parameters
        base_params = [
            "--atom", "H -0.25 -0.25 -0.25\nH  0.25  0.25  0.25",
            "--a", "4.0655, 0.0,    0.0\n0.0,    4.0655, 0.0\n0.0,    0.0,    4.0655\n",
            "--output_path", str(output_h5),
            "--df_int", "0",
            "--nk", "3",
            "--use_j2c_eig_decomposition", "false",
        ]
        params = base_params.copy()
        params.extend(extra_flags)

        # run mean‑field generation
        args = comm.init_pbc_params(params=params)
        pyscf_init = pymb.pyscf_pbc_init(args)
        pyscf_init.mean_field_input()

        # compare key HF datasets
        expected_h5 = test_data_dir / subdir / "input.h5"
        datasets = [
            "HF/Energy",
            "HF/Energy_nuc",
            "HF/Fock-k",
            "HF/S-k",
            "HF/H-k",
            "HF/mo_energy",
        ]
        compare_datasets(output_h5, expected_h5, datasets)
    finally:
        # clean up immediately
        os.chdir(old_cwd)
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_anisotropic_nk_matches_symmetric(data_path) -> None:
    """Verify that --nk 3 3 3 produces the same result as --nk 3."""
    import os

    test_data_dir = Path(pytest.test_data_dir) / "H2_pbc"
    tmp_dir = Path(__file__).parent / "tmp"

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()
    old_cwd = Path.cwd()
    os.chdir(tmp_dir)
    output_h5 = tmp_dir / "input.h5"

    try:
        params = [
            "--atom", "H -0.25 -0.25 -0.25\nH  0.25  0.25  0.25",
            "--a", "4.0655, 0.0,    0.0\n0.0,    4.0655, 0.0\n0.0,    0.0,    4.0655\n",
            "--output_path", str(output_h5),
            "--df_int", "0",
            "--nk", "3", "3", "3",
            "--basis", "gth-dzvp-molopt-sr",
            "--pseudo", "gth-pbe",
            "--use_j2c_eig_decomposition", "false",
        ]

        args = comm.init_pbc_params(params=params)
        pyscf_init = pymb.pyscf_pbc_init(args)
        pyscf_init.mean_field_input()

        expected_h5 = test_data_dir / "UHF" / "input.h5"
        datasets = [
            "HF/Energy",
            "HF/Energy_nuc",
            "HF/Fock-k",
            "HF/S-k",
            "HF/H-k",
            "HF/mo_energy",
        ]
        compare_datasets(output_h5, expected_h5, datasets)
    finally:
        os.chdir(old_cwd)
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.parametrize("bad_nk", [["2", "3"], ["1", "2", "3", "4"]])
def test_invalid_nk_count_raises(bad_nk) -> None:
    """Providing 2 or 4 values to --nk must raise a ValueError."""
    params = [
        "--atom", "H 0 0 0",
        "--a", "5 0 0, 0 5 0, 0 0 5",
        "--basis", "sto-3g",
        "--nk", *bad_nk,
    ]
    with pytest.raises(ValueError, match="--nk must be given 1 or 3 integers"):
        comm.init_pbc_params(params=params)
