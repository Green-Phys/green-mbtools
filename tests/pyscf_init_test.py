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
    actual_path,
    expected_path,
    dataset_names,
    rtol = 1e-6,
    atol = 1e-7
) -> None:
    """Helper to assert that each named dataset in two HDF5 files is allclose."""
    with h5py.File(actual_path, "r") as actual, h5py.File(expected_path, "r") as expected:
        for name in dataset_names:
            a = actual[name][...]
            e = expected[name][...]
            np.testing.assert_allclose(
                a, e, rtol=rtol, atol=atol,
                err_msg=f"HDF5 dataset '{name}' mismatch"
            )

@pytest.mark.parametrize("extra_flags, subdir", [
    ([], "UHF"),
    (["--restricted", "1"], "RHF"),
    (["--x2c", "2"], "GHF"),
    (["--xc", "lda"], "UKS"),
    (["--xc", "lda", "--restricted", "1"], "RKS"),
    (["--xc", "lda", "--x2c", "2"], "GKS"),
])
def test_meanfield_variants(data_path, extra_flags, subdir) -> None:
    import os
    test_data_dir = Path(pytest.test_data_dir) / "H2_mol"
    tmp_dir = Path(__file__).parent / "tmp"

    # ensure a clean slate
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()
    old_cwd = Path.cwd()
    os.chdir(tmp_dir)
    # prepare scratch locations 
    output_h5 = tmp_dir / "input.h5"
    df_hf_int = tmp_dir / "df_hf_int"

    try:
        # build parameters 
        base_params = [
            "--atom",        "H -0.25 -0.25 -0.25\nH  0.25  0.25  0.25",
            "--basis",       "sto3g",
            "--output_path", str(output_h5),
            "--int_path",    str(df_hf_int),
            "--hf_int_path", str(df_hf_int),
        ]
        params = base_params.copy()
        params.extend(extra_flags)

        # run mean‑field generation 
        args       = comm.init_mol_params(params=params)
        pyscf_init = pymb.pyscf_mol_init(args)
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