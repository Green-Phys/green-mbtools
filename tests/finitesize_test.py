import numpy as np
import h5py
import green_mbtools.mint as pymb
import pytest

from green_mbtools.mint import gdf_s_metric as gdf_S
from green_mbtools.mint import common_utils as comm
from green_mbtools.mint import integral_utils as int_utils
from green_mbtools.mint.pyscf_init import pyscf_init, pyscf_pbc_init

def test_finitesize(data_path):
    """Test function for finite-size correction of two-body integrals.
    """
    data_dir = pytest.test_data_dir

    # Load test-data files
    with h5py.File(data_dir + '/H2_int/AqQ.h5', "r") as f:
        test_data = f["AqQ"][()]
    params = ["--a", "0.0,  2.7155, 2.7155\n2.7155, 0.0,  2.7155\n2.7155, 2.7155, 0.0", "--atom", "H 0.0  0.0  0.0\nH 1.35775 1.35775 1.35775", 
              "--nk", "2", "--basis", "sto3g", "--keep_cderi", "false", "--output_path", "/tmp/input.h5",
              "--int_path", "/tmp/df_int/", "--hf_int_path", "/tmp/df_hf_int/",
              "--job", "ewald_corr", "--finite_size_kind", "gw",
              "--use_j2c_eig_decomposition", "false", "--space_symm", "false"]
    args = comm.init_pbc_params(params=params)
    print(args)

    pymb_system = pymb.pyscf_pbc_init(args)
    pymb_system.compute_twobody_finitesize_correction()
    print("Computed finite-size correction")
    with h5py.File('/tmp/df_hf_int/AqQ.h5', "r") as f:
        data = f["AqQ"][()]

    import shutil
    import os
    shutil.rmtree('/tmp/df_int')
    shutil.rmtree('/tmp/df_hf_int')
    assert np.allclose(test_data, data, atol=1e-7)
    
if __name__ == "__main__":
    test_finitesize("/home/munkhw/lib/green-mbtools/tests/test_data")
