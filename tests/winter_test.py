import numpy as np
from mbanalysis.src.winter import interpolate


# TODO: Add test functions for interpolate_G

def test_winter():
    """Test function for Wannier interpolation.
    """

    # Load test-data files
    Sk10 = np.load('test_data/Sk10.npy')
    Sk10 = Sk10.reshape((1,) + Sk10.shape)
    kmesh_scaled_nk10 = np.load('test_data/kmesh_k10.npy')
    Sk6 = np.load('test_data/Sk6.npy')
    Sk6 = Sk6.reshape((1,) + Sk6.shape)
    kmesh_scaled_nk6 = np.load('test_data/kmesh_k6.npy')

    # Perform Wannier interpolation
    # Here, we go from 6x6x6 k-mesh to 10x10x10 k-mesh
    # We have exact data for the latter
    Sk10_inter = interpolate(
        Sk6, kmesh_scaled_nk6, kmesh_scaled_nk10, hermi=True, debug=True
    )
    diff = Sk10_inter - Sk10
    ref_diff = 0.00038569290459276347
    diff_max = np.max(np.abs(diff))
    assert np.abs(diff_max - ref_diff) < 1e-5
    print(
        "Largest difference between the exact and the interpolated one: ",
        np.max(np.abs(diff))
    )
    print("Reference value is ", 0.00038569290459276347)
