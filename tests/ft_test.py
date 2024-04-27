from mbanalysis.ft import construct_rmesh, \
    compute_fourier_coefficients, k_to_real, real_to_k
import numpy as np
import pytest


def test_fourier_transform(data_path):
    """Test function for Fourier Transform functions.
    FT between real and reciprocal space.
    """
    data_dir = pytest.test_data_dir

    # Read test overlap matrices
    Sk10 = np.load(data_dir + '/winter/Sk10.npy')
    Sk6 = np.load(data_dir + '/winter/Sk6.npy')

    # Get k-mesh
    kmesh_scaled_nk10 = np.load(data_dir + '/winter/kmesh_k10.npy')
    kmesh_scaled_nk6 = np.load(data_dir + '/winter/kmesh_k6.npy')

    # Get number of k-points
    nk6 = int(np.cbrt(Sk6.shape[0]))
    nk10 = int(np.cbrt(Sk10.shape[0]))

    # Check Fourier transformation for nk = 6
    rmesh_nk6 = construct_rmesh(nk6, nk6, nk6)
    fkr_nk6, frk_nk6 = compute_fourier_coefficients(
        kmesh_scaled_nk6, rmesh_nk6
    )
    Si6 = k_to_real(frk_nk6, Sk6, [1]*kmesh_scaled_nk6.shape[0])
    Sk6_check = real_to_k(fkr_nk6, Si6)
    diff = Sk6_check - Sk6
    assert np.max(np.abs(diff)) < 1e-8

    # Check Fourier transformation for nk = 10
    rmesh_nk10 = construct_rmesh(nk10, nk10, nk10)
    fkr_nk10, frk_nk10 = compute_fourier_coefficients(
        kmesh_scaled_nk10, rmesh_nk10
    )
    Si10 = k_to_real(frk_nk10, Sk10, [1]*kmesh_scaled_nk10.shape[0])
    Sk10_check = real_to_k(fkr_nk10, Si10)
    diff = Sk10_check - Sk10
    assert np.max(np.abs(diff)) < 1e-8
