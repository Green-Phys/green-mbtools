from mbanalysis.src.analyt_cont import nevan_run
from mbanalysis.src.hardy import sobolev_wrapper, optimize, hardy_optimization
import numpy as np
import os
import pytest


#
# Test functions for Hardy optimization on Nevanlinna
#

def test_sobolev_wrapper_on_dirac_delta():
    """Test the functionality of sobolev wrapper on dirac delta spectrum.
    """

    #
    # Test Case: Dirac delta centered at w = 1
    #

    # Generate imaginary axis data for f(z) = 1 / (z - 1)
    iw_vals = np.linspace(0.1, 10, 100)
    G_iw = 1 / (1j * iw_vals - 1)
    G_iw = G_iw.reshape((G_iw.shape[0], 1))

    outdir = 'DiracNevan'
    n_real = 101
    w_min = 0.5
    w_max = 1.5
    eta = 0.01

    _, _ = nevan_run(
        G_iw, iw_vals, outdir=outdir, n_real=n_real,
        w_min=w_min, w_max=w_max, eta=eta, green=True
    )

    wkdir = os.path.abspath(os.getcwd())
    os.chdir(wkdir + '/DiracNevan/0')

    n_basis = 15
    params = np.zeros((2 * n_basis), dtype=np.complex128)
    norm = sobolev_wrapper(
        params, coeff_file='coeff.txt', n_real=n_real,
        w_min=w_min, w_max=w_max, eta=eta
    )

    os.chdir(wkdir)

    assert norm > 0


def test_optimize():
    """Test optimize function, that runs Hardy optimization for a single
    spin, k, and AO index.
    """
    # parameters from DiracNevan in test_sobolev_wrapper_on_dirac_delta
    n_real = 101
    w_min = 0.5
    w_max = 1.5
    eta = 0.01
    wkdir = os.path.abspath(os.getcwd())
    os.chdir(wkdir + '/DiracNevan/0')
    n_basis = 15

    # initialize parameters and check norm
    params_in = np.zeros((4 * n_basis), dtype=np.float64)
    params_cmplx = params_in[0::2] + 1j * params_in[1::2]
    norm_in = sobolev_wrapper(
        params_cmplx, coeff_file='coeff.txt', n_real=n_real,
        w_min=w_min, w_max=w_max, eta=eta
    )
    print(norm_in)

    # perform optimization and get norm
    params_out = optimize(
        params_in, tol=1e-5, max_iter=20000, n_real=n_real, w_min=w_min,
        w_max=w_max, eta=eta, lagr=1e-5
    )
    params_out = params_out[0::2] + 1j * params_out[1::2]
    norm_out = sobolev_wrapper(
        params_out, n_real=n_real, w_min=w_min, w_max=w_max, eta=eta
    )
    print(norm_out)

    # change back to test directory
    os.chdir(wkdir)

    assert norm_out <= norm_in


@pytest.mark.skip(reason='Hardy optimization too slow')
def test_hardy_optimization_function():
    """Test if Hardy optimization function runs without error.
    The output is rather difficult to check. But, we will at least verify
    that the peak location is correct.
    """
    # parameters from DiracNevan in test_sobolev_wrapper_on_dirac_delta
    n_real = 101
    w_min = 0.5
    w_max = 1.5
    eta = 0.01
    nevan_dir = 'DiracNevan'

    freqs, A_w = hardy_optimization(
        nevanlinna_dir=nevan_dir, n_real=n_real,
        w_min=w_min, w_max=w_max, eta=eta
    )

    # find max for frequency
    f_max = freqs[np.argmax(A_w)]
    assert f_max == 1.
