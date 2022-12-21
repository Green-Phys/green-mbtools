from mbanalysis.src.analyt_cont import nevan_run
from mbanalysis.src.hardy import sobolev_wrapper
import numpy as np
import os


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

    freqs, A_w = nevan_run(
        G_iw, iw_vals, outdir=outdir, n_real=n_real,
        w_min=w_min, w_max=w_max, eta=eta, green=True
    )

    wkdir = os.path.abspath(os.getcwd())
    os.chdir(wkdir + '/DiracNevan/0')

    n_basis = 15
    n_params = 4 * 15
    params = np.zeros((2 * n_basis), dtype=np.complex128)
    norm = sobolev_wrapper(
        params, coeff_file='coeff.txt', n_real=n_real,
        w_min=w_min, w_max=w_max, eta=eta
    )

    os.chdir(wkdir)

    assert norm > 0
