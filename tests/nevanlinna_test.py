from green_mbtools.pesto.analyt_cont import nevan_run
import numpy as np

#
# Test function for Nevanlinna analytic continuation
#


def test_nevan_exe_on_dirac_delta_spectrum():
    """Test basic functionality of Nevanlinna AC, i.e. if we can
    recover the delta peak from imaginary axis data.
    """

    #
    # Test case 1: Dirac delta centered at w = 0
    #

    # Generate imaginary axis data for f(z) = 1 / z
    iw_vals = np.linspace(0.1, 10, 100)
    G_iw = 1 / (1j * iw_vals)
    G_iw = G_iw.reshape((G_iw.shape[0], 1))

    n_real = 101
    w_min = -0.1
    w_max = 0.1
    eta = 0.01

    freqs, A_w = nevan_run(
        G_iw, iw_vals, n_real=n_real,
        w_min=w_min, w_max=w_max, eta=eta, spectral=True
    )

    f_max = freqs[np.argmax(A_w)]
    assert f_max == 0

    #
    # Test case 2: Dirac delta centerd at z0 = 1 + 1e-4j
    #

    # Generate imaginary axis data for f(z) = 1 / (z - z0)
    iw_vals = np.linspace(0.1, 10, 100)
    G_iw = 1 / (1j * iw_vals - (1 + 1e-4j))
    G_iw = G_iw.reshape((G_iw.shape[0], 1))

    n_real = 201
    w_min = -0.5
    w_max = 1.5
    eta = 0.01

    freqs, A_w = nevan_run(
        G_iw, iw_vals, n_real=n_real,
        w_min=w_min, w_max=w_max, eta=eta, spectral=True
    )

    f_max = freqs[np.argmax(A_w)]
    assert f_max == 1.


def test_nevan_exe_on_selfenergy():
    """Test functionality of `nevan_run_selfenergy`.
    """

    # Generate data for complex shifted Gaussian distribution
    # F(z) = exp[-(z - z0)^2 / 2]
    # with z0 = 1 + 1j
    iw_vals = np.linspace(0.1, 10, 100)
    Sigma_iw = 1 / (1j * iw_vals - (1 + 1e-4j))
    Sigma_iw = Sigma_iw.reshape((Sigma_iw.shape[0], 1))

    n_real = 201
    w_min = -0.5
    w_max = 1.5
    eta = 1.

    freqs, Sigma_w = nevan_run(
        Sigma_iw, iw_vals, n_real=n_real,
        w_min=w_min, w_max=w_max, eta=eta, spectral=False
    )

    # check that Sigma_w is complex number
    assert Sigma_w.dtype == complex

    # Find maxima (should be at freqs = 1)
    f_max = freqs[np.argmax(np.abs(Sigma_w))]
    assert f_max == 1.
