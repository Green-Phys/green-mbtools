from mbanalysis.src.analyt_cont import nevan_run
from mbanalysis.src.analyt_cont import es_nevan_run
from mbanalysis.src.orth import sao_orth
import numpy as np
import pytest

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

    outdir = 'DiracNevan'
    n_real = 101
    w_min = -0.1
    w_max = 0.1
    eta = 0.01

    freqs, A_w = nevan_run(
        G_iw, iw_vals, outdir=outdir, n_real=n_real,
        w_min=w_min, w_max=w_max, eta=eta, green=True
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

    outdir = 'DiracNevan'
    n_real = 201
    w_min = -0.5
    w_max = 1.5
    eta = 0.01

    freqs, A_w = nevan_run(
        G_iw, iw_vals, outdir=outdir, n_real=n_real,
        w_min=w_min, w_max=w_max, eta=eta, green=True
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

    outdir = 'SigmaNevan'
    n_real = 201
    w_min = -0.5
    w_max = 1.5
    eta = 1.

    freqs, Sigma_w = nevan_run(
        Sigma_iw, iw_vals, outdir=outdir, n_real=n_real,
        w_min=w_min, w_max=w_max, eta=eta, green=False
    )

    # check that Sigma_w is complex number
    assert Sigma_w.dtype == complex

    # Find maxima (should be at freqs = 1)
    f_max = freqs[np.argmax(np.abs(Sigma_w))]
    assert f_max == 1.


def test_pes_nevan_exe_on_dirac_delta_spectrum():
    """Test basic functionality of Nevanlinna AC, i.e. if we can
    recover the delta peak from imaginary axis data.
    """

    #
    # Test case 1: Dirac delta centered at w = 0
    #

    # Generate imaginary axis data for f(z) = 1 / z
    iw_vals = np.linspace(0.1, 10, 100)
    G_iw = 1 / (1j * iw_vals)

    outdir = 'DiracESNevan'
    n_real = 101
    w_min = -0.1
    w_max = 0.1
    eta = 0.01

    freqs, G_w = es_nevan_run(
        G_iw, iw_vals, n_real=n_real, w_min=w_min, w_max=w_max,
        eta=eta, outdir=outdir
    )
    assert G_w.shape == (n_real, ) + G_iw.shape[1:]

    A_w = -(1 / np.pi) * np.imag(G_w)
    f_max = freqs[np.argmax(A_w)]
    assert f_max == 0

    #
    # Test case 2: Dirac delta centerd at z0 = 1 + 1e-4j
    #

    # Generate imaginary axis data for f(z) = 1 / (z - z0)
    iw_vals = np.linspace(0.1, 10, 100)
    G_iw = 1 / (1j * iw_vals - (1 + 1e-4j))

    outdir = 'DiracESNevan'
    n_real = 201
    w_min = -0.5
    w_max = 1.5
    eta = 0.01

    freqs, G_w = es_nevan_run(
        G_iw, iw_vals, n_real=n_real, w_min=w_min, w_max=w_max,
        eta=eta, outdir=outdir
    )
    assert G_w.shape == (n_real, ) + G_iw.shape[1:]

    A_w = -(1 / np.pi) * np.imag(G_w)
    f_max = freqs[np.argmax(A_w)]
    assert f_max == 1.


@pytest.mark.skip(reason='ES not the right way for self-energy continuation')
def test_pes_nevan_exe_on_selfenergy(mbo):
    """Test functionality of `pes_nevan_run` on self-energy.
    """
    # Get mu, green's function, and self-energy
    mu = mbo.mu
    gtau = mbo.gtau
    sigma_tau = mbo.sigma
    Sk = mbo.S
    fock = mbo.fock

    # transform from tau -> iw
    nw = len(mbo.ir.wsample)
    iw_vals = mbo.ir.wsample[nw//2:]
    g_iw = mbo.ir.tau_to_w(gtau)[nw//2:]
    sigma_iw = mbo.ir.tau_to_w(sigma_tau)[nw//2:]

    # TODO: First transform to SAO
    f_sao = sao_orth(fock, Sk, type='f')
    g_iw_sao = sao_orth(g_iw, Sk, type='g')
    sig_iw_sao = sao_orth(sigma_iw, Sk, type='f')

    # perform ES analytic continuation of self-energy and green's function
    sig_outdir = 'SigmaESNevan'
    green_outdir = 'GreenESNevan'
    n_real = 101
    w_min = -5.0
    w_max = 5.0
    eta = 0.01
    freqs, Sigma_w = es_nevan_run(
        sig_iw_sao, iw_vals, n_real=n_real, w_min=w_min, w_max=w_max, eta=eta,
        outdir=sig_outdir, diag=False, parallel='sk', ofile='sig_w.txt'
    )
    freqs, Green_w = es_nevan_run(
        g_iw_sao, iw_vals, n_real=n_real, w_min=w_min, w_max=w_max, eta=eta,
        outdir=green_outdir, diag=False, parallel='sk', ofile='g_w.txt'
    )

    # check that Sigma_w is complex number
    assert Green_w.shape == (n_real, ) + g_iw.shape[1:]
    assert Sigma_w.shape == (n_real, ) + sigma_iw.shape[1:]

    # Do dyson from sigma
    G_new = Green_w * 0
    ns, nk = g_iw.shape[1:3]
    for s in range(ns):
        for k in range(nk):
            for wi, w_re in enumerate(freqs):
                tmp = (w_re + 1j * eta + mu) - f_sao[s, k] - Sigma_w[wi, s, k]
                G_new[wi, s, k] = np.linalg.inv(tmp)

    # Find maxima (should be at freqs = 1)
    assert np.linalg.norm(G_new - Green_w) < 1e-3
