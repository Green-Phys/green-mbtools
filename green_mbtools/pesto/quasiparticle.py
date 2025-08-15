import numpy as np
from warnings import warn
from . import analyt_cont as ac

"""
Perform quasiparticle approximation
"""

"""
Construct quasiparticle renormalization factors in a local Hamiltonian
F = (ns, nao, nao)
Sigma_iw = (nw, ns, nao, nao)
"""


def Z_factor(F, Sigma_iw, iwsample, n_real=1001, w_min=-5.0, w_max=5., eta=0.01):
    """Calculate the quasiparticle mass factor

    Parameters
    ----------
    F : numpy.ndarray
        Fock matrix of dimension (nspin, nk, nao, nao)
    Sigma_iw : numpy.ndarray
        Self-energy on Matsubara axis
    iwsample : numpy.ndarray
        Array of IR-grid Matsubara frequencies
    n_real : int, optional
        Number of real frequency grid points, by default 1001
    w_min : float, optional
        Minimum value on real frequency grid, by default -5
    w_max : float, optional
        Maximum value on real frequency grid, by default 5
    eta : float, optional
        Broadening parameter, by default 0.01

    Returns
    -------
    numpy.ndarray
        Quasiparticle weight factors of shape (nspin, nao)
    """
    ns, nao = F.shape[:2]
    # Nevanlinna continuation of self-energy.
    Sigma_iw_diag = np.einsum('wsii->wsi', Sigma_iw)
    print("Sigma_iw_diag shape = {}".format(Sigma_iw_diag.shape))
    nw = iwsample.shape[0]
    iwsample_pos = iwsample[nw//2:]
    Sigma_iw_diag = Sigma_iw_diag[nw//2:]
    freqs, Sigma_w = ac.nevan_run(Sigma_iw_diag, iwsample_pos, n_real, w_min, w_max, eta, spectral=False)

    # F_diag = (ns, nno)
    F_diag = np.array([np.diag(f) for f in F])
    dw = freqs[1] - freqs[0]
    Zs = np.zeros((ns, nao))
    for s in range(ns):
        for i in range(nao):
            e_approx, idx = _find_nearest(freqs, F_diag[s, i])
            # Z = [1 - d(Sigma)/d(w)|w=e]^{-1}
            # Z^{-1} >= 1
            Z = 1 - np.gradient(Sigma_w[:, s, i].real, dw)
            Z = 1 / Z
            Zs[s, i] = Z[idx]

    return Zs


def _find_nearest(array, value):
    import math
    idx = np.searchsorted(array, value, side="left")
    cond1 = idx > 0
    cond2 = idx == len(array) or \
        math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])

    if cond1 and cond2:
        return array[idx-1], idx-1
    else:
        return array[idx], idx
