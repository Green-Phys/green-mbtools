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


def Z_factor(
    F, Sigma_iw, iwsample, outdir='sigma_nevan',
    ifile='Sigma_iw.txt', ofile='Sigma_w.txt', coefile='coeff',
    n_real=10001, w_min=-10, w_max=10, eta=0.01
):
    if outdir:
        warn("'outdir' parameter is deprecated for Nevanlinna", DeprecationWarning, 2)
    ns, nao = F.shape[:2]
    # Nevanlinna continuation of self-energy.
    Sigma_iw_diag = np.einsum('wsii->wsi', Sigma_iw)
    print("Sigma_iw_diag shape = {}".format(Sigma_iw_diag.shape))
    nw = iwsample.shape[0]
    iwsample_pos = iwsample[nw//2:]
    Sigma_iw_diag = Sigma_iw_diag[nw//2:]
    freqs, Sigma_w = ac.nevan_run(
        Sigma_iw_diag, iwsample_pos, ifile=ifile, ofile=ofile,
        coeff_file=coefile, n_real=n_real, w_min=w_min, w_max=w_max, eta=eta,
        green=False
    )

    # F_diag = (ns, nno)
    F_diag = np.array([np.diag(f) for f in F])
    dw = freqs[1] - freqs[0]
    Zs = np.zeros((ns, nao))
    for s in range(ns):
        for i in range(nao):
            e_approx, idx = find_nearest(freqs, F_diag[s, i])
            # Z = [1 - d(Sigma)/d(w)|w=e]^{-1}
            # Z^{-1} >= 1
            Z = 1 - np.gradient(Sigma_w[:, s, i].real, dw)
            Z = 1 / Z
            Zs[s, i] = Z[idx]

    return Zs


def find_nearest(array, value):
    import math
    idx = np.searchsorted(array, value, side="left")
    cond1 = idx > 0
    cond2 = idx == len(array) or \
        math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])

    if cond1 and cond2:
        return array[idx-1], idx-1
    else:
        return array[idx], idx
