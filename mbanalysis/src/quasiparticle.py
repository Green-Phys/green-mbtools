import os
import h5py
import numpy as np
import mbanalysis.src.analyt_cont as ac

"""
Perform quasiparticle approximation
"""

"""
Construct quasiparticle renormalization factors in a local Hamiltonian
F = (ns, nao, nao)
Sigma_iw = (nw, ns, nao, nao)
"""


def Z_factor(F, Sigma_iw, iwsample, nevan_sigma_exe, outdir='sigma_nevan'):
    ns, nao = F.shape[:2]
    # Nevanlinna continuation of self-energy.
    Sigma_iw_diag = np.einsum('wsii->wsi', Sigma_iw)
    print("Sigma_iw_diag shape = {}".format(Sigma_iw_diag.shape))
    nw = iwsample.shape[0]
    iwsample_pos = iwsample[nw//2:]
    Sigma_iw_diag = Sigma_iw_diag[nw//2:]
    input_parser = 'Sigma_iw.txt ' + str(nw//2) + ' Sigma_w.txt coeff'
    ac.nevan_run_selfenergy(
        Sigma_iw_diag, iwsample_pos, input_parser, nevan_sigma_exe, outdir
    )

    if not os.path.exists(outdir):
        raise ValueError("Directory {} doesn't exist!".format(outdir))
    f = h5py.File(outdir+"/Sigma_w.h5", 'r')

    # Only diagonal terms are analytically continued.
    # Sigma_w = (freqs, ns, nao)
    Sigma_w = f["Sigma_w"][()].view(complex)
    freqs = f["freqs"][()]
    f.close()

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