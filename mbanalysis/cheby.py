import numpy as np
import h5py


def iwmesh(iw_list, beta):
    """Defines imaginary frequency grid from imaginary time grid.
    """
    return 1j * (2 * iw_list + 1) * np.pi / beta


def get_tau_mesh(beta, ncheb):
    """Generate tau grid for a given beta in the chebishev basis.
    """
    nts = ncheb + 2
    tau_mesh = np.zeros(nts)
    tau_mesh[0], tau_mesh[-1] = 0, beta
    # Define tau mesh
    for k in range(1, nts - 1):
        xk = np.cos(np.pi * (k - 0.5) / ncheb)
        tau_mesh[nts - k - 1] = (xk + 1) * beta / 2
    return tau_mesh


def get_Cheby(ncheb, tau_mesh):
    nts = ncheb + 2
    beta = tau_mesh[-1]
    # Define Chebyshev polynomials
    _Ttc = np.zeros((nts, ncheb))
    for it in range(nts):
        x = 2.0 * tau_mesh[it] / beta - 1.0
        _Ttc[it, 0] = 1.0
        _Ttc[it, 1] = x
        for ic in range(2, ncheb):
            _Ttc[it, ic] = 2.0 * x * _Ttc[it, ic - 1] - _Ttc[it, ic - 2]
    return _Ttc


###############
# Input    - Fermionic object in tau domain (nts, nk, nao, nao)
#
# Output - Fermionic object in Matsubara domain (nk, nao, nao)
###############
# TODO make beta as a parameter
# TODO make TNC.h5 path as a parameter and should be initialized at mb class.
# TODO make it compatible with local object (i.e no k-dependency)
def sigma_w_uniform(Sigma_tau, nw):
    beta = 100
    nk, nao = np.shape(Sigma_tau)[1:3]
    nts, ncheb = np.shape(Sigma_tau)[0], np.shape(Sigma_tau)[0] - 2
    # Define tau mesh on nodes
    tau_mesh = get_tau_mesh(beta, ncheb)
    # Compute Chebyshev on nodes
    _Ttc = get_Cheby(ncheb, tau_mesh)

    normal = 1.0/ncheb
    _Tct = np.zeros((ncheb, nts))
    for ic in range(ncheb):
        if ic == 0:
            factor = 1.0
        else:
            factor = 2.0
        for it in range(1, nts-1):
            _Tct[ic, it] = _Ttc[it, ic] * factor * normal

    _Tnc = np.zeros((nw, ncheb), dtype=complex)
    Tnc = h5py.File('/Users/CanonYeh/Projects/chebyshev_input/TNC.h5')
    for ic in range(ncheb):
        re = Tnc["TNC_" + str(ic) + "_r"][()]
        im = Tnc["TNC_" + str(ic) + "_i"][()]
        for iw in range(nw):
            _Tnc[iw, ic] = complex(re[iw], im[iw]) * beta / 2.0
    Tnc.close()

    Sigma_tau = Sigma_tau.reshape(nts, nk*nao*nao)
    # Selfenergy in Chebyshev representation
    Sigma_c = np.einsum('ij,jk -> ik', _Tct, Sigma_tau)
    # Selfenergy in Matsubara axes
    Sigma_w = np.einsum('ij,jk->ik', _Tnc, Sigma_c)
    Sigma_w = Sigma_w.reshape(nw, nk, nao, nao)
    return Sigma_w


###############
# Input    - Fermionic object in tau domain (nts, nk, nao, nao)
#
# Output - Fermionic object in Matsubara domain at n = 0 (nk, nao, nao)
# TODO make beta as a parameter
###############
def sigma_zero(Sigma):
    # Setup
    beta = 100
    nk, nao = np.shape(Sigma)[1:3]
    nts, ncheb = np.shape(Sigma)[0], np.shape(Sigma)[0] - 2
    tau_mesh = get_tau_mesh(beta, ncheb)
    # Define Chebyshev polynomials
    _Ttc = get_Cheby(ncheb, tau_mesh)

    normal = 1.0/ncheb
    _Tct = np.zeros((ncheb, nts))
    for ic in range(ncheb):
        if ic == 0:
            factor = 1.0
        else:
            factor = 2.0
        for it in range(1, nts-1):
            _Tct[ic, it] = _Ttc[it, ic] * factor * normal

    _T_0l = np.zeros(ncheb, dtype=complex)
    Tnl = h5py.File('/Users/CanonYeh/Projects/chebyshev_input/TNC.h5')
    for ic in range(ncheb):
        re = Tnl["TNC_"+str(ic)+"_r"][()]
        im = Tnl["TNC_"+str(ic)+"_i"][()]
        _T_0l[ic] = complex(re[0], im[0]) * beta / 2.0
    Tnl.close()

    Sigma = Sigma.reshape(nts, nk*nao*nao)
    # Selfenergy in Chebyshev representation
    Sigma_c = np.dot(_Tct, Sigma)
    # Sigma_c = np.einsum('ij,jklm -> iklm', _Tct, Sigma)
    # Selfenergy in zero frequency
    Sigma_w0 = np.dot(_T_0l, Sigma_c)
    # Sigma_w0 = np.einsum('i,ijkl->jkl', _T_0l, Sigma_c)

    Sigma_w0 = Sigma_w0.reshape(nk, nao, nao)
    return Sigma_w0
