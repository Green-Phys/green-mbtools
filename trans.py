import numpy as np
import scipy.linalg as LA
import h5py

def iwmesh(iw_list, beta):
    return 1j*(2*iw_list + 1)*np.pi/beta

# Orthogonalization
# type = 'g': Green's function and density matrix
# type = 'f': Fock matrix and self-energy
def orthogonal(object, S, type):
    if type != 'g' and type != 'f':
        raise ValueError("Need to specify transformation type: 'g' or 'f' ")
    if len(S.shape) == 2:
        ns = 1
        ink = 1
    elif len(S.shape) == 3:
        ns = 1
        ink = np.shape(S)[0]
    elif len(S.shape) == 4:
        ns  = np.shape(S)[0]
        ink = np.shape(S)[1]
    else:
        raise ValueError("Dims of S are wrong!")
    nao = np.shape(S)[-1]
    object = object.reshape(ns*ink,nao,nao)
    S = S.reshape(ns*ink,nao,nao)
    object_orth = np.zeros(np.shape(object), dtype=np.complex)
    for iks in range(ns*ink):
        S12 = LA.sqrtm(S[iks, :, :])
        if type == 'g':
            object_orth[iks] = np.dot(S12, np.dot(object[iks, :, :], S12))
        elif type == 'f':
            S12_inv = np.linalg.inv(S12)
            object_orth[iks] = np.dot(S12_inv, np.dot(object[iks, :, :], S12_inv))

    if len(S.shape) == 2:
        object = object.reshape(nao, nao)
        S = S.reshape(nao, nao)
        object_orth = object_orth.reshape(nao, nao)
    elif len(S.shape) == 3:
        object = object.reshape(ink, nao, nao)
        S = S.reshape(ink, nao, nao)
        object_orth = object_orth.reshape(ink, nao, nao)
    elif len(S.shape) == 4:
        object = object.reshape(ns, ink, nao, nao)
        S = S.reshape(ns, ink, nao, nao)
        object_orth = object_orth.reshape(ns, ink, nao, nao)

    return object_orth

###############
# Input  - Static object in non-orhtogonal basis (e.g. AO basis). Dim = (nk, nao, nao) or (ns, nk, nao, nao)
# Input  - Overlap matrix S (nk, nao, nao)
#
# Output - Object in orthogonal basis
###############
def orth(F, S, U = False):
    if np.shape(F)[-3:] != np.shape(S)[-3:]:
        raise ValueError("Dims of F doesn't match S!")

    nk, nao = F.shape[-3:-1]
    if U == True:
        ns = F.shape[0]
        F = F.reshape(ns*nk, nao, nao)
        S = S.reshape(ns*nk, nao, nao)

    F_orth = np.zeros(F.shape, dtype=np.complex)
    for iks in range(F.shape[0]):
        S12 = LA.sqrtm(S[iks,:,:])
        F_orth[iks, :, :] = np.dot(S12, np.dot(F[iks, :, :], S12))

    if U == True:
        F_orth = F_orth.reshape(ns, nk, nao, nao)
    else:
        F_orth = F_orth.reshape(nk, nao, nao)

    return F_orth


def get_tau_mesh(beta, ncheb):
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
# Input - k-resolved object in non-orthogonal/orthogonal basis. object = [nk, nao, nao]
# Input - List of reduced k-points
# Input - Corresponding weight
#
# Ouput - local object
###############
def to_local(object_k, S_k = None, ir_list=None, weight=None, type=None):
    ink = np.shape(object_k)[0]
    nao = np.shape(object_k)[1]
    object_loc = np.zeros((nao,nao))
    if ir_list is not None and weight is not None:
        nk = np.shape(weight)[0]
    else:
        nk = ink
        ir_list = np.arange(nk)
        weight = np.array([1 for i in range(nk)])
    if S_k is not None:
        # Orthogonalization first
        object_k_orth = orthogonal(object_k, S_k, type = type)

    # Sum over k points
    for ik_ind in range(ink):
        ik = ir_list[ik_ind]
        object_loc += weight[ik] * object_k_orth[ik_ind].real
    object_loc/=nk

    return object_loc


def w_to_tau_ir(Sigma_w, ir_path, beta):
    if len(Sigma_w.shape) == 4:
        ns = 1
        nk, nao = Sigma_w.shape[1:3]
        nw = Sigma_w.shape[0]
    elif len(Sigma_w.shape) == 5:
        ns = Sigma_w.shape[1]
        nk, nao = Sigma_w.shape[2:4]
        nw = Sigma_w.shape[0]
    Sigma_w = Sigma_w.reshape(nw, ns*nk, nao, nao)
    ir_file = h5py.File(ir_path)
    iw_list = ir_file["/fermi/wsample"][()]
    txl_tmp = ir_file["/fermi/uxl"][()]
    txl_one = ir_file["/fermi/ux1l"][()]
    txl_minone = ir_file["/fermi/ux1l_minus"][()]
    txl = np.zeros((txl_tmp.shape[0]+2,txl_tmp.shape[1]))
    txl[1:-1] = txl_tmp
    txl[0] = txl_minone
    txl[-1] = txl_one
    tnl = ir_file["/fermi/uwl"][()].view(np.complex)
    ir_file.close()
    nw = iw_list.shape[0]
    txl *= np.sqrt(2.0 / beta)
    tnl *= np.sqrt(beta)
    tln = np.linalg.pinv(tnl)
    Sigma_c = np.einsum('ij,j...->i...', tln, Sigma_w)
    Sigma_t = np.einsum('ij,j...->i...', txl, Sigma_c)

    if len(Sigma_w.shape) == 5:
        Sigma_t = Sigma_t.reshape(txl_tmp.shape[0]+2, ns, nk, nao, nao)
        Sigma_w = Sigma_w.reshape(nw, ns, nk, nao, nao)

    return Sigma_t


def tau_to_w_ir(Sigma_tau, ir_path, beta):
    if len(Sigma_tau.shape) == 4:
        ns = 1
        nk, nao = Sigma_tau.shape[1:3]
        nts, ni = Sigma_tau.shape[0], Sigma_tau.shape[0] - 2
    elif len(Sigma_tau.shape) == 5:
        ns = Sigma_tau.shape[1]
        nk, nao = Sigma_tau.shape[2:4]
        nts, ni = Sigma_tau.shape[0], Sigma_tau.shape[0] - 2
    Sigma_tau = Sigma_tau.reshape(nts, ns*nk, nao, nao)
    ir_file = h5py.File(ir_path)
    iw_list = ir_file["/fermi/wsample"][()]
    txl = ir_file["/fermi/uxl"][()]
    tnl = ir_file["/fermi/uwl"][()].view(np.complex)
    ir_file.close()
    nw = iw_list.shape[0]
    txl *= np.sqrt(2.0 / beta)
    tlx = np.linalg.pinv(txl)
    tnl *= np.sqrt(beta)
    #Sigma_c = np.zeros((ni, ns*nk, nao, nao))
    #Sigma_w = np.zeros((nw, ns*nk, nao, nao), dtype=np.complex)
    Sigma_c = np.einsum('ij,j...->i...',tlx, Sigma_tau[1:nts-1])
    Sigma_w = np.einsum('ij,j...->i...',tnl, Sigma_c)

    if len(Sigma_tau.shape) == 5:
        Sigma_tau = Sigma_tau.reshape(nts, ns, nk, nao, nao)
        Sigma_w = Sigma_w.reshape(nw, ns, nk, nao, nao)

    return Sigma_w



###############
# Input  - Fermionic object in tau domain (nts, nk, nao, nao)
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
        for it in range(1,nts-1):
            _Tct[ic, it] = _Ttc[it, ic] * factor * normal

    _Tnc = np.zeros((nw,ncheb), dtype=complex)
    Tnc = h5py.File('/Users/CanonYeh/Projects/chebyshev_input/TNC.h5')
    for ic in range(ncheb):
        re = Tnc["TNC_" + str(ic) + "_r"][()]
        im = Tnc["TNC_" + str(ic) + "_i"][()]
        for iw in range(nw):
            _Tnc[iw, ic] = complex(re[iw],im[iw]) * beta/2.0
    Tnc.close()

    Sigma_tau = Sigma_tau.reshape(nts, nk*nao*nao)
    # Selfenergy in Chebyshev representation
    Sigma_c = np.einsum('ij,jk -> ik', _Tct, Sigma_tau)
    # Selfenergy in Matsubara axes
    Sigma_w = np.einsum('ij,jk->ik', _Tnc, Sigma_c)
    Sigma_w = Sigma_w.reshape(nw, nk, nao, nao)
    return Sigma_w

###############
# Input  - Fermionic object in tau domain (nts, nk, nao, nao)
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
        for it in range(1,nts-1):
            _Tct[ic, it] = _Ttc[it, ic] * factor * normal

    _T_0l = np.zeros(ncheb, dtype=complex)
    Tnl = h5py.File('/Users/CanonYeh/Projects/chebyshev_input/TNC.h5')
    for ic in range(ncheb):
        re = Tnl["TNC_"+str(ic)+"_r"][()]
        im = Tnl["TNC_"+str(ic)+"_i"][()]
        _T_0l[ic] = complex(re[0],im[0]) * beta/2.0
    Tnl.close()

    Sigma = Sigma.reshape(nts, nk*nao*nao)
    # Selfenergy in Chebyshev representation
    Sigma_c = np.dot(_Tct, Sigma)
    #Sigma_c = np.einsum('ij,jklm -> iklm', _Tct, Sigma)
    # Selfenergy in zero frequency
    Sigma_w0 = np.dot(_T_0l, Sigma_c)
    #Sigma_w0 = np.einsum('i,ijkl->jkl', _T_0l, Sigma_c)

    Sigma_w0 = Sigma_w0.reshape(nk, nao, nao)
    return Sigma_w0



