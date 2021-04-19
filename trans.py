from functools import reduce
import numpy as np
import scipy.linalg as LA

def orth(X, S, type=None):
    if type != 'g' and type != 'f':
        raise ValueError("Valid transformation types are 'g' for density, 'f' for Fock")
    ns  = S.shape[0]
    nk  = S.shape[1]
    nao = S.shape[2]
    original_shape = X.shape
    X = X.reshape(-1, ns, nk, nao, nao)
    print("Shape of X = ", X.shape)
    X_orth = np.zeros(X.shape, dtype=X.dtype)
    for d in range(X.shape[0]):
        for s in range(ns):
            for ik in range(nk):
                S12 = LA.sqrtm(S[s,ik])
                if type == 'g':
                    X_orth[d,s,ik] = reduce(np.dot, (S12, X[d,s,ik], S12))
                elif type == 'f':
                    S12_inv = np.linalg.inv(S12)
                    X_orth[d,s,ik] = reduce(np.dot, (S12_inv, X[d,s,ik], S12_inv))
    X_orth = X_orth.reshape(original_shape)
    return X_orth

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
    object_loc = np.zeros((nao,nao), dtype=np.complex)
    if ir_list is not None and weight is not None:
        sym = True
        nk = np.shape(weight)[0]
    else:
        sym = False
        nk = ink
        ir_list = np.arange(nk)
        weight = np.array([1 for i in range(nk)])
    # Transform to orthogonal basis
    if S_k is not None:
        # Orthogonalization first
        object_k_orth = orthogonal(object_k, S_k, type = type)

    # Sum over k points
    for ik_ind in range(ink):
        object_loc += weight[ik_ind] * object_k_orth[ik_ind]
    object_loc/=nk
    if sym is False:
        return object_loc
    else:
        return object_loc.real
