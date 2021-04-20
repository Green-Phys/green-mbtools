import numpy as np
import MB_analysis.spectral as spec

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


###############
# Input - k-resolved object in orthogonal basis. object = [nk, nao, nao]
# Input - List of reduced k-points
# Input - Corresponding weight
#
# Ouput - local object
###############
def to_local(object_k, ir_list=None, weight=None):
    ink = np.shape(object_k)[0]
    nao = np.shape(object_k)[1]
    object_loc = np.zeros((nao,nao))
    if ir_list is not None and weight is not None:
        nk = np.shape(weight)[0]
    else:
        nk = ink
        ir_list = np.arange(nk)
        weight = np.array([1 for i in range(nk)])
    # Sum over k points
    for ik_ind in range(ink):
        ik = ir_list[ik_ind]
        object_loc += weight[ik] * object_k[ik_ind].real
    object_loc/=nk

    return object_loc



