import numpy as np
import MB_analysis.spectral as spec
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

def get_mo(fock, S, mo_basis = False):
    mo_energy, mo_coeff = spec.eig(fock[:, :, :], S[:, :, :])
    if mo_basis == True:
        return mo_energy, mo_coeff
    else:
        return mo_energy



def ao_to_mo(object, mo_coeff = None, type = 'f'):
    if mo_coeff is None:
        raise ValueError("MO basis is missing!")
    elif mo_coeff is not None:
        if type == 'f':
            tmp = np.einsum('...ij,...jl->...il', object, mo_coeff)
            object_mo = np.einsum('...ji,...jl->...il', np.conjugate(mo_coeff), tmp)
        elif type == 'g':
            tmp = np.einsum('...ij,...jl->...il', object, mo_coeff)
            object_mo = np.einsum('...ji,...jl->...il', np.conjugate(mo_coeff), tmp)
        else:
            raise ValueError("Wrong type of transformation!")
    return object_mo



