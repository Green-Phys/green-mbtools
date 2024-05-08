import numpy as np
import scipy.linalg as LA
from . import orth

#################
# Input - Fock matrix. Dim = (ns, nk, nao, nao)
# Input - Unrestricted or not
# Input - return quasi-particle basis or not
#
# Output - Quasi-particle states
#################


def compute_mo(F, S, eigh_solver=LA.eigh, thr=1e-7):
    '''
    Solve the generalized eigen problem: FC = SCE

    :param F: Fock matrix, dim = (ns, nk, nao, nao)
    :param S: Overlap matrix, dim = (ns, nk, nao, nao)
    :param eigh_solver: eigenvalue problem solver
    :param thr: lowest eigenvalues of S.
    Only used in canonical orthogonalization.

    :return:
    eig_sk: molecular energies
    mo_coeff_k: molecular orbital coefficient in AO basis
    '''
    ns, nk, nao = F.shape[0:3]
    eiv_sk = np.zeros((ns, nk, nao))
    mo_coeff_sk = np.zeros((ns, nk, nao, nao), dtype=F.dtype)
    # eiv_sk = []
    # mo_coeff_sk = []
    if S is None:
        S = np.array([[np.eye(nao)]*nk]*ns)
    for ss in range(ns):
        for k in range(nk):
            eiv, mo = eigh_solver(F[ss, k], S[ss, k], thr)
            # Re-order
            idx = np.argmax(abs(mo.real), axis=0)
            mo[:, mo[idx, np.arange(len(eiv))].real < 0] *= -1
            nbands = eiv.shape[0]
            eiv_sk[ss, k, :nbands] = eiv
            mo_coeff_sk[ss, k, :, :nbands] = mo
            # eiv_sk.append(eiv)
            # mo_coeff_sk.append(mo)

    # eig_sk = np.asarray(eiv_sk).reshape(ns, nk, nao)
    # mo_coeff_sk = np.asarray(mo_coeff_sk).reshape(ns, nk, nao, nao)

    return eiv_sk, mo_coeff_sk


def compute_no(dm, S=None):
    """Compute natural orbitals by diagonalizing density matrix
    :return:
    """
    ns, ink = dm.shape[0], dm.shape[1]
    dm_orth = orth.sao_orth(dm, S, 'g') if S is not None else dm.copy()
    occ = np.zeros(np.shape(dm)[:-1])
    no_coeff = np.zeros(np.shape(dm), dtype=complex)
    for ss in range(ns):
        for ik in range(ink):
            occ[ss, ik], no_coeff[ss, ik] = np.linalg.eigh(dm_orth[ss, ik])

    occ, no_coeff = occ[:, :, ::-1], no_coeff[:, :, :, ::-1]
    return occ, no_coeff
