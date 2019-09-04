import numpy as np
import scipy.linalg as LA

#################
# Input - Fock matrix. Dim = (ns, nk, nao, nao)
# Input - Unrestricted or not
# Input - return quasi-particle basis or not
#
# Output - Quasi-particle states
#################
def disper(fock, U = False, mo_basis = False):
  nk, nao = fock.shape[-3:-1]
  if U == True:
    ns = fock.shape[0]
    fock = fock.reshape(ns*nk, nao, nao)

  # Compute quasi-particle states
  if mo_basis == False:
    if U == True:
      eps_k = np.array([np.linalg.eigvalsh(fock[iks]) for iks in range(nk*ns)])
      eps_k = eps_k.reshape(ns, nk, nao)
    else:
      eps_k = np.array([np.linalg.eigvalsh(fock[ik]) for ik in range(nk)])

    return eps_k
  else:
    eps_k = np.zeros((fock.shape[:2]))
    mo_coeff = np.zeros((fock.shape), dtype=np.complex)
    for iks in range(fock.shape[0]):
      eps_k[iks], mo_coeff[iks] = np.linalg.eigh(fock[iks])

    if U == True:
      eps_k = eps_k.reshape(ns, nk, nao)
      mo_coeff = mo_coeff.reshape(ns, nk, nao, nao)

    return eps_k, mo_coeff

#################
# Solve the generalized eigen problem: FC = SCE
# Input - Fock matrix in non-orthogonal basis. Dim = (nk, nao, nao)
# Input - Overlap matrix. Dim = (nk, nao, nao)
#
# Output - One particles energies: E
# Output - Molecular orbital coefficients: C
#################

def eig(fock, S):
  nk, nao = fock.shape[0:2]
  eig_k = []
  mo_coeff_k = []

  for k in range(nk):
    eig, mo = LA.eigh(fock[k],S[k])
    # Re-order
    idx = np.argmax(abs(mo.real), axis=0)
    mo[:, mo[idx, np.arange(len(eig))].real < 0] *= -1
    eig_k.append(eig)
    mo_coeff_k.append(mo)
  eig_k = np.asarray(eig_k)
  mo_coeff_k = np.asarray(mo_coeff_k)

  return eig_k, mo_coeff_k