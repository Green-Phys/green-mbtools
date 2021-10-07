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
    mo_coeff = np.zeros((fock.shape), dtype=complex)
    for iks in range(fock.shape[0]):
      eps_k[iks], mo_coeff[iks] = np.linalg.eigh(fock[iks])

    if U == True:
      eps_k = eps_k.reshape(ns, nk, nao)
      mo_coeff = mo_coeff.reshape(ns, nk, nao, nao)

    return eps_k, mo_coeff

def compute_mo(F, S, eigh_solver=LA.eigh, thr=1e-7):
  '''
  Solve the generalized eigen problem: FC = SCE

  :param F: Fock matrix, dim = (ns, nk, nao, nao)
  :param S: Overlap matrix, dim = (ns, nk, nao, nao)
  :param eigh_solver: eigenvalue problem solver
  :param thr: lowest eigenvalues of S. Only used in canonical orthogonalization.

  :return:
  eig_sk: molecular energies
  mo_coeff_k: molecular orbital coefficient in AO basis
  '''
  ns, nk, nao = F.shape[0:3]
  eiv_sk = np.zeros((ns, nk, nao))
  mo_coeff_sk = np.zeros((ns, nk, nao, nao), dtype=F.dtype)
  #eiv_sk = []
  #mo_coeff_sk = []
  if S is None:
    S = np.array([[np.eye(nao)]*nk]*ns)
  for ss in range(ns):
    for k in range(nk):
      eiv, mo = eigh_solver(F[ss,k], S[ss, k], thr)
      # Re-order
      idx = np.argmax(abs(mo.real), axis=0)
      mo[:, mo[idx, np.arange(len(eiv))].real < 0] *= -1
      nbands = eiv.shape[0]
      eiv_sk[ss, k, :nbands] = eiv
      mo_coeff_sk[ss, k, :, :nbands] = mo
      #eiv_sk.append(eiv)
      #mo_coeff_sk.append(mo)

  #eig_sk = np.asarray(eiv_sk).reshape(ns, nk, nao)
  #mo_coeff_sk = np.asarray(mo_coeff_sk).reshape(ns, nk, nao, nao)

  return eiv_sk, mo_coeff_sk