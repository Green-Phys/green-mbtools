import numpy as np
import scipy.linalg as LA
import h5py

import MB_analysis.spectral as spec
import MB_analysis.trans as trans
import MB_analysis.ir as ir


def compute_mo(fock, S=None):
  '''
  Compute molecular orbital energy by solving FC=SCE
  :return:
  '''
  mo_energy, mo_coeff = spec.eig(fock, S)

  return mo_energy, mo_coeff

def compute_no(dm, S):
  '''
  Compute natural orbitals by diagonalizing density matrix
  :return:
  '''
  ns, ink, nao = dm.shape[0], dm.shape[1], dm.shape[2]
  dm_orth = trans.orthogonal(dm, S, 'g')
  occ = np.zeros(np.shape(dm)[:-1])
  no_coeff = np.zeros(np.shape(dm), dtype=np.complex)
  for ss in range(ns):
    for ik in range(ink):
      occ[ss,ik], no_coeff[ss,ik] = np.linalg.eigh(dm_orth[ss,ik])

  occ, no_coeff = occ[:, :, ::-1], no_coeff[:, :, :, ::-1]
  return occ, no_coeff


class mb:
  '''Many-body analysis class'''
  def __init__(self, fock, sigma=None, mu=None, gtau=None, S=None,
               ir_list=None, weight=None, beta=None, lamb=None):
    if fock.ndim == 4 and fock.shape[0] == 2:
      self._ns = 2
    elif fock.ndim == 3:
      self._ns = 1
      if gtau is not None: gtau  = gtau.reshape((gtau.shape[0], 1) + gtau.shape[1:])
      if sigma is not None: sigma = sigma.reshape((sigma.shape[0], 1) + sigma.shape[1:])
      fock  = fock.reshape((1) + fock.shape)
      if S is not None: S   = S.reshape((1,) + S.shape)
    else:
      raise ValueError('Incorrect dimensions of self-energy or Fock. '
                       'Accetable shapes are (nts, ns, nk, nao, nao) or (nts, nk, nao, nao) for self-energy and '
                       '(ns, nk, nao, nao) or (nk, nao, nao) for Fock matrix.')

    if beta is None:
      print("Warning: Inverse temperature is set to the default value 1000 a.u.^{-1}.")
    else:
      self.beta = beta
    if lamb is None:
      print("Warning: Lambda is set to the default value 1e6.")
    else:
      self.lamb = lamb
    self._ir = ir.IR(self.beta, self.lamb)
    # Dims
    self._nts = self._ir.nts
    self._ink = fock.shape[1]
    self._nao = fock.shape[2]


    if sigma is not None: self._sigma = sigma.copy()
    self._fock = fock.copy()
    if S is not None: self._S = S.copy()
    if mu is not None: self.mu = mu

    if gtau is not None:
      self._gtau = gtau.copy()
      self._dm = self._gtau[-1]
    else:
      self.solve_dyson()

    if ir_list is not None and weight is not None:
      self._ir_list = ir_list
      self._weight = weight
    else:
      self._ir_list = np.arange(self._ink)
      self._weight = np.array([1 for i in range(self._ink)])

  # Private class variables
  _gtau  = None
  _sigma = None
  _dm    = None
  _fock  = None
  _S     = None
  _S_inv_12 = None

  _nts  = None
  _ns   = None
  _ink  = None
  _nao  = None

  _mo_sk_energy = None
  _mo_sk_coeff = None

  _iw_list = None
  _no = None
  _no_coeff = None

  _ir_list = None
  _weight = None

  _ir = None

  # Public class variables
  mu = 0.0
  beta = 1000
  lamb = 1e6
  mo_energy = None
  mo_coeff = None

  def solve_dyson(self):
    '''
    Compute Green's function through Dyson's equation and update self._gtau and self._dm.
    :return:
    '''
    sigma_w = None
    if self._sigma is not None:
      sigma_w = self._ir.tau_to_w(self._sigma)

    G_w = np.zeros((self._ir.nw, self._ns, self._ink, self._nao, self._nao), dtype=np.complex)
    for n in range(self._ir.nw):
      for s in range(self._ns):
        for k in range(self._ink):
          if sigma_w is None:
            tmp = (1j*self._ir.wsample[n] + self.mu) * self._S[s, k] - self._fock[s, k]
          else:
            tmp = (1j * self._ir.wsample[n] + self.mu) * self._S[s, k] - self._fock[s, k] - self._sigma[n, s, k]
          G_w[n, s, k] = np.linalg.inv(tmp)

    self._gtau = self._ir.w_to_tau(G_w)
    self._dm = self._gtau[-1]

  def get_mo(self):
    '''
    Compute molecular orbital energy by solving FC=SCE
    :return:
    '''
    self.mo_energy, self.mo_coeff = compute_mo(self._fock, self._S)
    return self.mo_energy, self.mo_coeff

  def get_no(self):
    '''
    Compute natural orbitals by diagonalizing density matrix
    :return:
    '''
    occ, no_coeff = compute_no(self._dm, self._S)
    return occ, no_coeff

  def mulliken_analysis(self, orbitals=None):
    if orbitals is None:
      orbitals = np.arange(self._nao)
    occupations = np.zeros((self._ns, orbitals.shape[0]))
    for ss in range(self._ns):
      for ik in range(self._ink):
        n_k = np.zeros(orbitals.shape[0])
        for i in orbitals:
          for j in range(self._nao):
            n_k[i] += self._dm[ss,ik,i,j] * self._S[ss,ik,j,i]
        k_ir = self._ir_list[ik]
        occupations[s] += self._weight[k_ir] * n_k
    num_k = len(self._weight)
    occupations /= num_k

    # Check imaginary part
    imag = np.max(np.abs(occupations.imag))
    print("The maximum of imaginary part is ", imag)

    return occupations.real

if __name__ == '__main__':
  f = h5py.File("data/H2_GW/sim.h5", 'r')
  S = f["S-k"][()].view(np.complex)
  S = S.reshape(S.shape[:-1])
  F = f["iter14/Fock-k"][()].view(np.complex)
  F = F.reshape(F.shape[:-1])
  Sigma = f["iter14/Selfenergy/data"][()].view(np.complex)
  Sigma = Sigma.reshape(Sigma.shape[:-1])
  mu = f["iter14/mu"][()]
  f.close()

  '''
  Results from mean-field calculations
  '''
  # Standard way to initialize
  # density and non-interacting Green's function are computed internally
  #manybody = mb(fock, S=S, beta=1000, lamb='1e4')

  '''
  Results from correlated methods
  '''
  # Standard way to initialize
  #manybody = mb(fock=fock, sigma=sigma, gtau=gtau, S=S, beta=1000, lamb='1e4')
  # If G(t) is not known, Dyson euqation can be solved on given beta and ir grid.
  manybody = mb(fock=F, sigma=Sigma, mu=mu, S=S, beta=1000, lamb='1e4')