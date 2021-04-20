import numpy as np
import scipy.linalg as LA
import h5py

import MB_analysis.spectral as spec
import MB_analysis.orth as orth
import MB_analysis.ir as ir
import MB_analysis.dyson as dyson


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
  dm_orth = orth.sao_orth(dm, S, 'g')
  occ = np.zeros(np.shape(dm)[:-1])
  no_coeff = np.zeros(np.shape(dm), dtype=np.complex)
  for ss in range(ns):
    for ik in range(ink):
      occ[ss,ik], no_coeff[ss,ik] = np.linalg.eigh(dm_orth[ss,ik])

  occ, no_coeff = occ[:, :, ::-1], no_coeff[:, :, :, ::-1]
  return occ, no_coeff


class mb(object):
  '''Many-body analysis class'''
  def __init__(self, fock, sigma=None, mu=None, gtau=None, S=None,
               ir_list=None, weight=None, beta=None, lamb=None):
    if fock.ndim == 4 and fock.shape[0] == 2:
      self._ns = 2
    elif fock.ndim == 3:
      self._ns = 1
      fock = fock.reshape((1,) + fock.shape)
      if gtau is not None: gtau  = gtau.reshape((gtau.shape[0], 1) + gtau.shape[1:])
      if sigma is not None: sigma = sigma.reshape((sigma.shape[0], 1) + sigma.shape[1:])
      if S is not None: S   = S.reshape((1,) + S.shape)
    else:
      raise ValueError('Incorrect dimensions of self-energy or Fock. '
                       'Accetable shapes are (nts, ns, nk, nao, nao) or (nts, nk, nao, nao) for self-energy and '
                       '(ns, nk, nao, nao) or (nk, nao, nao) for Fock matrix.')

    if ir_list is not None and weight is not None:
      self._ir_list = ir_list
      self._weight = weight[ir_list]
    else:
      self._ir_list = np.arange(self._ink)
      self._weight = np.array([1 for i in range(self._ink)])

    if beta is None:
      print("Warning: Inverse temperature is set to the default value 1000 a.u.^{-1}.")
    else:
      self.beta = beta
    if lamb is None:
      print("Warning: Lambda is set to the default '1e6'.")
    else:
      self.lamb = lamb
    self._ir = ir.IR(self.beta, self.lamb)
    # Dims
    self._nts = self._ir.nts
    self._ink = fock.shape[1]
    self._nao = fock.shape[2]

    self.fock = fock.copy()
    if sigma is not None: self.sigma = sigma.copy()
    if S is not None: self.S = S.copy()
    if mu is not None: self.mu = mu

    if gtau is not None:
      self.gtau = gtau.copy()
      self.dm = -1.0 * self.gtau[-1]
    else:
      self.solve_dyson()

  # Private class variables
  gtau  = None
  sigma = None
  dm    = None
  fock  = None
  S     = None
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
  lamb = '1e6'
  mo_energy = None
  mo_coeff = None

  def solve_dyson(self):
    '''
    Compute Green's function through Dyson's equation and update self.gtau and self.dm.
    :return:
    '''
    self.gtau, self.dm = dyson.solve_dyson(self.fock, self.S, self.sigma, self.mu, self._ir)

  def get_mo(self):
    '''
    Compute molecular orbital energy by solving FC=SCE
    :return:
    '''
    self.mo_energy, self.mo_coeff = compute_mo(self.fock, self.S)
    return self.mo_energy, self.mo_coeff

  def get_no(self):
    '''
    Compute natural orbitals by diagonalizing density matrix
    :return:
    '''
    occ, no_coeff = compute_no(self.dm, self.S)
    return occ, no_coeff

  def mulliken_analysis(self, orbitals=None):
    if orbitals is None:
      orbitals = np.arange(self._nao)
    occupations = np.zeros((self._ns, orbitals.shape[0]), dtype=np.complex)
    for ss in range(self._ns):
      for ik in range(self._ink):
        n_k = np.zeros(orbitals.shape[0], dtype=np.complex)
        for i in orbitals:
          for j in range(self._nao):
            n_k[i] += self.dm[ss,ik,i,j] * self.S[ss,ik,j,i]
        #k_ir = self._ir_list[ik]
        occupations[ss] += self._weight[ik] * n_k
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
  G = f["iter14/G_tau/data"][()].view(np.complex)
  G = G.reshape(G.shape[:-1])
  mu = f["iter14/mu"][()]
  f.close()

  f = h5py.File("data/H2_GW/input.h5", 'r')
  ir_list = f["/grid/ir_list"][()]
  weight = f["/grid/weight"][()]
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
  manybody = mb(fock=F, sigma=Sigma, mu=mu, gtau=G, S=S, ir_list=ir_list, weight=weight, beta=1000, lamb='1e4')
  G = manybody.gtau
  # If G(t) is not known, Dyson euqation can be solved on given beta and ir grid.
  manybody = mb(fock=F, sigma=Sigma, mu=mu, S=S, ir_list=ir_list, weight=weight, beta=1000, lamb='1e4')
  G2 = manybody.gtau

  diff = G - G2
  print("Maximum G difference = ", np.max(np.abs(diff)))

  '''
  Mulliken analysis
  '''
  occs = manybody.mulliken_analysis()
  print("Spin up:", occs[0])
  print("Spin donw:", occs[1])

  '''
  Natural orbitals
  '''
  occ, no_coeff = manybody.get_no()
  print(occ[0,0])
  print(occ[1,0])
