import numpy as np
import scipy.linalg as LA

import MB_analysis.src.spectral as spec
import MB_analysis.src.orth as orth
from MB_analysis.src.ir import IR_factory
import MB_analysis.src.dyson as dyson
import MB_analysis.src.winter as winter
import MB_analysis.src.maxent as maxent


def compute_mo(fock, S=None):
  '''
  Compute molecular orbital energy by solving FC=SCE
  :return:
  '''
  mo_energy, mo_coeff = spec.eig(fock, S)

  return mo_energy, mo_coeff

def compute_no(dm, S=None):
  '''
  Compute natural orbitals by diagonalizing density matrix
  :return:
  '''
  ns, ink, nao = dm.shape[0], dm.shape[1], dm.shape[2]
  dm_orth = orth.sao_orth(dm, S, 'g') if S is not None else dm.copy()
  occ = np.zeros(np.shape(dm)[:-1])
  no_coeff = np.zeros(np.shape(dm), dtype=np.complex)
  for ss in range(ns):
    for ik in range(ink):
      occ[ss,ik], no_coeff[ss,ik] = np.linalg.eigh(dm_orth[ss,ik])

  occ, no_coeff = occ[:, :, ::-1], no_coeff[:, :, :, ::-1]
  return occ, no_coeff


class MB_post(object):
  '''Many-body analysis class'''
  def __init__(self, fock, sigma=None, mu=None, gtau=None, S=None, kmesh=None, beta=None, lamb=None):
    ''' Initialization '''
    # Public instance variables
    self.gtau = None
    self.sigma = None
    self.dm = None
    self.fock = None
    self.S = None
    self.mu = 0.0
    self.kmesh = None
    self.beta = 1000
    self.lamb = '1e6'

    # Private instance variables
    self._S_inv_12 = None
    self._nts = None
    self._ns = None
    self._ink = None
    self._nao = None
    self._ir_list = None
    self._weight = None
    self._ir = None

    '''Setup'''
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

    if mu is None:
      print("Warning: Default chemical potential, mu = 0.0, is used.")
    else:
      self.mu = mu
    if beta is None:
      print("Warning: Inverse temperature is set to the default value 1000 a.u.^{-1}.")
    else:
      self.beta = beta
    if lamb is None:
      print("Warning: Lambda is set to the default '1e6'.")
    else:
      self.lamb = lamb
    self._ir = IR_factory(self.beta, self.lamb)
    self._nts = self._ir.nts
    self._ink = fock.shape[1]
    self._nao = fock.shape[2]
    self._ir_list = np.arange(self._ink)
    self._weight = np.array([1 for i in range(self._ink)])

    self.fock = fock.copy()
    if sigma is not None: self.sigma = sigma.copy()
    if S is not None: self.S = S.copy()
    if kmesh is not None: self.kmesh = kmesh.copy()

    if gtau is not None:
      self.gtau = gtau.copy()
      self.dm = -1.0 * self.gtau[-1]
    #else:
    #  self.solve_dyson()

    self.input_summary()

  def input_summary(self):
    print("######### MBPT analysis class #########")
    print("nts    =", self._nts)
    print("ns     =", self._ns)
    print("nk     =", self._ink)
    print("nao    =", self._nao)
    print("mu     =", self.mu)
    print("beta   =", self.beta)
    print("lambda =", self.lamb)
    print("#######################################")

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
    mo_energy, mo_coeff = compute_mo(self.fock, self.S)
    return mo_energy, mo_coeff

  def get_no(self):
    '''
    Compute natural orbitals by diagonalizing density matrix
    :return:
    '''
    if self.dm is None:
      self.solve_dyson()
    occ, no_coeff = compute_no(self.dm, self.S)
    return occ, no_coeff

  def mulliken_analysis(self, orbitals=None):
    if self.dm is None:
      self.solve_dyson()
    if orbitals is None:
      orbitals = np.arange(self._nao)
    occupations = np.zeros((self._ns, orbitals.shape[0]), dtype=np.complex)
    for ss in range(self._ns):
      for ik in range(self._ink):
        n_k = np.zeros(orbitals.shape[0], dtype=np.complex)
        for i in orbitals:
          if self.S is None:
            n_k[i] += self.dm[ss, ik, i, i]
          else:
            for j in range(self._nao):
              n_k[i] += self.dm[ss,ik,i,j] * self.S[ss,ik,j,i]
        occupations[ss] += self._weight[ik] * n_k
    num_k = len(self._weight)
    occupations /= num_k

    # Check imaginary part
    imag = np.max(np.abs(occupations.imag))
    print("The maximum of imaginary part is ", imag)

    return occupations.real

  def wannier_interpolation(self, kpts_inter, hermi=False, debug=False):
    '''
    Wannier interpolation
    :param kpts_int: Target k grid
    :return:
    '''
    if self.kmesh is None:
      raise ValueError("kmesh of input data is unknown. Please provide it.")
    Gtk_int, Sigma_tk_int, tau_mesh, Fk_int, Sk_int = winter.interpolate_G(self.fock, self.sigma, self.mu, self.S,
                                                                 self.kmesh, kpts_inter, self._ir, hermi=hermi, debug=debug)
    return Gtk_int, Sigma_tk_int, tau_mesh, Fk_int, Sk_int

  def analyt_cont(self, error=5e-3, maxent_exe='maxent', params='green.param', outdir='Maxent', gtau_orth=None):
    '''
    Analytical continuation using Maxent
    :param error:
    :param maxent_exe:
    :param params:
    :param outdir:
    :param gtau:
    :return:
    '''
    if gtau_orth is None:
      gtau_orth = orth.sao_orth(self.gtau, self.S, type='g') if self.S is not None else self.gtau
      gtau_inp = np.einsum("...ii->...i", gtau_orth)
    else:
      gtau_inp = gtau_orth
    tau_mesh = self._ir.tau_mesh

    maxent.run(gtau_inp, tau_mesh, error, params, maxent_exe, outdir)



def to_full_bz(X, conj_list, ir_list, bz_index, k_ind):
  index_list = np.zeros(bz_index.shape, dtype=int)
  for i, irn in enumerate(ir_list):
    index_list[irn] = i
  old_shape = X.shape
  new_shape = np.copy(old_shape)
  new_shape[k_ind] = conj_list.shape[0]
  Y = np.zeros(new_shape, dtype=X.dtype)
  for ik, kk in enumerate(bz_index):
    k = index_list[kk]
    if k_ind == 0:
      Y[ik, ::] = X[k, ::].conj() if conj_list[ik] else X[k, ::]
    elif k_ind == 1:
      Y[:, ik, ::] = X[:, k, ::].conj() if conj_list[ik] else X[:, k, ::]
    elif k_ind == 2:
      Y[:, :, ik, ::] = X[:, :, k, ::].conj() if conj_list[ik] else X[:, :, k, ::]
  return Y

if __name__ == '__main__':
  import h5py
  import MB_analysis

  MB_path = MB_analysis.__path__[0]
  f = h5py.File(MB_path + '/data/H2_GW/sim.h5', 'r')
  Sr = f["S-k"][()].view(np.complex)
  Sr = Sr.reshape(Sr.shape[:-1])
  Fr = f["iter14/Fock-k"][()].view(np.complex)
  Fr = Fr.reshape(Fr.shape[:-1])
  Sigmar = f["iter14/Selfenergy/data"][()].view(np.complex)
  Sigmar = Sigmar.reshape(Sigmar.shape[:-1])
  Gr = f["iter14/G_tau/data"][()].view(np.complex)
  Gr = Gr.reshape(Gr.shape[:-1])
  mu = f["iter14/mu"][()]
  f.close()

  f = h5py.File(MB_path + '/data/H2_GW/input.h5', 'r')
  ir_list = f["/grid/ir_list"][()]
  weight = f["/grid/weight"][()]
  index = f["/grid/index"][()]
  conj_list = f["grid/conj_list"][()]
  f.close()

  ''' All k-dependent matrices should lie on a full Monkhorst-Pack grid. '''
  F = to_full_bz(Fr, conj_list, ir_list, index, 1)
  S = to_full_bz(Sr, conj_list, ir_list, index, 1)
  Sigma = to_full_bz(Sigmar, conj_list, ir_list, index, 2)
  G = to_full_bz(Gr, conj_list, ir_list, index, 2)

  ''' Results from correlated methods '''
  # Standard way to initialize
  manybody = MB_post(fock=F, sigma=Sigma, mu=mu, gtau=G, S=S, beta=1000, lamb='1e4')
  G = manybody.gtau
  # If G(t) is not known, Dyson euqation can be solved on given beta and ir grid.
  manybody = MB_post(fock=F, sigma=Sigma, mu=mu, S=S, beta=1000, lamb='1e4')
  manybody.solve_dyson()
  G2 = manybody.gtau

  diff = G - G2
  print("Maximum G difference = ", np.max(np.abs(diff)))

  ''' Mulliken analysis '''
  print("Mullinken analysis: ")
  occs = manybody.mulliken_analysis()
  print("Spin up:", occs[0], ", Spin down:", occs[1])
  print("References: [0.5 0.5] and [0.5 0.5]")

  ''' Natural orbitals '''
  print("Natural orbitals: ")
  occ, no_coeff = manybody.get_no()
  print(occ[0,0])
  print(occ[1,0])

  #''' Maxent '''
  # Run Maxent for given Green's function, G_MoSum
  #manybody.analyt_cont(error=5e-3, maxent_exe='maxent', params='green.param', outdir='Maxent', gtau=G_MoSum)
  # By default, run Maxent for manybody.gtau
  #manybody.analyt_cont(error=5e-3, maxent_exe='maxent', params='green.param', outdir='Maxent')

  ''' Orthogonal input '''
  F_orth = orth.sao_orth(F, S, type='f')
  Sigma_orth = orth.sao_orth(Sigma, S, type='f')
  manybody = MB_post(fock=F_orth, sigma=Sigma_orth, mu=mu, beta=1000, lamb='1e4')

  print("Mullinken analysis: ")
  occs = manybody.mulliken_analysis()
  print("Spin up:", occs[0], ", Spin down:", occs[1])
  print("References: [0.5 0.5] and [0.5 0.5]")

  print("Natural orbitals: ")
  occ, no_coeff = manybody.get_no()
  print(occ[0, 0])
  print(occ[1, 0])

