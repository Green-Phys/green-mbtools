import numpy as np
import scipy.linalg as LA
import h5py

import MB_analysis.spectral as spec
import MB_analysis.util as util
import MB_analysis.trans as trans


# TODO combine sim and inp files together
class umb:
  '''Many-body analysis object'''
  def __init__(self, gtau, sigma, fock, S, ir_list=None, weight=None):
    self._gtau = gtau
    self._sigma = sigma
    self._fock = fock
    self._S = S
    self._dm = -1 * gtau[-1]
    # Dims
    self._nts = np.shape(self._gtau)[0]
    self._ns = np.shape(self._gtau)[1]
    self._ink = np.shape(self._gtau)[2]
    self._nao = np.shape(self._gtau)[3]

    if ir_list is not None and weight is not None:
      self._ir_list = ir_list
      self._weight = weight
    else:
      self._ir_list = np.arange(self._ink)
      self._weight = np.array([1 for i in range(self._ink)])

    umb.compute_mo(self)

    #if self.S is not None:
    #  self.S_inv_12 = np.zeros(np.shape(self.S)[:-1],dtype=complex)
    #  for ss in range(self.ns):
    #    for ik in range(self.ink):
    #      S_12 = LA.sqrtm(self.S[ss,ik])
    #      self.S_inv_12[ss,ik] = np.linalg.inv(S_12)

  # class variables
  _gtau  = None
  _sigma = None
  _dm    = None
  _fock  = None
  _S     = None
  _ns    = None
  _nts   = None
  _beta  = None
  _ink   = None
  _nao   = None
  _iw_list = None
  _mo_k_energy = None
  _mo_k_coeff  = None
  _no        = None
  _no_coeff  = None
  _S_inv_12  = None
  _ir_list = None
  _weight = None

  # class functions
  def fock(self):
    return self._fock[:,:,:,:]

  # Get MO energy and orbitals
  def compute_mo(self):
    self._mo_k_energy = np.zeros(np.shape(self._fock)[:-1])
    self._mo_k_coeff = np.zeros(np.shape(self._fock), dtype=complex)
    for ss in range(self._ns):
      self._mo_k_energy[ss], self._mo_k_coeff[ss] = spec.eig(self._fock[ss,:,:,:], self._S[ss,:,:,:])

  # Local occupation numbers and natural orbitals
  def get_no(self):
    dm_orth = trans.orthogonal(self._dm, self._S, 'g')
    local_dm = np.zeros((self._ns,self._nao,self._nao))
    occ      = np.zeros((self._ns, self._nao))
    no_coeff = np.zeros((self._ns, self._nao, self._nao))
    for ss in range(self._ns):
      local_dm[ss] = util.to_local(dm_orth[ss], self._ir_list, self._weight)
      # By default, the eigenvalues are in ascending order
      occ[ss], no_coeff[ss] = np.linalg.eigh(local_dm[ss])
    occ = occ[:,::-1]
    no_coeff = no_coeff[:,:,::-1]
    return occ, no_coeff

  # k-dependence natural orbitals
  def get_no_k(self):
    dm_orth = trans.orthogonal(self._dm, self._S, 'g')
    occ = np.zeros((self._ns, self._ink, self._nao))
    no_coeff = np.zeros((self._ns, self._ink, self._nao, self._nao), dtype=complex)
    for ss in range(self._ns):
      for ik in range(self._ink):
        occ[ss,ik], no_coeff[ss,ik] = np.linalg.eigh(dm_orth[ss,ik])

    occ, no_coeff = occ[:, :, ::-1], no_coeff[:, :, :, ::-1]
    return occ, no_coeff

  def sigma_w_ir(self, ir_path = None):
    if self._beta is None:
      raise ValueError("Define _beta first!")
    if self._iw_list is None:
      raise ValueError("Define _iw_list first!")
    if ir_path is None:
      raise ValueError("Please specify ir_path!")
    nw = np.shape(self._iw_list)[0]
    #sig_w = np.zeros((nw, self._ns, self._ink, self._nao, self._nao), dtype=np.complex)
    sig_w = trans.tau_to_w_ir(self._sigma, ir_path, self._beta)
    sig_w = sig_w.reshape(nw, self._ns, self._ink, self._nao, self._nao)
    return sig_w

  def g_w_ir(self, ir_path = None):
    if self._beta is None:
      raise ValueError("Define _beta first!")
    if self._iw_list is None:
      raise ValueError("Define _iw_list first!")
    if ir_path is None:
      raise ValueError("Please specify ir_path!")
    nw = np.shape(self._iw_list)[0]
    #sig_w = np.zeros((nw, self._ns, self._ink, self._nao, self._nao), dtype=np.complex)
    g_w = trans.tau_to_w_ir(self._gtau, ir_path, self._beta)
    g_w = g_w.reshape(nw, self._ns, self._ink, self._nao, self._nao)
    return g_w




# TODO Refactor for below are not done yet
  # From AO in k-space to NO_k in k-space
  def ao_to_no_k(self,object, type='f'):
    occ, no_coeff = umb.get_no_k(self)
    object_no_k = np.zeros(np.shape(object), dtype=np.complex)
    if type == 'g':
      tmp = np.einsum('skij,skjl->skil', object, no_coeff)
      object_no_k = np.einsum('skji,skjl->skil', np.conjugate(no_coeff), tmp)
    elif type == 'f':
      for ss in range(self._ns):
        for ik in range(self._ink):
          no_coeff_H = np.transpose(np.conjugate(no_coeff[ss,ik]))
          no_coeff_inv = np.linalg.inv(no_coeff_H)
          tmp = np.einsum('ij,jk->ik', object[ss,ik], no_coeff_inv)
          object_no_k[ss,ik] = np.einsum('ji,jk->ik', np.conjugate(no_coeff_inv), tmp)
    else:
      raise ValueError("Wrong type of transformation!")
    return object_no_k

  # Transform object from AO to MO basis all in k-space. object = (ns,nk,nao,nao)
  # type f : for Fock matrix and self-energy
  # type g : for Green's function and density matrix
  def ao_to_mo(self, object, type='f'):
    object_mo = np.zeros(np.shape(object), dtype=np.complex)
    if type == 'f':
      tmp = np.einsum('skij,skjl->skil', object, self._mo_coeff)
      object_mo = np.einsum('skji,skjl->skil', np.conjugate(self._mo_coeff), tmp)
    elif type == 'g':
      for ss in range(self._ns):
        for ik in range(self._ink):
          mo_coeff_H = np.transpose(np.conjugate(self._mo_coeff[ss, ik]))
          mo_coeff_inv = np.linalg.inv(mo_coeff_H)
          tmp = np.einsum('ij,jk->ik', object[ss, ik], mo_coeff_inv)
          object_mo[ss, ik] = np.einsum('ji,jk->ik', np.conjugate(mo_coeff_inv), tmp)
    else:
      raise ValueError("Wrong type of transformation!")
    return object_mo

  # FIXME not yet done
  # From AO in k-space to local NO basis
  def ao_to_no(self,object, type='f'):
    object_ao_orth = np.zeros(np.shape(object), dtype=complex)
    if type == 'f':
      for ss in range(self._ns):
        object_ao_orth[ss] = umb.f_orthogonal(self,object[ss])
    elif type == 'g':
      for ss in range(self._ns):
        object_ao_orth[ss] = umb.g_orthogonal(self,object[ss])
    else:
      raise ValueError("Wrong type of transformation!")
    occ, no_coeff = umb.get_no(self)
    for ik in range(self._ink):
      object_ao_orth[ik] = np.einsum('ij,jk->ik',object_ao_orth[ik],no_coeff)
      object_ao_orth[ik] = np.einsum('ji,jk->ik',np.conjugate(no_coeff),object_ao_orth[ik])
    return object_ao_orth

  def get_mo_k_energy(self):
    mo_k_energy = np.asarray(self._mo_k_energy)
    return mo_k_energy

  def get_mo_k_coeff(self):
    mo_k_coeff = np.asarray(self._mo_k_coeff)
    return mo_k_coeff

  def mulliken_charge_gamma(self, orbitals, Z):
    dm = self._dm[:,0,:,:,0].real
    S  = self._S[:,0,:,:,0].real

    e = 0.0
    for ss in range(self._ns):
      for i in orbitals:
        for j in range(self._nao):
          e -= dm[ss,i,j] * S[ss,j,i]

    return Z + e

  def mulliken_magnetic_moment_gamma(self, orbitals):
    dm = self._dm[:,0,:,:,0].real
    S  = self._S[:,0,:,:,0].real

    n_a = 0.0
    n_b = 0.0
    for i in orbitals:
      for j in range(self._nao):
        n_a += dm[0,i,j] * S[0,j,i]
        n_b += dm[1,i,j] * S[1,j,i]

    mu = n_a - n_b
    return mu



  # Mulliken analysis for charges
  def mulliken_charge_tmp(self, orbitals, Z):
    dm_orth = umb.g_orthogonal(self, self._dm[:,:,:,:,0])
    dm_loc = []
    S_loc  = []
    for ss in range(self._ns):
      dm = util.to_local(dm_orth[ss,:,:,:],self._ir_list,self._weight)
      S  = util.to_local(self._S[ss,:,:,:,0],self._ir_list,self._weight)
      dm_loc.append(dm)
      S_loc.append(S)
    dm_loc = np.asarray(dm_loc)
    S_loc  = np.asarray(S_loc)

    e = 0.0
    for ss in range(self._ns):
      for i in orbitals:
        #for j in range(self.nao):
          #e -= dm_loc[ss,i,j] * S_loc[ss,j,i]
        e -= dm_loc[ss,i,i]

    return Z + e

  def mulliken_charge(self, orbitals, Z):
    e = 0.0
    for ik in range(self._ink):
      n_k = 0.0
      for ss in range(self._ns):
        for i in orbitals:
          for j in range(self._nao):
            n_k += self._dm[ss,ik,i,j] * self._S[ss,ik,j,i]
      k_ir = self._ir_list[ik]
      e -= self._weight[k_ir] * n_k
    num_k = len(self._weight)
    e /= num_k
    return Z + e.real


  def mulliken_mu(self, orbitals):
    na, nb = 0.0, 0.0
    for ik in range(self._ink):
      na_k = 0.0
      nb_k = 0.0
      for i in orbitals:
        for j in range(self._nao):
          na_k += self._dm[0,ik,i,j] * self._S[0,ik,j,i]
          nb_k += self._dm[1,ik,i,j] * self._S[1,ik,j,i]
      k_ir = self._ir_list[ik]
      na += self._weight[k_ir] * na_k
      nb += self._weight[k_ir] * nb_k
    num_k = len(self._weight)
    na /= num_k
    nb /= num_k
    mu = na.real - nb.real

    return mu
