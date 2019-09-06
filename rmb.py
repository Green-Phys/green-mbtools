import numpy as np
import scipy.linalg as LA
import h5py

import MB_analysis.spectral as spec
import MB_analysis.util as util
import MB_analysis.trans as trans

# TODO combine sim and inp files together
class rmb:
  '''Many-body analysis object'''
  def __init__(self, gtau, sigma, fock, S, ir_list=None, weight=None):
    self._gtau = gtau
    self._sigma = sigma
    self._fock = fock
    self._S = S
    self._dm = -2 * gtau[-1]
    # Dims
    self._nts = np.shape(self._gtau)[0]
    self._ink = np.shape(self._gtau)[1]
    self._nao = np.shape(self._gtau)[2]

    if ir_list is not None and weight is not None:
      self._ir_list = ir_list
      self._weight = weight
    else:
      self._ir_list = np.arange(self._ink)
      self._weight = np.array([1 for i in range(self._ink)])

    rmb.compute_mo(self)

    # if self.S is not None:
    #  self.S_inv_12 = np.zeros(np.shape(self.S)[:-1],dtype=complex)
    #  for ss in range(self.ns):
    #    for ik in range(self.ink):
    #      S_12 = LA.sqrtm(self.S[ss,ik])
    #      self.S_inv_12[ss,ik] = np.linalg.inv(S_12)

    # class variables

  _gtau = None
  _sigma = None
  _dm = None
  _fock = None
  _S = None
  _nts = None
  _beta = None
  _ink = None
  _nao = None
  _iw_list = None
  _mo_k_energy = None
  _mo_k_coeff = None
  _no = None
  _no_coeff = None
  _S_inv_12 = None
  _ir_list = None
  _weight = None



  # class functions
  def fock(self):
    return self.fock[:,:,:]

  # Get MO energy and orbitals
  def compute_mo(self):
    self._mo_k_energy = np.zeros(np.shape(self._fock)[:-1])
    self._mo_k_coeff = np.zeros(np.shape(self._fock), dtype=complex)
    self._mo_k_energy, self._mo_k_coeff = spec.eig(self._fock[:,:,:],self._S[:,:,:])

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
    sig_w = sig_w.reshape(nw, self._ink, self._nao, self._nao)
    return sig_w

  # TODO Refactor for below are not done yet
  # Get local occupation numbers
  def get_occ(self):
    dm_orth = rmb.g_orthogonal(self, self.dm[:, :, :, 0])
    local_dm = util.loc_dens(dm_orth, self.ir_list, self.weight)
    # By default, the eigenvalues are in ascending order
    occ = np.linalg.eigvalsh(local_dm)
    occ = occ[::-1]
    return occ

  # Natural orbitals information
  def get_no(self):
    dm_orth = rmb.g_orthogonal(self, self.dm[:, :, :, 0])
    local_dm = util.loc_dens(dm_orth, self.ir_list, self.weight)
    occ, no_coeff = np.linalg.eigh(local_dm)
    occ = occ[::-1]
    no_coeff = no_coeff[:,::-1]
    return occ, no_coeff

  def get_kno(self):
    S_inv = np.zeros(np.shape(self.S[:, :, :, 0]), dtype=np.complex)
    for ik in range(self.ink):
      S_inv[ik] = np.linalg.inv(self.S[ik, :, :, 0])

    occ, no_coeff = spec.eig(self.dm[:, :, :, 0], S_inv[:, :, :])
    occ, no_coeff = occ[:, ::-1], no_coeff[:, :, ::-1]
    return occ, no_coeff

  # Get Fermionic Green's function in Matsubara axis with  n = 0 (AO basis)
  def get_g_w0(self):
    gtau = self.gtau[:,:,:,:,0]
    return trans.sigma_zero(gtau)

  def get_sigma_w0(self):
    sig_tau = self.sigma[:,:,:,:,0]
    return trans.sigma_zero(sig_tau)

  def get_sigma_w(self, nw):
    sig_tau = self.sigma[:,:,:,:,0]
    return trans.sigma_w(sig_tau, nw)

  def ao_to_kno(self,object, type='f'):
    occ, no_coeff = rmb.get_kno(self)

    object_no = np.zeros(np.shape(object), dtype=np.complex)
    if type == 'g':
      for ik in range(self.ink):
        tmp = np.einsum('ij,jk->ik', object[ik], no_coeff[ik])
        object_no[ik] = np.einsum('ji,jk->ik', np.conjugate(no_coeff[ik]), tmp)
    if type == 'f':
      for ik in range(self.ink):
        no_coeff_H = np.transpose(np.conjugate(no_coeff[ik]))
        no_coeff_inv = np.linalg.inv(no_coeff_H)
        tmp = np.einsum('ij,jk->ik', object[ik], no_coeff_inv)
        object_no[ik] = np.einsum('ji,jk->ik', np.conjugate(no_coeff_inv), tmp)
    return object_no

  def ao_to_no(self,object, type='f'):
    if type == 'f':
      object_no = rmb.f_orthogonal(self,object)
    elif type == 'g':
      object_no = rmb.g_orthogonal(self,object)
    else:
      raise ValueError("Wrong type of transformation!")
    occ, no_coeff = rmb.get_no(self)
    for ik in range(self.ink):
      object_no[ik] = np.einsum('ij,jk->ik',object_no[ik],no_coeff)
      object_no[ik] = np.einsum('ji,jk->ik',np.conjugate(no_coeff),object_no[ik])
    return object_no

  # Transform object from orthogonal to MO basis.
  def orth_to_mo(self, object):
    if self.mo_coeff is None or self.mo_energy is None:
      rmb.compute_mo(self)
    object_mo = np.zeros(np.shape(object), dtype=np.complex)
    for ik in range(self.ink):
      tmp = np.einsum('ij,jk->ik',object[ik], self.mo_coeff[ik])
      #object_mo[ik, :, :] = np.dot(object[ik, :, :], self.mo_coeff[ik, :, :])
      object_mo[ik] = np.einsum('ji,jk->ik',np.conjugate(self.mo_coeff[ik]),tmp)
      #object_mo[ik, :, :] = np.dot(np.conjugate(np.transpose(self.mo_coeff[ik, :, :])), object_mo[ik, :, :])
    return object_mo

  # Transform object from AO to MO basis.
  # type f : for Fock matrix and self-energy
  # type g : for Green's function and density matrix
  def ao_to_mo(self, object, type = 'f'):
    #if type == 'f':
    #  object_mo = mb.f_orthogonal(self,object)
    #elif type == 'g':
    #  object_mo = mb.g_orthogonal(self,object)
    #else:
    #  raise ValueError("Wrong type of transformation!")
    object_mo = np.zeros(np.shape(object), dtype=np.complex)
    for ik in range(self.ink):
      tmp = np.einsum('ij,jk->ik',object[ik],self.mo_coeff[ik])
      object_mo[ik] = np.einsum('ji,jk->ik',np.conjugate(self.mo_coeff[ik]),tmp)
      #object_mo[ik, :, :] = np.dot(object_mo[ik, :, :], self.mo_coeff[ik, :, :])
      #object_mo[ik, :, :] = np.dot(np.conjugate(np.transpose(self.mo_coeff[ik, :, :])), object_mo[ik, :, :])
    return object_mo

  def get_mo_energy(self):
    mo_energy = np.asarray(self.mo_energy)
    return mo_energy

  def get_mo_coeff(self):
    mo_coeff = np.asarray(self.mo_coeff)
    return mo_coeff