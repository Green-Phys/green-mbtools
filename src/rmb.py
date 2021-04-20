import numpy as np
import scipy.linalg as LA
import h5py

import MB_analysis.spectral as spec
import MB_analysis.util as util
import MB_analysis.trans as trans

# TODO combine sim and inp files together
class rmb:
  '''Many-body analysis class'''
  def __init__(self, fock, sigma=None, gtau=None, S=None, ir_list=None, weight=None):
    # TODO Sigma is optional
    if sigma.ndim == 5 and sigma.shape[1] == 2 and fock.ndim == 4 and fock.shape[0] == 2:
      self._ns = 2
    elif sigma.ndim == 4 and fock.ndim == 3:
      self._ns = 1
      gtau  = gtau.reshape((gtau.shape[0], 1) + gtau.shape[1:])
      sigma = sigma.reshape((sigma.shape[0], 1) + sigma.shape[1:])
      fock  = fock.reshape((1) + fock.shape)
      if S is not None:
        S   = S.reshape((1) + S.shape)
    else:
      raise ValueError('Incorrect dimensions of self-energy or Fock. '
                       'Accetable shapes are (nts, ns, nk, nao, nao) or (nts, nk, nao, nao) for self-energy and '
                       '(ns, nk, nao, nao) or (nk, nao, nao) for Fock matrix.')

    self._gtau = gtau.copy()
    self._sigma = sigma.copy()
    self._fock = fock.copy()
    if S is not None:
      self._S = S
    self._dm = -2 * gtau[-1]
    # Dims
    self._nts = self._gtau.shape[0]
    self._ink = self._gtau.shape[2]
    self._nao = self._gtau.shape[3]

    if ir_list is not None and weight is not None:
      self._ir_list = ir_list
      self._weight = weight
    else:
      self._ir_list = np.arange(self._ink)
      self._weight = np.array([1 for i in range(self._ink)])

    #rmb.compute_mo(self)

    # if self.S is not None:
    #  self.S_inv_12 = np.zeros(np.shape(self.S)[:-1],dtype=complex)
    #  for ss in range(self.ns):
    #    for ik in range(self.ink):
    #      S_12 = LA.sqrtm(self.S[ss,ik])
    #      self.S_inv_12[ss,ik] = np.linalg.inv(S_12)

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

  # Public class variables
  beta = None
  mo_energy = None
  mo_coeff = None

  # class functions
  def fock(self):
    return self.fock[:,:,:]

  # TODO move out of mb class
  def compute_mo(self):
    '''
    Compute molecular orbital energy by solving FC=SCE
    :return:
    '''
    self.mo_energy = np.zeros(np.shape(self._fock)[:-1])
    self.mo_coeff = np.zeros(np.shape(self._fock), dtype=complex)
    #self._mo_k_energy, self._mo_k_coeff = spec.eig(self._fock[:,:,:],self._S[:,:,:])
    self.mo_energy, self.mo_coeff = spec.eig(self._fock, self._S)

    return self.mo_energy, self.mo_coeff

  # TODO move out of mb class
  def compute_no(self):
    '''
    Compute natural orbitals by diagonalizing density matrix
    :return:
    '''
    dm_orth = trans.orthogonal(self._dm, self._S, 'g')
    occ = np.zeros((self._ns, self._ink, self._nao))
    no_coeff = np.zeros((self._ns, self._ink, self._nao, self._nao), dtype=complex)
    for ss in range(self._ns):
      for ik in range(self._ink):
        occ[ss,ik], no_coeff[ss,ik] = np.linalg.eigh(dm_orth[ss,ik])

    occ, no_coeff = occ[:, :, ::-1], no_coeff[:, :, :, ::-1]
    return occ, no_coeff

  # TODO Refactor for below are not done yet
  def sigma_w_ir(self, beta=None, iw_list=None, ir_path = None):
    if beta is not None:
      self.beta = beta
    if iw_list is not None:
      self.iw_list = iw_list

    if self.beta is None:
      raise ValueError("Define beta first!")
    if self.iw_list is None:
      raise ValueError("Define iw_list first!")
    if ir_path is None:
      raise ValueError("Please specify ir_path!")
    nw = np.shape(self._iw_list)[0]
    #sig_w = np.zeros((nw, self._ns, self._ink, self._nao, self._nao), dtype=np.complex)
    sig_w = trans.tau_to_w_ir(self._sigma, ir_path, self.beta)
    sig_w = sig_w.reshape(nw, self._ink, self._nao, self._nao)
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

if __name__ == '__main__':
  # TODO Examples and tests needed.