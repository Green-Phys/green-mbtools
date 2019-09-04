import numpy as np
import scipy.linalg as LA
import h5py

import MB_analysis.spectral as spec
import MB_analysis.util as util
import MB_analysis.trans as trans

# TODO combine sim and inp files together
class rmb:
  '''Many-body analysis object'''
  def __init__(self, sim_path=None, inp_path = None):
    # Read data for analysis
    if sim_path is not None:
      sim = h5py.File(sim_path)
      _iter = sim["iter"][()]
      self.gtau = sim["/iter"+str(_iter)+"/G_tau/data"][()].view(np.complex)
      self.sigma = sim["/iter"+str(_iter)+"/Selfenergy/data"][()].view(np.complex)
      self.fock = sim["/iter"+str(_iter)+"/Fock-k"][()].view(np.complex)
      self.S = sim["/S-k"][()].view(np.complex)
      self.dm = -1 * self.gtau[-1]
      sim.close()
      # Dims

      self.nts = np.shape(self.gtau)[0]
      self.ink = np.shape(self.gtau)[1]
      self.nao = np.shape(self.gtau)[2]
      if inp_path is not None:
        inp = h5py.File(inp_path)
        self.ir_list = inp["/grid/ir_list"][()]
        self.weight = inp["/grid/weight"][()]
        inp.close()
      else:
        self.ir_list = np.arange(self.ink)
        self.weight = np.array([1 for i in range(self.ink)])

      rmb.compute_mo(self)
    elif sim_path is None:
      if inp_path is not None:
        inp = h5py.File(inp_path)
        self.gtau  = None
        self.sigma = None
        self.dm    = None
        self.fock  = inp["/HF/Fock-k"][()].view(np.complex)
        self.S     = inp["/HF/S-k"][()].view(np.complex)
        inp.close()
        # Dims
        self.nts = None
        self.ink = np.shape(self.fock)[0]
        self.nao = np.shape(self.fock)[1]

        rmb.compute_mo(self)
      else:
        self.gtau  = None
        self.sigma = None
        self.dm    = None
        self.fock  = None
        self.S     = None
        self.nts   = None
        self.ink   = None
        self.nao   = None
        self.mo_energy = None
        self.mo_coeff  = None

  # class variables



  # class functions
  def get_fock(self):
    return self.fock[:,:,:,0]

  # Get MO energy and orbitals
  def compute_mo(self):
    self.mo_energy, self.mo_coeff = spec.eig(self.fock[:,:,:,0],self.S[:,:,:,0])

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

  # Orthogonalization for Green's function and density matrix
  # object = (nk, nao, nao)
  def g_orthogonal(self, object):
    ink = np.shape(object)[0]
    if self.ink is None:
      self.ink = ink
    elif ink != self.ink and self.ink is not None:
      raise ValueError("Dims of k-points doesn't match S!")
    object_orth = np.zeros(np.shape(object), dtype=np.complex)
    for ik in range(self.ink):
      S12 = LA.sqrtm(self.S[ik, :, :, 0])
      object_orth[ik] = np.dot(S12, np.dot(object[ik, :, :], S12))

    return object_orth

  # Orthogonalization for Fock matrix and self-energy
  # object = (nk, nao, nao)
  def f_orthogonal(self, object):
    ink = np.shape(object)[0]
    if ink != self.ink:
      raise ValueError("Dims of k-points doesn't match S!")
    object_orth = np.zeros(np.shape(object), dtype=np.complex)
    for ik in range(self.ink):
      S12 = LA.sqrtm(self.S[ik, :, :, 0])
      S12_inv = np.linalg.inv(S12)
      object_orth[ik] = np.dot(S12_inv, np.dot(object[ik, :, :], S12_inv))

    return object_orth

  def get_mo_energy(self):
    mo_energy = np.asarray(self.mo_energy)
    return mo_energy

  def get_mo_coeff(self):
    mo_coeff = np.asarray(self.mo_coeff)
    return mo_coeff