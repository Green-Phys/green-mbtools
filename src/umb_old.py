import numpy as np
import scipy.linalg as LA
import h5py

import MB_analysis.spectral as spec
import MB_analysis.util as util
import MB_analysis.trans as trans

class umb:
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
      self.ns  = np.shape(self.gtau)[1]
      self.ink = np.shape(self.gtau)[2]
      self.nao = np.shape(self.gtau)[3]
      if inp_path is not None:
        inp = h5py.File(inp_path)
        self.ir_list = inp["/grid/ir_list"][()]
        self.weight = inp["/grid/weight"][()]
        inp.close()
      else:
        self.ir_list = np.arange(self.ink)
        self.weight = np.array([1 for i in range(self.ink)])
      # Initialize mo informations
      self.mo_energy = np.zeros(np.shape(self.fock)[:3])
      self.mo_coeff = np.zeros(np.shape(self.fock)[:4], dtype=complex)
      umb.compute_mo(self)
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
        self.ns  = np.shape(self.fock)[0]
        self.ink = np.shape(self.fock)[1]
        self.nao = np.shape(self.fock)[2]
        # Initialize mo informations
        self.mo_energy = np.zeros(np.shape(self.fock)[:3])
        self.mo_coeff = np.zeros(np.shape(self.fock)[:4], dtype=complex)
        umb.compute_mo(self)

    #if self.S is not None:
    #  self.S_inv_12 = np.zeros(np.shape(self.S)[:-1],dtype=complex)
    #  for ss in range(self.ns):
    #    for ik in range(self.ink):
    #      S_12 = LA.sqrtm(self.S[ss,ik])
    #      self.S_inv_12[ss,ik] = np.linalg.inv(S_12)

  # class variables
  gtau  = None
  sigma = None
  dm    = None
  fock  = None
  S     = None
  ns    = None
  nts   = None
  ink   = None
  nao   = None
  mo_energy = None
  mo_coeff  = None
  no        = None
  no_coeff  = None
  S_inv_12  = None


  # class functions
  def get_fock(self):
    return self.fock[:,:,:,:,0]

  # Get MO energy and orbitals
  def compute_mo(self):
    if self.ns is None:
      self.ns = np.shape(self.fock)[0]
    if self.mo_energy is None:
      self.mo_energy = np.zeros(np.shape(self.fock)[:3])
      self.mo_coeff = np.zeros(np.shape(self.fock)[:4], dtype=complex)

    for ss in range(self.ns):
      self.mo_energy[ss], self.mo_coeff[ss] = spec.eig(self.fock[ss,:,:,:,0],self.S[ss,:,:,:,0])

  # Local occupation numbers and natural orbitals
  def get_no(self):
    if self.ns is None:
      self.ns, self.nao = np.shape(self.dm)[0], np.shape(self.dm)[2]
    dm_orth  = umb.g_orthogonal(self, self.dm[:, :, :, :, 0])
    local_dm = np.zeros((self.ns,self.nao,self.nao))
    occ      = np.zeros((self.ns, self.nao))
    no_coeff = np.zeros((self.ns, self.nao, self.nao))
    for ss in range(self.ns):
      local_dm[ss] = util.to_local(dm_orth[ss], self.ir_list, self.weight)
      # By default, the eigenvalues are in ascending order
      occ[ss], no_coeff[ss] = np.linalg.eigh(local_dm[ss])
    occ = occ[:,::-1]
    no_coeff = no_coeff[:,:,::-1]
    return occ, no_coeff

  # k-dependence natural orbitals
  def get_no_k(self):
    S_inv = np.zeros(np.shape(self.S[:, :, :, :, 0]), dtype=np.complex)
    for ss in range(self.ns):
      for ik in range(self.ink):
        S_inv[ss, ik] = np.linalg.inv(self.S[ss, ik, :, :, 0])

    occ = np.zeros((self.ns, self.ink, self.nao))
    no_coeff = np.zeros((self.ns, self.ink, self.nao, self.nao),dtype=complex)
    for ss in range(self.ns):
      occ[ss], no_coeff[ss] = spec.eig(self.dm[ss, :, :, :, 0], S_inv[ss, :, :, :])

    occ, no_coeff = occ[:,:,::-1], no_coeff[:,:,:,::-1]
    return occ, no_coeff

  # Get Fermionic Green's function in Matsubara axis with  n = 0 (AO basis)
  def get_g_w0(self):
    gw0 = np.zeros((self.ns, self.ink, self.nao, self.nao), dtype=np.complex)
    for ss in range(self.ns):
      gtau = self.gtau[:,ss,:,:,:,0]
      gw0[ss] = trans.sigma_zero(gtau)
    return gw0

  def get_sigma_w0(self):
    sig_w0 = np.zeros((self.ns, self.ink, self.nao, self.nao), dtype=np.complex)
    for ss in range(self.ns):
      sig_tau = self.sigma[:,ss,:,:,:,0]
      sig_w0[ss] = trans.sigma_zero(sig_tau)
    return sig_w0

  # sig_w = (nw, ns, nk, nao, nao)
  def get_sigma_w(self, nw):
    sig_w = np.zeros((nw, self.ns, self.ink, self.nao, self.nao), dtype=np.complex)
    for ss in range(self.ns):
      sig_tau = self.sigma[:,ss,:,:,:,0]
      sig_w[:,ss,:,:,:] = trans.sigma_w(sig_tau, nw)
    return sig_w

  # From AO in k-space to NO_k in k-space
  def ao_to_no_k(self,object, type='f'):
    occ, no_coeff = umb.get_no_k(self)
    object_no_k = np.zeros(np.shape(object), dtype=np.complex)
    if type == 'g':
      tmp = np.einsum('skij,skjl->skil', object, no_coeff)
      object_no_k = np.einsum('skji,skjl->skil', np.conjugate(no_coeff), tmp)
    elif type == 'f':
      for ss in range(self.ns):
        for ik in range(self.ink):
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
      tmp = np.einsum('skij,skjl->skil', object, self.mo_coeff)
      object_mo = np.einsum('skji,skjl->skil', np.conjugate(self.mo_coeff), tmp)
    elif type == 'g':
      for ss in range(self.ns):
        for ik in range(self.ink):
          mo_coeff_H = np.transpose(np.conjugate(self.mo_coeff[ss, ik]))
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
      for ss in range(self.ns):
        object_ao_orth[ss] = umb.f_orthogonal(self,object[ss])
    elif type == 'g':
      for ss in range(self.ns):
        object_ao_orth[ss] = umb.g_orthogonal(self,object[ss])
    else:
      raise ValueError("Wrong type of transformation!")
    occ, no_coeff = umb.get_no(self)
    for ik in range(self.ink):
      object_ao_orth[ik] = np.einsum('ij,jk->ik',object_ao_orth[ik],no_coeff)
      object_ao_orth[ik] = np.einsum('ji,jk->ik',np.conjugate(no_coeff),object_ao_orth[ik])
    return object_ao_orth

  # Orthogonalization for Green's function and density matrix
  # object = (ns, nk, nao, nao)
  def g_orthogonal(self, object):
    ns  = np.shape(object)[0]
    ink = np.shape(object)[1]
    if self.ink is None:
      self.ink, self.ns = ink, ns
    elif ink != self.ink and self.ink is not None:
      raise ValueError("Dims of k-points doesn't match S!")
    object_orth = np.zeros(np.shape(object), dtype=np.complex)
    for ss in range(self.ns):
      for ik in range(self.ink):
        S12 = LA.sqrtm(self.S[ss, ik, :, :, 0])
        object_orth[ss, ik] = np.dot(S12, np.dot(object[ss, ik, :, :], S12))

    return object_orth

  # Orthogonalization for Fock matrix and self-energy
  # object = (ns, nk, nao, nao)
  def f_orthogonal(self, object):
    ns  = np.shape(object)[0]
    ink = np.shape(object)[1]
    if self.ink is None:
      self.ink, self.ns = ink, ns
    elif ink != self.ink and self.ink is not None:
      raise ValueError("Dims of k-points doesn't match S!")
    object_orth = np.zeros(np.shape(object), dtype=np.complex)
    for ss in range(self.ns):
      for ik in range(self.ink):
        S12 = LA.sqrtm(self.S[ss, ik, :, :, 0])
        S12_inv = np.linalg.inv(S12)
        object_orth[ss,ik] = np.dot(S12_inv, np.dot(object[ss, ik, :, :], S12_inv))

    return object_orth

  def get_mo_energy(self):
    mo_energy = np.asarray(self.mo_energy)
    return mo_energy

  def get_mo_coeff(self):
    mo_coeff = np.asarray(self.mo_coeff)
    return mo_coeff

  def mulliken_charge_gamma(self, orbitals, Z):
    dm = self.dm[:,0,:,:,0].real
    S  = self.S[:,0,:,:,0].real

    e = 0.0
    for ss in range(self.ns):
      for i in orbitals:
        for j in range(self.nao):
          e -= dm[ss,i,j] * S[ss,j,i]

    return Z + e

  def mulliken_magnetic_moment_gamma(self, orbitals):
    dm = self.dm[:,0,:,:,0].real
    S  = self.S[:,0,:,:,0].real

    n_a = 0.0
    n_b = 0.0
    for i in orbitals:
      for j in range(self.nao):
        n_a += dm[0,i,j] * S[0,j,i]
        n_b += dm[1,i,j] * S[1,j,i]

    mu = n_a - n_b
    return mu



  # Mulliken analysis for charges
  def mulliken_charge_tmp(self, orbitals, Z):
    dm_orth = umb.g_orthogonal(self, self.dm[:,:,:,:,0])
    dm_loc = []
    S_loc  = []
    for ss in range(self.ns):
      dm = util.to_local(dm_orth[ss,:,:,:],self.ir_list,self.weight)
      S  = util.to_local(self.S[ss,:,:,:,0],self.ir_list,self.weight)
      dm_loc.append(dm)
      S_loc.append(S)
    dm_loc = np.asarray(dm_loc)
    S_loc  = np.asarray(S_loc)

    e = 0.0
    for ss in range(self.ns):
      for i in orbitals:
        #for j in range(self.nao):
          #e -= dm_loc[ss,i,j] * S_loc[ss,j,i]
        e -= dm_loc[ss,i,i]

    return Z + e

  def mulliken_charge(self, orbitals, Z):
    e = 0.0
    for ik in range(self.ink):
      n_k = 0.0
      for ss in range(self.ns):
        for i in orbitals:
          for j in range(self.nao):
            n_k += self.dm[ss,ik,i,j,0] * self.S[ss,ik,j,i,0]
      k_ir = self.ir_list[ik]
      e -= self.weight[k_ir] * n_k
    num_k = len(self.weight)
    e /= num_k
    return Z + e.real


  def mulliken_mu(self, orbitals):
    na, nb = 0.0, 0.0
    for ik in range(self.ink):
      na_k = 0.0
      nb_k = 0.0
      for i in orbitals:
        for j in range(self.nao):
          na_k += self.dm[0,ik,i,j,0] * self.S[0,ik,j,i,0]
          nb_k += self.dm[1,ik,i,j,0] * self.S[1,ik,j,i,0]
      k_ir = self.ir_list[ik]
      na += self.weight[k_ir] * na_k
      nb += self.weight[k_ir] * nb_k
    num_k = len(self.weight)
    na /= num_k
    nb /= num_k
    mu = na.real - nb.real

    return mu