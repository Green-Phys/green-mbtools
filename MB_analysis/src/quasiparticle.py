import os
import h5py
import numpy as np
import MB_analysis.src.analyt_cont as ac

'''
Perform quasiparticle approximation
'''

'''
Construct quasiparticle renormalization factors in a local Hamiltonian  
F = (ns, nao, nao)
Sigma_iw = (nw, ns, nao, nao)
'''
def Z_factor(F, Sigma_iw, iwsample, nevan_sigma_exe, outdir='sigma_nevan'):
  ns, nao = F.shape[:2]
  # Nevanlinna continuation of self-energy.
  Sigma_iw_diag = np.einsum('wsii->wsi', Sigma_iw)
  print("Sigma_iw_diag shape = {}".format(Sigma_iw_diag.shape))
  nw = iwsample.shape[0]
  iwsample_pos = iwsample[nw//2:]
  Sigma_iw_diag = Sigma_iw_diag[nw//2:]
  input_parser = 'Sigma_iw.txt ' + str(nw//2) + ' Sigma_w.txt coeff'
  ac.nevan_run_selfenergy(Sigma_iw_diag, iwsample_pos, input_parser, nevan_sigma_exe, outdir)

  if not os.path.exists(outdir):
    raise ValueError("Directory {} doesn't exist!".format(outdir))
  f = h5py.File(outdir+"/Sigma_w.h5", 'r')
  # Only diagonal terms are analytically continued. Sigma_w = (freqs, ns, nao)
  Sigma_w = f["Sigma_w"][()].view(complex)
  freqs = f["freqs"][()]
  f.close()

  # H0_diag = (ns, nno)
  F_diag = np.array([np.diag(f) for f in F])
  Fqs = F.copy()
  #delta_qs = delta.copy() if delta is not None else None
  dw = freqs[1] - freqs[0]
  Zs = np.zeros((ns, nao))
  for s in range(ns):
    for i in range(nao):
      e_approx, idx = find_nearest(freqs, F_diag[s,i])
      # Z = [1 - d(Sigma)/d(w)|w=e]^{-1}
      # Z^{-1} >= 1
      Z = 1 - np.gradient(Sigma_w[:, s, i].real, dw)
      Z = 1 / Z
      Zs[s,i] = Z[idx]

  return Zs

def find_nearest(array,value):
  import math
  idx = np.searchsorted(array, value, side="left")
  if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
    return array[idx-1], idx-1
  else:
    return array[idx], idx



if __name__ == "__main__":
  import h5py
  import MB_analysis
  import MB_analysis.src.orth as orth

  sim_path = MB_analysis.__path__[0] + '/../data/H2_GW/sim.h5'
  input_path = MB_analysis.__path__[0] + '/../data/H2_GW/input.h5'
  manybody = MB_analysis.mb.initialize_MB_post(sim_path, input_path, '1e4')

  F_sao = orth.sao_orth(manybody.fock, manybody.S, type='f')
  F_sao = F_sao[:,0]
  Sigma_sao = orth.sao_orth(manybody.sigma, manybody.S, type='f')
  Sigma_sao = manybody.ir.tau_to_w(Sigma_sao[:, :, 0])

  MB_path = MB_analysis.__path__[0] + '/../'
  nevan_sigma_exe = MB_path + '/Nevanlinna/nevanlinna_sigma'
  Zs = Z_factor(F_sao, Sigma_sao, manybody.ir.wsample, nevan_sigma_exe, 'nevan_sigma')
  print("Quasiparticle renormalization factor: {}", Zs)