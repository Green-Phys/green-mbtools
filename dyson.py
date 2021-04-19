import numpy as np

def solve_dyson(fock, S, sigma=None, mu=0, ir=None):
  '''
  Compute Green's function through Dyson's equation.
  :return:
  '''
  nts, nw = ir.nts, ir.nw
  ns, nk, nao = fock.shape[0], fock.shape[1], fock.shape[2]

  gtau = np.zeros((nts, ns, nk, nao, nao), dtype=np.complex)
  Gw = np.zeros((nw, nao, nao), dtype=np.complex)
  for s in range(ns):
    for k in range(nk):
      if sigma is not None:
        sigma_w = ir.tau_to_w(sigma[:, s, k])
      for n in range(nw):
        if sigma is None:
          tmp = (1j * ir.wsample[n] + mu) * S[s, k] - fock[s, k]
        else:
          tmp = (1j * ir.wsample[n] + mu) * S[s, k] - fock[s, k] - sigma_w[n]
        Gw[n] = np.linalg.inv(tmp)
      gtau[:, s, k] = ir.w_to_tau(Gw)

  dm = -1.0 * gtau[-1]
  return gtau, dm

