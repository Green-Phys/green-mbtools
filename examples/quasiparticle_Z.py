import h5py
import MB_analysis
import MB_analysis.src.quasiparticle as qs
import MB_analysis.src.orth as orth

sim_path = MB_analysis.__path__[0] + '/../data/H2_GW/sim.h5'
input_path = MB_analysis.__path__[0] + '/../data/H2_GW/input.h5'
manybody = MB_analysis.mb.initialize_MB_post(sim_path, input_path, '1e4')
nts, ns, nk, nao = manybody.sigma.shape[:4]
nw = manybody.ir.nw

'''
Quasiparticles have to be defined in an orthogonal orbitals basis.  
'''

''' Z in SAO basis '''
F_sao = orth.sao_orth(manybody.fock, manybody.S, type='f')
F_sao = F_sao[:,0]
Sigma_sao = orth.sao_orth(manybody.sigma, manybody.S, type='f')
Sigma_sao = manybody.ir.tau_to_w(Sigma_sao[:, :, 0])

MB_path = MB_analysis.__path__[0] + '/../'
nevan_sigma_exe = MB_path + '/Nevanlinna/nevanlinna_sigma'
Zs = qs.Z_factor(F_sao, Sigma_sao, manybody.ir.wsample, nevan_sigma_exe, 'nevan_sigma')
# Z = 0.98563148 from Nevanlinna AC with eta = 0.005 Ha
print("Quasiparticle renormalization factor: {}".format(Zs))

''' Z in one-electron basis of the Fock matrix '''
import scipy.linalg as LA
import numpy as np
from functools import reduce
Sigma = manybody.ir.tau_to_w(manybody.sigma[:,0,0])
#U^{\dag}*S*U = I and U^{\dag}*F*U = evals
e, U = LA.eigh(manybody.fock[0,0],manybody.S[0,0])
Sigma_w_mo = np.array([reduce(np.dot, (U.conj().T, sigma_w, U)) for sigma_w in Sigma])
F_mo = np.diag(e).reshape(1, nao, nao)
Sigma_w_mo = Sigma_w_mo.reshape(nw, 1, nao, nao)
MB_path = MB_analysis.__path__[0] + '/../'
nevan_sigma_exe = MB_path + '/Nevanlinna/nevanlinna_sigma'
Zs = qs.Z_factor(F_mo, Sigma_w_mo, manybody.ir.wsample, nevan_sigma_exe, 'nevan_sigma')
print("Quasiparticle renormalization factor: {}".format(Zs))