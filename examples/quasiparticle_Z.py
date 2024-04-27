from functools import reduce
from mbanalysis.mb import initialize_MB_post
import mbanalysis.quasiparticle as qs
import mbanalysis.orth as orth
import scipy.linalg as LA
import numpy as np


#
# Initialize the mbanalysis object
#

data_dir = '../tests/test_data'
sim_path = data_dir + '/H2_GW/sim.h5'
input_path = data_dir + '/H2_GW/input.h5'
ir_file = data_dir + '/ir_grid/1e4_104.h5'
manybody = initialize_MB_post(sim_path, input_path, ir_file)
nts, ns, nk, nao = manybody.sigma.shape[:4]
nw = manybody.ir.nw


#
# Quasiparticles have to be defined in an orthogonal orbitals basis.
# Note: this example only calculates Z at gamma point.
#

''' Z in SAO basis '''
F_sao = orth.sao_orth(manybody.fock, manybody.S, type='f')
F_sao = F_sao[:, 0]
Sigma_sao = orth.sao_orth(manybody.sigma, manybody.S, type='f')
Sigma_sao = manybody.ir.tau_to_w(Sigma_sao[:, :, 0])

Zs = qs.Z_factor(F_sao, Sigma_sao, manybody.ir.wsample)
# Z = 0.98563148 from Nevanlinna AC with eta = 0.005 Ha
print('In SAO basis:')
print("Quasiparticle renormalization factor: {}".format(Zs))


''' Z in one-electron basis of the Fock matrix '''
Sigma = manybody.ir.tau_to_w(manybody.sigma[:, 0, 0])
# U^{\dag}*S*U = I and U^{\dag}*F*U = evals
e, U = LA.eigh(manybody.fock[0, 0], manybody.S[0, 0])
Sigma_w_mo = np.array(
    [reduce(np.dot, (U.conj().T, sigma_w, U)) for sigma_w in Sigma]
)
F_mo = np.diag(e).reshape(1, nao, nao)
Sigma_w_mo = Sigma_w_mo.reshape(nw, 1, nao, nao)
Zs = qs.Z_factor(F_mo, Sigma_w_mo, manybody.ir.wsample)
print('In MO basis:')
print("Quasiparticle renormalization factor: {}".format(Zs))
