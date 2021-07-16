import numpy as np
import h5py

import MB_analysis
from MB_analysis import mb
from MB_analysis.src import orth

##################
#
# Input parameters
#
##################
# Inverse temperature
T_inv = 1000
debug = True

# Input files
MB_path = MB_analysis.__path__[0] + '/../'
input_path = MB_path + '/data/H2_GW/input.h5'
sim_path = MB_path + '/data/H2_GW/sim.h5'
lamb = '1e4'

maxent_exe = '/home/cnyeh/Project/Maxent/build/maxent'

##################
#
# Read input data
#
##################

f = h5py.File(sim_path, 'r')
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

f = h5py.File(input_path, 'r')
ir_list = f["/grid/ir_list"][()]
weight = f["/grid/weight"][()]
index = f["/grid/index"][()]
conj_list = f["grid/conj_list"][()]
f.close()

''' All k-dependent matrices should lie on a full Monkhorst-Pack grid. '''
# Transform from reduced BZ to full BZ
Fk = mb.to_full_bz(Fr, conj_list, ir_list, index, 1)
Sk = mb.to_full_bz(Sr, conj_list, ir_list, index, 1)
Sigmak = mb.to_full_bz(Sigmar, conj_list, ir_list, index, 2)
Gk = mb.to_full_bz(Gr, conj_list, ir_list, index, 2) 

##################
#
# Maxent analytical continuation
#
##################

# Construct MB_post class
MB = mb.MB_post(fock=Fk, sigma=Sigmak, gtau=Gk, mu=mu, S=Sk, beta=T_inv, lamb=lamb)

# By default, running Maxent for all diagonal elements of MB.gtau in SAO basis
MB.AC_maxent(error=5e-3, maxent_exe=maxent_exe, params=MB_path+'/data/Maxent/green.param', outdir='Maxent')

# Running Maxent for given G(t) in whatever orthogonal basis
Gt_sao = orth.sao_orth(MB.gtau, MB.S, type='g')
Gt_orbsum = np.einsum("tskii->tsk", Gt_sao)
MB.AC_maxent(error=5e-3, maxent_exe=maxent_exe, params=MB_path+'/data/Maxent/green.param', outdir='Maxent_orbsum', gtau_orth=Gt_orbsum)
