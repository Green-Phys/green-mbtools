import h5py
import numpy as np
from green_mbtools.pesto import mb, orth

#
# Example
# Perform orthogonalization of Fock matrices
# explores symmetric and canonical orthogonalization
#

##################
#
# Read input data
#
##################

data_dir = '../tests/test_data'
f = h5py.File(data_dir + '/H2_GW/sim.h5', 'r')
it = f["iter"][()]
Sigma1r = f["iter" + str(it) + "/Sigma1"][()].view(complex)
f.close()

f = h5py.File(data_dir + '/H2_GW/input.h5', 'r')
Sk = f["HF/S-k"][()].view(complex)
Sk = Sk.reshape(Sk.shape[:-1])
Hk = f["HF/H-k"][()].view(complex)
Hk = Hk.reshape(Hk.shape[:-1])
ir_list = f["/grid/ir_list"][()]
weight = f["/grid/weight"][()]
index = f["/grid/index"][()]
conj_list = f["grid/conj_list"][()]
f.close()

''' All k-dependent matrices should lie on a full Monkhorst-Pack grid. '''
# Transform from reduced BZ to full BZ
Sigma1 = mb.to_full_bz(Sigma1r, conj_list, ir_list, index, 1)
Fk = Sigma1 + Hk

# NOTE: type = 'f' denotes that the transoformation is for Fock-type objects
#       type = 'g' will be used for transforming Green's function or
#       density-matrix type objects
F_sao = orth.sao_orth(Fk, Sk, type='f')
e_sao = np.linalg.eigh(F_sao)[0]
F_can = orth.canonical_orth(Fk, Sk, thr=1e-7, type='f')
e_can = np.linalg.eigh(F_can)[0]
print("Orbital energies from SAO basis = ")
print(e_sao)
print("Orbital energies from canonical orthogonoalization = ")
print(e_can)
