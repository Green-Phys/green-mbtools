from functools import reduce
import h5py
import numpy as np

import MB_analysis
from MB_analysis import mb
from MB_analysis.src import orth

##################
#
# Read input data
#
##################

MB_path = MB_analysis.__path__[0] + '/../'
f = h5py.File(MB_path + '/data/H2_GW/sim.h5', 'r')
Sr = f["S-k"][()].view(np.complex)
Sr = Sr.reshape(Sr.shape[:-1])
Fr = f["iter14/Fock-k"][()].view(np.complex)
Fr = Fr.reshape(Fr.shape[:-1])
f.close()

f = h5py.File(MB_path + '/data/H2_GW/input.h5', 'r')
ir_list = f["/grid/ir_list"][()]
weight = f["/grid/weight"][()]
index = f["/grid/index"][()]
conj_list = f["grid/conj_list"][()]
f.close()

''' All k-dependent matrices should lie on a full Monkhorst-Pack grid. '''
# Transform from reduced BZ to full BZ
F = mb.to_full_bz(Fr, conj_list, ir_list, index, 1)
S = mb.to_full_bz(Sr, conj_list, ir_list, index, 1)

F_sao = orth.sao_orth(F, S, type='f')
e_sao = np.linalg.eigh(F_sao)[0]
F_can = orth.canonical_orth(F, S, thr=1e-7, type='f')
e_can = np.linalg.eigh(F_can)[0]
print("Orbital energies from SAO basis = ")
print(e_sao)
print("Orbital energies from canonical orthogonoalization = ")
print(e_can)
