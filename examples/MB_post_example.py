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
Sigmar = f["iter14/Selfenergy/data"][()].view(np.complex)
Sigmar = Sigmar.reshape(Sigmar.shape[:-1])
Gr = f["iter14/G_tau/data"][()].view(np.complex)
Gr = Gr.reshape(Gr.shape[:-1])
mu = f["iter14/mu"][()]
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
Sigma = mb.to_full_bz(Sigmar, conj_list, ir_list, index, 2)
G = mb.to_full_bz(Gr, conj_list, ir_list, index, 2)

##################
#
# Initialize MB_post with non-orthogonal AO basis
#
##################

MB = mb.MB_post(fock=F, sigma=Sigma, mu=mu, gtau=G, S=S, beta=1000, lamb='1e4')
G = MB.gtau
# If G(t) is not known, Dyson euqation can be solved on given beta and ir grid.
MB = mb.MB_post(fock=F, sigma=Sigma, mu=mu, S=S, beta=1000, lamb='1e4')
G2 = MB.gtau

diff = G - G2
print("Maximum G differences = ", np.max(np.abs(diff)))

''' Mulliken analysis '''
print("Mullinken analysis: ")
occs = MB.mulliken_analysis()
print("Spin up:", occs[0], ", Spin down:", occs[1])
print("References: [0.5 0.5] and [0.5 0.5]")

''' Natural orbitals '''
print("Natural orbitals: ")
occ, no_coeff = MB.get_no()
print(occ[0,0])
print(occ[1,0])

''' Molecular energies from different orthogonalization '''
print("Molecular energies: ")
# Standard way to solve FC = SCE by calling scipy.linalg.eigh()
mo_sao, c_sao = MB.get_mo()
print(mo_sao[0,0])
print("Molecular energies from cacnonical orthogonalization: ")
#  Lowdin canonical orthogonalization with threshold = 1e-7
mo_can, c_can = MB.get_mo(canonical=True, thr=1e-7)
print(mo_can[0,0])
# Note that c_sao and c_can will differ by a phase factor!


##################
#
# Initialize MB_post with SAO basis
#
##################

''' Orthogonalized objects from AO to SAO basis. Use type='f' for Hamiltonian and type='g' from Green's function and density matrix. '''
F_orth = orth.sao_orth(F, S, type='f')
Sigma_orth = orth.sao_orth(Sigma, S, type='f')
G_orth = orth.sao_orth(G, S, type='g')
MB = mb.MB_post(fock=F_orth, sigma=Sigma_orth, mu=mu, beta=1000, lamb='1e4')

print("Mullinken analysis: ")
occs = MB.mulliken_analysis()
print("Spin up:", occs[0], ", Spin down:", occs[1])
print("References: [0.5 0.5] and [0.5 0.5]")

print("Natural orbitals: ")
occ, no_coeff = MB.get_no()
print(occ[0, 0])
print(occ[1, 0])
