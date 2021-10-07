import numpy as np
import scipy.linalg as LA
import h5py

from ase.spacegroup import crystal
from ase.spacegroup import Spacegroup
from MB_analysis import mb

##################
#
# Wannier interpolation example for correlated methods
#
##################


##################
#
# Input parameters
#
##################
# Inverse temperature
T_inv = 1000
debug = True

# Crystal structure
a, b, c = 5.43, 5.43, 5.43
alpha, beta, gamma = 90, 90, 90
group = 227

cc = crystal(symbols=['Si'], 
             basis=[(0.0, 0.0, 0.0)], 
             spacegroup=group, 
             cellpar=[a, b, c, alpha, beta, gamma], primitive_cell=True)

path = cc.cell.bandpath('WGXWLG', npoints=100)
kpts_inter = path.kpts

# Input files
input_path = "/pauli-storage/cnyeh/Si/nk6/LDA/input.h5"
GW_path = "/pauli-storage/cnyeh/Si/nk6/GW_1000_104/sim_last.h5"
lamb = '1e4'

# Output file
output = "test.h5" #"666Si_GW_WGXWLF.h5"

##################
#
# Reading input data
#
##################

f = h5py.File(input_path, 'r')
kmesh_scaled = f["/grid/k_mesh_scaled"][()]
index   = f["grid/index"][()]
ir_list = f["/grid/ir_list"][()]
conj_list = f["grid/conj_list"][()]
reduced_kmesh_scaled = kmesh_scaled[ir_list]
nk = index.shape[0]
ink = ir_list.shape[0]
f.close()

f = h5py.File(GW_path, 'r')
it = f["iter"][()]
rSk = f["/S-k"][()].view(complex)
rFk = f["iter"+str(it)+"/Fock-k"][()].view(complex)
rGk = f["iter"+str(it)+"/G_tau/data"][()].view(complex)
rSigmak = f["iter"+str(it)+"/Selfenergy/data"][()].view(complex)
tau_mesh = f["iter"+str(it)+"/G_tau/mesh"][()]
mu = f["iter"+str(it)+"/mu"][()]
rFk = rFk.reshape(rFk.shape[:-1])
rGk = rGk.reshape(rGk.shape[:-1])
rSigmak = rSigmak.reshape(rSigmak.shape[:-1])
rSk = rSk.reshape(rSk.shape[:-1])
nao = rFk.shape[-1]
nts = rSigmak.shape[0]
f.close()

Fk = mb.to_full_bz(rFk, conj_list, ir_list, index, 0)
Sk = mb.to_full_bz(rSk, conj_list, ir_list, index, 0)
Sigma_tk = mb.to_full_bz(rSigmak, conj_list, ir_list, index, 1)
del rFk, rSk, rSigmak 

##################
#
# Wannier interpolation
#
##################

# MB_post class. (i) Input data e.g. fock, sigma, gtau, S have to be in full BZ.
MB = mb.MB_post(fock=Fk, sigma=Sigma_tk, mu=mu, S=Sk, kmesh=kmesh_scaled, beta=T_inv, lamb=lamb)
# Wannier interpolation for basis defined by MB_post.S. Emperically, AO basis seems to be much more localized than SAO. 
G_tk_int, Sigma_tk_int, tau_mesh, Fk_int, Sk_int = MB.wannier_interpolation(kpts_inter, hermi=True, debug=debug)

f = h5py.File(output,'w')
f["S-k"] = Sk_int
f["kpts_interpolate"] = kpts_inter
f["iter"] = it
f["iter"+str(it)+"/Fock-k"] = Fk_int
f["iter"+str(it)+"/Selfenergy/data"] = Sigma_tk_int
f["iter"+str(it)+"/Selfenergy/mesh"] = tau_mesh
f["iter"+str(it)+"/G_tau/data"] = G_tk_int
f["iter"+str(it)+"/G_tau/mesh"] = tau_mesh
f["iter"+str(it)+"/mu"] = mu
f.close()
