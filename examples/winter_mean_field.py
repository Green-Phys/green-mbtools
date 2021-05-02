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
debug = False
mu = 0.29

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
lamb = '1e4'

# Output file
bands_output = "bands.npy"
output = "666Si_LDA_WGXWLG.h5" 

##################
#
# Reading input data
#
##################

f = h5py.File(input_path, 'r')
kmesh_scaled = f["/grid/k_mesh_scaled"][()]
nk = f["HF/nk"][()]
Fk = f ["HF/Fock-k"][()].view(np.complex)
Sk = f ["HF/S-k"][()].view(np.complex)  
Fk = Fk.reshape(Fk.shape[:-1])
Sk = Sk.reshape(Sk.shape[:-1])
nao = Fk.shape[-1]
f.close()

##################
#
# Wannier interpolation
#
##################

# MB_post class. ***Input data e.g. fock, sigma, gtau, S have to be in full BZ.*** 
MB = mb.MB_post(fock=Fk, sigma=None, mu=mu, S=Sk, kmesh=kmesh_scaled, beta=T_inv, lamb=lamb)
# Wannier interpolation for basis defined by MB_post.S. Emperically, AO basis seems to be much more localized than SAO. 
G_tk_int, Sigma_tk_int, tau_mesh, Fk_int, Sk_int = MB.wannier_interpolation(kpts_inter, hermi=True, debug=debug)

# Overlap 0.00041291502948366776
# Fock 0.0003271901185318618

# Solve the generalized eigenvalue problems
ns = Fk_int.shape[0]
evals = np.zeros((ns, kpts_inter.shape[0], nao))
for s in range(ns):
  for ik in range(kpts_inter.shape[0]):
    evals[s, ik] = LA.eigvalsh(Fk_int[s, ik], Sk_int[s, ik])
np.save(bands_output, evals)


f = h5py.File(output,'w')
it = 0
f["S-k"] = Sk_int
f["kpts_interpolate"] = kpts_inter
f["iter"] = it
f["iter"+str(it)+"/Fock-k"] = Fk_int
f["iter"+str(it)+"/G_tau/data"] = G_tk_int
f["iter"+str(it)+"/G_tau/mesh"] = tau_mesh
f["iter"+str(it)+"/mu"] = mu
f.close()
