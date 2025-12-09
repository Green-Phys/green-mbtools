import numpy as np
import scipy.linalg as LA
import h5py

from ase.spacegroup import crystal
from green_mbtools.pesto import mb


#
# Example
# Perform wannier interpolation for mean-field methods
#
# System: cubic crystal of H2 molecules

# NOTE: This example also interpolates the overlap matrix,
#       which is not the ideal approach. Instead, one should simply
#       use PySCF to get the overlap matrix on the high-symmetry path.
#

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
a, b, c = 4.0655, 4.0655, 4.0655
alpha, beta, gamma = 90, 90, 90
group = 221

cc = crystal(
    symbols=["H", "H"],
    basis=[(-0.25, -0.25, -0.25), (0.25, 0.25, 0.25)],
    spacegroup=group,
    cellpar=[a, b, c, alpha, beta, gamma],
    primitive_cell=True,
)

path = cc.cell.bandpath("GXMGR", npoints=100)
kpts_inter = path.kpts

# Input files
data_dir = "../tests/test_data"
input_path = data_dir + "/H2_GW/input.h5"
ir_file = data_dir + "/ir_grid/1e4.h5"

# Output file
bands_output = "bands.npy"
output = "H2_LDA_GXMGR.h5"

##################
#
# Reading input data
#
##################

f = h5py.File(input_path, "r")
kmesh_scaled = f["/grid/k_mesh_scaled"][()]
nk = f["HF/nk"][()]
Fk = f["HF/Fock-k"][()].view(complex)
Sk = f["HF/S-k"][()].view(complex)
Fk = Fk.reshape(Fk.shape[:-1])
Sk = Sk.reshape(Sk.shape[:-1])
nao = Fk.shape[-1]
f.close()

##################
#
# Wannier interpolation
#
##################

# MB_post class.
# Input data e.g. fock, sigma, gtau, S have to be in full BZ.
MB = mb.MB_post(
    fock=Fk, sigma=None, mu=mu, S=Sk, kmesh=kmesh_scaled, beta=T_inv, ir_file=ir_file
)
# Wannier interpolation for basis defined by MB_post.S.
# Emperically, AO basis seems to be much more localized than SAO.
G_tk_int, Sigma_tk_int, tau_mesh, Fk_int, Sk_int = MB.wannier_interpolation(
    kpts_inter, hermi=True, debug=debug
)

# Solve the generalized eigenvalue problems
ns = Fk_int.shape[0]
evals = np.zeros((ns, kpts_inter.shape[0], nao))
for s in range(ns):
    for ik in range(kpts_inter.shape[0]):
        evals[s, ik] = LA.eigvalsh(Fk_int[s, ik], Sk_int[s, ik])
np.save(bands_output, evals)


f = h5py.File(output, "w")
it = 0
f["S-k"] = Sk_int
f["kpts_interpolate"] = kpts_inter
f["iter"] = it
f["iter" + str(it) + "/Fock-k"] = Fk_int
f["iter" + str(it) + "/G_tau/data"] = G_tk_int
f["iter" + str(it) + "/G_tau/mesh"] = tau_mesh
f["iter" + str(it) + "/mu"] = mu
f.close()
