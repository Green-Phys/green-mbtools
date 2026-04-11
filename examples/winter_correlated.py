import h5py
from ase.spacegroup import crystal
from green_mbtools.pesto import mb


#
# Example
# Perform wannier interpolation for correlated methods
# 
# System: cubic crystal of H2 molecules

#
# NOTE: This example also interpolates the overlap matrix,
#       which is not the ideal approach. Instead, one should simply
#       use PySCF to get the overlap matrix on the high-symmetry path.
#

#
# Input parameters
#

# Inverse temperature
T_inv = 1000
debug = True

# Crystal structure
a, b, c = 4.0655, 4.0655, 4.0655
alpha, beta, gamma = 90, 90, 90
group = 221

cc = crystal(
    symbols=['H', 'H'],
    basis=[(-0.25, -0.25, -0.25), (0.25, 0.25, 0.25)],
    spacegroup=group,
    cellpar=[a, b, c, alpha, beta, gamma], primitive_cell=True
)

path = cc.cell.bandpath('GXMGR', npoints=100)
kpts_inter = path.kpts

# Input files
data_dir = '../tests/test_data'
input_path = data_dir + '/H2_GW/input.h5'
GW_path = data_dir + '/H2_GW/sim.h5'
ir_file = data_dir + '/ir_grid/1e4.h5'

# Output file
output = "test.h5"  # "666Si_GW_WGXWLF.h5"

##################
#
# Reading input data
#
##################

f = h5py.File(input_path, 'r')
Sk = f["HF/S-k"][()].view(complex)
Sk = Sk.reshape(Sk.shape[:-1])
Hk = f["HF/H-k"][()].view(complex)
Hk = Hk.reshape(Hk.shape[:-1])
kmesh_scaled = f["/symmetry/k/mesh_scaled"][()]
ibz2bz = f["/symmetry/k/ibz2bz"][()]
bz2ibz = f["/symmetry/k/bz2ibz"][()]
weight_ibz = f["/symmetry/k/weight_ibz"][()]
tr_conj = f["/symmetry/k/tr_conj"][()]
k_sym_trans = f["/symmetry/k/k_sym_transform_ao"][()]
reduced_kmesh_scaled = kmesh_scaled[ibz2bz]
nk = bz2ibz.shape[0]
ink = ibz2bz.shape[0]
f.close()

f = h5py.File(GW_path, 'r')
it = f["iter"][()]
rSigma1 = f["iter" + str(it) + "/Sigma1"][()].view(complex)
rGk = f["iter" + str(it) + "/G_tau/data"][()].view(complex)
rSigmak = f["iter" + str(it) + "/Selfenergy/data"][()].view(complex)
tau_mesh = f["iter" + str(it) + "/G_tau/mesh"][()]
mu = f["iter" + str(it) + "/mu"][()]
nao = rSigma1.shape[-1]
nts = rSigmak.shape[0]
f.close()

Sigma1 = mb.to_full_bz(rSigma1, tr_conj, ibz2bz, bz2ibz, 1, k_sym_trans)
Sigma_tk = mb.to_full_bz(rSigmak, tr_conj, ibz2bz, bz2ibz, 2, k_sym_trans)
del rSigma1, rSigmak

Fk = Hk + Sigma1

##################
#
# Wannier interpolation
#
##################

# MB_post class.
# (i) Input data e.g. fock, sigma, gtau, S have to be in full BZ.
MB = mb.MB_post(
    fock=Fk, sigma=Sigma_tk, mu=mu, S=Sk, kmesh=kmesh_scaled,
    beta=T_inv, ir_file=ir_file
)
# Wannier interpolation for basis defined by MB_post.S.
# Emperically, AO basis seems to be much more localized than SAO.
G_tk_int, Sigma_tk_int, tau_mesh, Fk_int, Sk_int = MB.wannier_interpolation(
    kpts_inter, hermi=True, debug=debug
)

f = h5py.File(output, 'w')
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
