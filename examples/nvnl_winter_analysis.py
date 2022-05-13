import h5py
import time
import argparse
import numpy as np
from ase.dft.kpoints import sc_special_points, get_bandpath
from pyscf.pbc import gto
from mbanalysis import mb
from mbanalysis.src import orth


#
# Input data for Wannier interpolation and analytic continuation
#

# Default parameters
a = """4.0655,    0.0,    0.0
    0.0,    4.0655,    0.0
    0.0,    0.0,    4.0655"""
atoms = """H -0.25 -0.25 -0.25
    H  0.25  0.25  0.25"""
basis = 'sto-3g'

parser = argparse.ArgumentParser(
    description="Wannier interpolaotion and Nevanlinna analytic continuation"
)
parser.add_argument(
    "--beta", type=float, default=100, help="Inverse temperature"
)
parser.add_argument(
    "--debug", type=bool, default=False, help="Debug mode (True/False)"
)
parser.add_argument(
    "--mu", type=float, default=0, help="Chemical potential"
)
parser.add_argument(
    "--a", type=str, default=a, help="Lattice geometry"
)
parser.add_argument(
    "--atoms", type=str, default=atoms, help="Positions of atoms in unit cell"
)
parser.add_argument(
    "--celltype", type=str, default="cubic",
    help="Type of lattice: cubic, diamond, etc."
)
parser.add_argument(
    "--bandpath", type=str, default="GMXGRX",
    help="High symmetry path for the band structure, e.g. 'LGXKG"
)
parser.add_argument(
    "--bandpts", type=int, default=50,
    help="Number of k-points used in the band path"
)
parser.add_argument(
    "--basis", type=str, default=basis, help="Atomic basis set used"
)
parser.add_argument(
    "--input", type=str, default="input.h5",
    help="Input file used in GW calculation"
)
parser.add_argument(
    "--sim", type=str, default="sim.h5",
    help="Output of UGF2 code, i.e., sim.h5"
)
# TODO: Replace lambda functionality to IR-grid file
parser.add_argument(
    "--lamb", type=str, default='1e4',
    help="Lambda used in the IR grid."
)
parser.add_argument(
    "--out", type=str, default='winter_out.h5',
    help="Name for output file (should be .h5 format)"
)
parser.add_argument(
    "--nevan_exe", type=str, default=None,
    help="Path to compiled Nevanlinna program"
)
args = parser.parse_args()

#
# First perform Wannier interpolation, then Nevanlinna to get the bands
#

# Parameters in the calculation
T_inv = args.beta
debug = args.debug
mu = args.mu
a = args.a
atoms = args.atoms
basis = args.basis
celltype = args.celltype
bandpath_str = args.bandpath
bandpts = args.bandpts.replace(' ', '')  # clean up spaces
input_path = args.input
sim_path = args.sim
lamb = args.lamb
output = args.out
nevan_exe = args.nevan_exe
if nevan_exe is None:
    ValueError('nevan_exe cannot be None')

# Pyscf object to generate k points
mycell = gto.M(a=a, atom=atoms, unit='A', basis=basis, verbose=0, spin=0)

# Use ase to generate the kpath
a_vecs = np.genfromtxt(a.replace(',', ' ').splitlines(), dtype=float)
points = sc_special_points[celltype]
kptlist = []
for kchar in bandpath_str:
    kptlist.append(points[kchar])
band_kpts, kpath, sp_points = get_bandpath(kptlist, a_vecs, npoints=bandpts)

# ASE will give scaled band_kpts. We need to transform them to absolute values
# using mycell.get_abs_kpts
band_kpts = mycell.get_abs_kpts(band_kpts)

#
# Read Input
#

print("Reading input file")
f = h5py.File(input_path, 'r')
kmesh_scaled = f["/grid/k_mesh_scaled"][()]
index = f["/grid/index"][()]
ir_list = f["/grid/ir_list"][()]
conj_list = f["/grid/conj_list"][()]
reduced_kmesh_scaled = kmesh_scaled[ir_list]
nk = index.shape[0]
ink = ir_list.shape[0]
f.close()

print("Reading sim file")
f = h5py.File(sim_path, 'r')
it = f["iter"][()]
rSk = f["/S-k"][()].view(complex)
rFk = f["iter" + str(it) + "/Fock-k"][()].view(complex)
rGk = f["iter" + str(it) + "/G_tau/data"][()].view(complex)
rSigmak = f["iter" + str(it) + "/Selfenergy/data"][()].view(complex)
tau_mesh = f["iter" + str(it) + "/G_tau/mesh"][()]
mu = f["iter"+str(it)+"/mu"][()]
rFk = rFk.reshape(rFk.shape[:-1])
rGk = rGk.reshape(rGk.shape[:-1])
rSigmak = rSigmak.reshape(rSigmak.shape[:-1])
rSk = rSk.reshape(rSk.shape[:-1])
nao = rFk.shape[-1]
nts = rSigmak.shape[0]
f.close()

print(rFk.shape)
print(rSk.shape)
print(rSigmak.shape)

print("Transfrom quantities to full BZ")
Fk = mb.to_full_bz(rFk, conj_list, ir_list, index, 1)
Sk = mb.to_full_bz(rSk, conj_list, ir_list, index, 1)
Sigma_tk = mb.to_full_bz(rSigmak, conj_list, ir_list, index, 2)
print(Fk.shape)
print(Sk.shape)
print(Sigma_tk.shape)
del rFk, rSk, rSigmak


#
# Wannier interpolation
#

# Initialize mbanalysis post processing
mbo = mb.MB_post(
    fock=Fk, sigma=Sigma_tk, mu=mu, S=Sk, kmesh=kmesh_scaled,
    beta=T_inv, lamb=lamb
)

# Wannier interpolation
print("Starting interpolation")
t1 = time.time()
G_tk_int, Sigma_tk_int, tau_mesh, Fk_int, Sk_int = mbo.wannier_interpolation(
    band_kpts, hermi=True, debug=debug
)
t2 = time.time()
print("Time required for Wannier interpolation: ", t2 - t1)


#
# Orthogonalization and Nevanlinna
#

print("Transforming interpolated Gtau to SAO basis")
Gt_sao = orth.sao_orth(G_tk_int, Sk_int, type='g')

print("Starting Nevanlinna")
t3 = time.time()
mbo.AC_nevanlinna(
    nevan_exe=nevan_exe, outdir='Nevanlinna'
)
t4 = time.time()
print("TIme required for Nevanlinna AC: ", t4 - t3)

# Save interpolated data to HDF5
# This file contains the Green's function, Fock matrix, etc. at the k-points
# along the selected path
f = h5py.File(output, 'w')
f["S-k"] = Sk_int
f["kpts_interpolate"] = band_kpts
f["iter"] = it
f["iter"+str(it)+"/Fock-k"] = Fk_int
f["iter"+str(it)+"/Selfenergy/data"] = Sigma_tk_int
f["iter"+str(it)+"/Selfenergy/mesh"] = tau_mesh
f["iter"+str(it)+"/G_tau/data"] = G_tk_int
f["iter"+str(it)+"/G_tau/mesh"] = tau_mesh
f["iter"+str(it)+"/mu"] = mu
f.close()
