import numpy as np
import argparse
import scipy.linalg as LA
import h5py

from ase.dft.kpoints import get_special_points, bandpath, special_paths
from pyscf.pbc import gto, dft
from mbanalysis.src import winter
import matplotlib.pyplot as plt


#
# Input data for Wannier interpolation of Fock eigenvalue
#

# Default parameters
parser = argparse.ArgumentParser(
    description="Wannier interpolaotion for Fock to get bands"
)
parser.add_argument(
    "--debug", type=bool, default=False, help="Debug mode (True/False)"
)
parser.add_argument(
    "--celltype", type=str, default="cubic",
    help="Type of lattice: cubic, diamond, etc."
)
parser.add_argument(
    "--bz_type", type=str, default="cubic",
    help="Brillouin zone to get special k-path."
)
parser.add_argument(
    "--bandpath", type=str, default=None,
    help="High symmetry path for the band structure, e.g. 'L G X G'. \
        NOTE: Use spaces."
)
parser.add_argument(
    "--bandpts", type=int, default=50,
    help="Number of k-points used in the band path."
)
parser.add_argument(
    "--input", type=str, default="input.h5",
    help="Input file used in GW calculation."
)
parser.add_argument(
    "--out", type=str, default='fock_bands.h5',
    help="Name for output file (should be .h5 format)."
)
parser.add_argument(
    "--x2c", type=int, default=0,
    help="level of x2c approximation: 0=none, 1=sfx2c1e, 2=x2c1e."
)
args = parser.parse_args()

#
# First perform Wannier interpolation, then Nevanlinna to get the bands
#

# Parameters in the calculation
debug = args.debug
celltype = args.celltype
bz_type = args.bz_type
bandpath_str = args.bandpath
bandpts = args.bandpts
input_path = args.input
output = args.out
x2c = args.x2c


#
# Read Input
#

print("Reading input file")
f = h5py.File(input_path, 'r')
cell = f["Cell"][()]
kmesh_abs = f["/grid/k_mesh"][()]
kmesh_scaled = f["/grid/k_mesh_scaled"][()]
index = f["/grid/index"][()]
ir_list = f["/grid/ir_list"][()]
conj_list = f["/grid/conj_list"][()]
Fk = f["HF/Fock-k"][()].view(complex)
Fk = Fk.reshape(Fk.shape[:-1])
nk = index.shape[0]
ink = ir_list.shape[0]
f.close()

# Pyscf object to generate k points
mycell = gto.loads(cell)

# Crystal structure
a_vecs = np.genfromtxt(mycell.a.replace(',', ' ').splitlines(), dtype=float)
points = get_special_points(a_vecs, lattice=celltype)

if bandpath_str is not None:
    kptlist = bandpath_str.split(' ')
else:
    special_path_str = special_paths[bz_type]
    kptlist = special_paths[bz_type]
path = bandpath(kptlist, a_vecs, npoints=bandpts)
band_kpts = path.kpts
kpath, sp_points, labels = path.get_linear_kpoint_axis()
band_kpts_abs = mycell.get_abs_kpts(band_kpts)


#
# Wannier interpolation
#

# mean-field object
kmf = dft.KUKS(mycell, kmesh_abs)
if x2c == 1:
    kmf = kmf.sfx2c1e()
elif x2c == 2:
    kmf = dft.KGKS(mycell, kmesh_abs).x2c1e()

# interpolate overlap
Sk_int = kmf.get_ovlp(mycell, band_kpts_abs)

# interpolate Fock
Fk_int = winter.interpolate(
    Fk, kmesh_scaled, band_kpts, dim=3, hermi=True, debug=debug
)

# Overlap 0.00041291502948366776
# Fock 0.0003271901185318618

# Solve the generalized eigenvalue problems
ns, nk_int, nao = Fk_int.shape[:3]
evals = np.zeros((ns, nk_int, nao))
for s in range(ns):
    for ik in range(nk_int):
        evals[s, ik] = LA.eigvalsh(Fk_int[s, ik], Sk_int[ik])

# fermi energy
print('Number of electrons: ', mycell.nelectron)
if x2c == 2:
    efermi = np.max(evals[:, :, mycell.nelectron - 1])
else:
    efermi = np.max(evals[:, :, mycell.nelectron // 2 - 1])

f = h5py.File(output, 'w')
it = 0
f["S-k"] = Sk_int
if bandpath_str is not None:
    ls_out = []
    for pt in kptlist:
        ls_out.append(pt.replace('G', r'$\Gamma$'))
    f["kptlist"] = ls_out
else:
    f["kptlist"] = labels
f["kpts_interpolate"] = band_kpts
f["kpath"] = kpath
f["sp_points"] = sp_points
f["e_kn"] = evals
f["efermi"] = efermi
f.close()


#
# Prepare a plot
#

au2ev = 27.211
emin = -10
emax = 15
evals -= efermi
# plt.style.use(['science', 'muted'])
plt.figure(figsize=(5, 6))

for n in range(nao):
    pa, = plt.plot(kpath, au2ev * evals[0, :, n])
    pb, = plt.plot(kpath, au2ev * evals[1, :, n], color=pa.get_color())

# Special points
for p in sp_points:
    plt.plot([p, p], [emin, emax], 'k-')

# x-axis
plt.axhline(0, color='k')

# x-ticks
if bandpath_str is not None:
    plt.xticks(sp_points, ls_out)
else:
    plt.xticks(sp_points, labels)

# save
plt.savefig(
    'fock_bands.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1
)
