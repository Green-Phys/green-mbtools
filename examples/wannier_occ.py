import numpy as np
import argparse
import h5py

from ase.dft.kpoints import get_special_points, bandpath, special_paths
from pyscf.pbc import gto, dft
from pyscf.tools import molden
from mbanalysis import winter
from mbanalysis.mb import to_full_bz
import matplotlib.pyplot as plt


#
# Input data for Wannier interpolation of density matrix eigenvalues
#
# Example of usage:
# python3 ~/dev/mbanalysis/examples/wannier_occ.py \
#   --bz_type "rhombohedral type 1" \
#   --celltype "rhombohedral" \
#   --inp input.h5 \
#   --sim sim_ao.h5

def drop_nos_molden(fout, mycell, c, occ):
    with open(fout, 'w') as f:
        molden.header(mycell, f)
        molden.orbital_coeff(mycell, f, c, ene=occ)


# Returns density matrix in the notation gamma_{pq} = <q^\dagger p>
def read_sim(finput):
    h_in = h5py.File(finput, 'r')
    # Pull data from the last iteration
    last_it = h_in["iter"][()]
    print("Last iteration in ", finput, " is ", last_it)
    print("Using the Green's function from the iteration ", last_it)
    # Read the Green's function
    G_tau = h_in['iter'+str(last_it) + "/G_tau" + "/data"][()].view(complex)
    G_tau = G_tau.reshape(G_tau.shape[:-1])
    nts = G_tau.shape[0]
    ns = G_tau.shape[1]
    nk = G_tau.shape[2]
    nao = G_tau.shape[3]
    dmr = np.zeros((ns, nk, nao, nao), dtype=np.cdouble)
    dmr[:, :, :, :] = G_tau[nts-1, :, :, :, :]
    dmr *= -1  # due to antiperiodicity of G
    h_in.close()
    return dmr, nao, nk


def power(A, p):
    res = np.zeros((A.shape[0], A.shape[1], A.shape[2]), dtype=np.cdouble)
    for i in range(A.shape[0]):
        tmp = power_k(A[i, :, :],  p)
        res[i, :, :] = tmp[:, :]
    return res


def power_k(A, p):
    e, w = np.linalg.eigh(A)
    e = np.float_power(e, p)
    return w @ np.diag(e) @ np.linalg.inv(w)


# Default parameters
parser = argparse.ArgumentParser(
    description="Wannier interpolation of spin-averaged density matrix \
        to get occupations"
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
    "--sim", type=str, default="sim.h5",
    help="Simulation file to be read"
)
parser.add_argument(
    "--out", type=str, default='occ_bands.h5',
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
fsim = args.sim
output = args.out
x2c = args.x2c

if x2c == 2:
    raise ValueError("GHF-like density matrices are not implemented yet")

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
f.close()

# Pyscf object to generate k points
mycell = gto.loads(cell)

print("Reading simulation file")
r_dmr, nao_sim, ink_sim = read_sim(fsim)

dmr = to_full_bz(r_dmr, conj_list, ir_list, index, 1)

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

# interpolate density matrix
print("Interpolating density matrix")
dmr_int = winter.interpolate(
    dmr, kmesh_scaled, band_kpts, dim=3, hermi=True, debug=debug
)

# Spin-averaged natural orbitals (complex-conjugated) and their occupancies

print("Computing SA-NOs")
ns, nk_int, nao = dmr_int.shape[:3]
# Get S^{1/2}
S_12 = np.zeros((nk_int, nao, nao), dtype=np.cdouble)
S_12 = power(Sk_int, 0.5)
occ_ab = np.zeros((nk_int, nao))
c_ab = np.zeros((nk_int, nao, nao), dtype=np.cdouble)
yamaguchi_int = np.zeros((nk_int))
headgordon_int = np.zeros((nk_int))
for i in range(nk_int):
    sa_dmr = np.zeros((nao, nao), dtype=np.cdouble)
    if ns == 1 or ns == 2:
        for js in range(ns):
            sa_dmr[:, :] += S_12[i, :, :] @ dmr_int[js, i, :, :] \
                @ S_12[i, :, :]
    else:
        raise ValueError(
            "The number of spin variables is larger than I can handle"
        )
    occ_ab[i, :], w_ab = np.linalg.eigh(sa_dmr)
    c_ab[i, :, :] = np.linalg.inv(S_12[i, :, :]) @ w_ab
    c_ab = c_ab.conj()  # chemical-friendly notation

    # Effective number of unpaired electrons
    yamaguchi_int[i] = 0
    headgordon_int[i] = 0
    for o in occ_ab[i, :]:
        cap_o = o
        if o > 2.0:
            cap_o = 2.0  # bound interpolated occupancies
        if o < 0.0:
            cap_o = 0.0  # bound interpolated occupancies
        yamaguchi_int[i] += min(cap_o, 2-cap_o)
        headgordon_int[i] += (cap_o**2) * ((2-cap_o)**2)

print('Number of electrons: ', mycell.nelectron)

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
f["occ_ab"] = occ_ab
f.close()

#
# Prepare a plot for an outside use
# This gives a very verbose output file,
# but it can be adjusted for a specific application

# Occupations
f_occ = open('occupancies_kpath.dat', 'w', encoding="utf-8")
for k in range(len(kpath)):
    str_to_write = str(kpath[k]) + " "
    for n in range(nao):
        str_to_write += (str(occ_ab[k, n]) + " ")
    print(str_to_write, file=f_occ)
f_occ.close()

# Labels to plot occupations with other software
f_sp = open('labels.dat', 'w', encoding="utf-8")
str_to_write = ""
for sp in sp_points:
    str_to_write += str(sp) + " "
print(str_to_write, file=f_sp)

str_to_write = ""
for ll in labels:
    str_to_write += str(ll) + " "
print(str_to_write, file=f_sp)
f_sp.close()

# Yamaguchi and Head-Gordon indices to plot later
f_ind = open('indices.dat', 'w', encoding="utf-8")
for k in range(len(kpath)):
    str_to_write = str(kpath[k]) + " "
    str_to_write += (str(yamaguchi_int[k]) + " ")
    str_to_write += (str(headgordon_int[k]) + " ")
    print(str_to_write, file=f_ind)
f_ind.close()

# AO coefficients
for n in range(nao):
    f_ao_re = open("coefs_" + str(n) + "_re_kpath.dat", 'w', encoding="utf-8")
    f_ao_im = open("coefs_" + str(n) + "_im_kpath.dat", 'w', encoding="utf-8")
    for k in range(len(kpath)):
        str_to_write_re = str(kpath[k]) + " "
        str_to_write_im = str(kpath[k]) + " "
        for ao in range(nao):
            str_to_write_re += (str(np.real(c_ab[k, ao, n])) + " ")
            str_to_write_im += (str(np.imag(c_ab[k, ao, n])) + " ")
        print(str_to_write_re, file=f_ao_re)
        print(str_to_write_im, file=f_ao_im)
    f_ao_re.close()
    f_ao_im.close()

for k in range(len(kpath)):
    fout = "int_NOs_" + str(k) + "_re.molden"
    drop_nos_molden(fout, mycell, np.real(c_ab[k, :, :]), occ_ab[k, :])
    fout = "int_NOs_" + str(k) + "_im.molden"
    drop_nos_molden(fout, mycell, np.imag(c_ab[k, :, :]), occ_ab[k, :])


#
# Prepare a plot through matplotlib
#

au2ev = 27.211
emin = 0.0
emax = 2.0
# plt.style.use(['science', 'muted'])
plt.figure(figsize=(5, 6))

for n in range(nao):
    pop = plt.plot(kpath, occ_ab[:, n])

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
    'occ_bands.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1
)
