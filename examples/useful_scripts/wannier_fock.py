import numpy as np
import argparse
import scipy.linalg as LA
import h5py

from ase.dft.kpoints import get_special_points, bandpath, special_paths
from pyscf.pbc import gto, dft
from green_mbtools.pesto import winter
import matplotlib.pyplot as plt

#
# Example
# Calculate mean-field band structure by performing wannier interpolation
# of the Fock matrix
# This is more of a script, rather than an example, that can be
# used directly for applications
#

#
# Input data for Wannier interpolation of Fock eigenvalue
#

# Default parameters
parser = argparse.ArgumentParser(
    description="Wannier interpolaotion for Fock to get bands"
)
parser.add_argument(
    "--debug",
    type=bool,
    default=False, help="Debug mode (True/False)"
)
parser.add_argument(
    "--celltype",
    type=str,
    default="cubic",
    help="Type of lattice: cubic, diamond, etc.",
)
parser.add_argument(
    "--bz_type",
    type=str,
    default="cubic",
    help="Brillouin zone to get special k-path."
)
parser.add_argument(
    "--bandpath",
    type=str,
    nargs="*",
    default=None,
    help="High symmetry path for the band structure, e.g. 'L G X G'. NOTE: Use spaces.",
)
parser.add_argument(
    "--bandpts",
    type=int,
    default=50,
    help="Number of k-points used in the band path."
)
parser.add_argument(
    "--input",
    type=str,
    default="input.h5",
    help="Input file used in GW calculation."
)
parser.add_argument(
    "--out",
    type=str,
    default="fock_bands.h5",
    help="Name for output file (should be .h5 format).",
)
parser.add_argument(
    "--x2c",
    type=int,
    default=0,
    help="level of x2c approximation: 0=none, 1=sfx2c1e, 2=x2c1e.",
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
f = h5py.File(input_path, "r")
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
a_vecs = np.genfromtxt(mycell.a.replace(",", " ").splitlines(), dtype=float)
points = get_special_points(a_vecs, lattice=celltype)

if bandpath_str is not None:
    kptlist = bandpath_str
else:
    # special_path_str = special_paths[bz_type]
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
print("Sk_int shape:", Sk_int.shape)
nk, nao, nao = Sk_int.shape
error = 0.0
for ik in range(nk):
    obj = Sk_int[ik]
    obj_sym = 0.5 * (obj + obj.conj().T)
    error = max(error, np.max(np.abs(obj_sym - obj)))
    Sk_int[ik] = obj_sym
print("The largest Hermitization error = ", error)
#Sk_int = 0.5 * (Sk_int + np.conjugate(np.transpose(Sk_int, (0, 2, 1))))

# interpolate Fock
Fk_int = winter.interpolate(Fk, kmesh_scaled, band_kpts, dim=3, hermi=True, debug=debug)

# Overlap 0.00041291502948366776
# Fock 0.0003271901185318618

# Solve the generalized eigenvalue problems
ns, nk_int, nao = Fk_int.shape[:3]
evals = np.zeros((ns, nk_int, nao))
for s in range(ns):
    for ik in range(nk_int):
       evals[s, ik] = LA.eigvalsh(Fk_int[s, ik], Sk_int[ik])



# Now do it more carefully to avoid numerical issues
#ns, nk_int, nao = Fk_int.shape[:3]
#evals = np.zeros((ns, nk_int, nao))
#eigvecs_ao = np.zeros((ns, nk_int, nao, nao), dtype=complex)
#
#ovlp_thresh = 1e-7  # tweak if needed
#
#for ik in range(nk_int):
#    # Diagonalize overlap at each k
#    s_eig, U = LA.eigh(Sk_int[ik])  # s_eig: (nao,), U: (nao, nao)
#
#    # Keep only well-conditioned subspace
#    mask = s_eig > ovlp_thresh
#    s_inv_sqrt = s_eig[mask] ** -0.5
#    X = U[:, mask] * s_inv_sqrt  # (nao, n_eff), Löwdin orthogonalizer
#
#    for s in range(ns):
#        # Transform F(k) to orthonormal basis: F' = X^† F X
#        F_ortho = X.conj().T @ Fk_int[s, ik] @ X
#        w, C = LA.eigh(F_ortho)  # now a standard Hermitian eigenproblem
#
#        # Store eigenvalues back. If some funcs were dropped, pad with NaNs
#        n_eff = w.shape[0]
#        evals[s, ik, :n_eff] = w
#        if n_eff < nao:
#            evals[s, ik, n_eff:] = np.nan
    
# fermi energy
print("Number of electrons: ", mycell.nelectron)
if x2c == 2:
    efermi = np.max(evals[:, :, mycell.nelectron - 1])
else:
    efermi = np.max(evals[:, :, mycell.nelectron // 2 - 1])
print("Fermi energy (a.u.): ", efermi)
f = h5py.File(output, "w")
it = 0
f["S-k"] = Sk_int
if bandpath_str is not None:
    ls_out = []
    for pt in kptlist:
        ls_out.append(pt.replace("G", r"$\Gamma$"))
    f["kptlist"] = ls_out
else:
    f["kptlist"] = labels
f["kpts_interpolate"] = band_kpts
f["kpath"] = kpath
f["sp_points"] = sp_points
f["e_kn"] = evals
f["efermi"] = efermi
f.close()

plt.figure(figsize=(5, 10), dpi=200)
ls = 0
ao_list = list(range(nao))
for p in sp_points:
    idx = np.where(kpath == p)[0][0]
    print("Plotting point at k =", p)
    plt.plot(ao_list, evals[0, idx, :], label=ls_out[ls])
    ls+=1

plt.legend()
plt.savefig("eigenval.pdf", format="pdf", bbox_inches="tight")
plt.close()
#
# Prepare a plot
#
Sm_f = list(range(55,62))
Sm_d = list(range(35,40))
S_p = list(range(92,95))
if args.x2c == 2:
    Sm_f = Sm_f + [x + 103  for x in Sm_f]
    Sm_d = Sm_d + [x + 103  for x in Sm_d]
    S_p = S_p + [x + 103  for x in S_p]


au2ev = 27.211396641308
emin = -5
emax = 5
evals -= efermi

# plt.style.use(['science', 'muted'])
plt.figure(figsize=(5, 10), dpi=200)


#for n in range(nao):
#    (pa,) = plt.plot(kpath, au2ev * evals[0, :, n])
#    if ns > 1:
#        (pb,) = plt.plot(kpath, au2ev * evals[1, :, n], color=pa.get_color())
#colors = np.zeros((nao),dtype=str)
#zorders = np.zeros((nao),dtype=int)
for n in range(nao):
    #if n in Sm_f:
    #    colors[n] = 'r'
    #    zorders[n] = 4
    #elif n in Sm_d:
    #    colors[n] = 'b'
    #    zorders[n] = 3
    #elif n in S_p:
    #    colors[n] = 'g'
    #    zorders[n] = 4
    #else:
    #    colors[n] = 'k'
    #    zorders[n] = 1
    plt.scatter(
        kpath,
        au2ev * evals[0, :, n],
        s=3,
        alpha=0.7,
        marker="o",
        #color=colors[n],
        #zorder=zorders[n]
    )

# spin 1 (if present)
if ns > 1:
    for n in range(nao):
        plt.scatter(
            kpath,
            au2ev * evals[1, :, n],
            s=3,
            alpha=0.7,
            marker="o",
            #color=colors[n],
            #zorder=zorders[n]
        )
# Special points
for p in sp_points:
    plt.plot([p, p], [emin, emax], "k-")

# x-axis
plt.axhline(0, color="k")

# x-ticks
if bandpath_str is not None:
    plt.xticks(sp_points, ls_out)
else:
    plt.xticks(sp_points, labels)

# x-limit
plt.xlim([kpath[0], kpath[-1]])
plt.ylim([emin, emax])

# save
plt.savefig(
    output.replace(".h5", ".pdf"), format="pdf", bbox_inches="tight", pad_inches=0.1
)
# plt.show()
