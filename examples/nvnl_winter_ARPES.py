#
# Python script used to generate ARPES 2D intersection planes of
# Density of states (DoS) in the momentum space.
#


import h5py
import time
import argparse
import numpy as np
from ase.dft.kpoints import sc_special_points, get_bandpath
from pyscf.pbc import gto
import pyscf.pbc.dft as pbcdft
from mbanalysis import mb
from mbanalysis import orth


#
# Input data for Wannier interpolation and analytic continuation
#

# Default parameters
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
    "--wannier", type=int, default=1, help="Toggle Wannier interpolation"
)
parser.add_argument(
    "--celltype", type=str, default="cubic",
    help="Type of lattice: cubic, diamond, etc."
)
parser.add_argument(
    "--bandpath", type=str, default="YGX",
    help="High symmetry path for the ARPES intersection analysis. \
        Limited to three points to define a plane."
)
parser.add_argument(
    "--bandpts", type=int, default=50,
    help="Minimal number of points to be interpolated in the \
        intersection plane."
)
parser.add_argument(
    "--input", type=str, default="input.h5",
    help="Input file used in GW calculation from green-mbpt code, \
        i.e., input.h5"
)
parser.add_argument(
    "--sim", type=str, default="sim.h5",
    help="Output of GW calculation from green-mbpt code. i.e., sim.h5"
)
parser.add_argument(
    "--iter", type=int, default=-1,
    help="Iteration number of the scGW cycle to use for continuation."
)
parser.add_argument(
    "--ir_file", type=str, default=None,
    help="HDF5 file that contains information about the IR grid."
)
parser.add_argument(
    "--out", type=str, default='winter_out.h5',
    help="Name for output file (should be .h5 format)"
)
parser.add_argument(
    "--nev_outdir", type=str, default="./Nevanlinna/",
    help="Directory to dump output from Nevanlinna analytic continuation"
)
parser.add_argument(
    "--orth", type=str, default='sao',
    help="Choice of orthonormal basis (sao or co)."
)
parser.add_argument(
    "--mol", type=int, default=0,
    help="Use non-zero value when performing analyt cont for molecules."
)
parser.add_argument(
    "--nw_real", type=int, default=1001,
    help="Number of real frequency points."
)
parser.add_argument(
    "--e_max", type=float, default=2.0,
    help="Maximum energy limit for extrapolation in eV."
)
parser.add_argument(
    "--e_min", type=int, default=-2.0,
    help="Minimal energy limit for extrapolation in eV."
)
parser.add_argument(
    "--eta", type=int, default=0.005,
    help="Broadening factor."
)
args = parser.parse_args()

#
# Load parameters
#

T_inv = args.beta
debug = args.debug
mu = args.mu
wannier = args.wannier
celltype = args.celltype
bandpath_str = args.bandpath.replace(' ', '')  # clean up spaces
bandpts = args.bandpts
input_path = args.input
sim_path = args.sim
it = args.iter
ir_file = args.ir_file
output = args.out
nev_outdir = args.nev_outdir
orth_ao = args.orth
molecule = args.mol
hardy = args.hardy


def mesh_bandpath(cell, kptlist, a_vecs, num_total):
    '''
    TODO: Add description
    '''

    band_kpts, kpath, sp_points = get_bandpath(kptlist, a_vecs, npoints=100)
    # The npoints used here is a mere placeholder,
    # for generating x and y vectors.

    # find kpath indices
    sp_points_idx = []
    for p in sp_points:
        idx = list(kpath).index(p)
        sp_points_idx.append(idx)

    print(sp_points_idx)
    for idx in sp_points_idx:
        print(band_kpts[idx])

    # find the fourth point to make a parallelogram
    y_dir = band_kpts[sp_points_idx[0]]
    origin = band_kpts[sp_points_idx[1]]
    x_dir = band_kpts[sp_points_idx[2]]

    # find the absolute coords.
    mycell = gto.loads(cell)
    y_abs = mycell.get_abs_kpts(y_dir)
    org_abs = mycell.get_abs_kpts(origin)
    x_abs = mycell.get_abs_kpts(x_dir)

    vec_1 = y_dir - origin
    vec_2 = x_dir - origin
    # pt_4 = vec_1 + vec_2 + origin

    vec_1_abs = y_abs - org_abs
    vec_2_abs = x_abs - org_abs

    # find aspect ratio and approximate pixel size
    # length is along the y axis
    # width  is along the x axis
    # not necessary suggesting length is longer than width,
    # or they are perpendicular
    # just for the sake of identifying vectors.
    length = np.linalg.norm(vec_1_abs)
    width = np.linalg.norm(vec_2_abs)
    area = length * width / num_total

    # try to make the pixel shape as close as possible to a square.
    approx_pixel_size = np.sqrt(area)
    num_pix_len = int(length//approx_pixel_size) + 1
    num_pix_wid = int(width//approx_pixel_size) + 1

    # actual size of the pixel.
    real_pixel_size_len = np.linalg.norm(vec_1)/num_pix_len
    real_pixel_size_wid = np.linalg.norm(vec_2)/num_pix_wid

    # generate the mesh
    unit_x = vec_2 / np.linalg.norm(vec_2)
    unit_y = vec_1 / np.linalg.norm(vec_1)
    mesh = np.zeros((num_pix_len, num_pix_wid, 3))
    for i in range(num_pix_len):
        for j in range(num_pix_wid):
            mesh[i, j, :] += (
                origin +
                i * real_pixel_size_len * unit_y +
                j * real_pixel_size_wid * unit_x
            )

    # flatten the mesh into a list of points
    # the index in mesh_list is idx_len * num_pix_wid + idx_wid
    mesh_list = mesh.reshape((num_pix_len * num_pix_wid, 3))
    # band_kpts_abs = kcell.get_abs_kpts(mesh_list)
    # return the angle or vectors as well.

    print("===============================================================")
    print(
        "  In order to fit at least %4d points in the 2D mesh... "
        % (num_total)
    )
    print(
        "  A %4d * %4d mesh is generated for the plane defined by: "
        % (num_pix_len, num_pix_wid)
    )
    print(kptlist)

    return mesh_list, kpath, sp_points, (num_pix_len, num_pix_wid)


#
# Read Input file
#

print("Reading input file")
f = h5py.File(input_path, 'r')
cell = f["Cell"][()]
kmesh_abs = f["/grid/k_mesh"][()]
kmesh_scaled = f["/grid/k_mesh_scaled"][()]
index = f["/grid/index"][()]
ir_list = f["/grid/ir_list"][()]
conj_list = f["/grid/conj_list"][()]
reduced_kmesh_scaled = kmesh_scaled[ir_list]
rSk = f["/HF/S-k"][()]  # Borrow from input.
rHk = f["/HF/H-k"][()]  # Core Hamiltonian.
nk = index.shape[0]
ink = ir_list.shape[0]
f.close()

# Pyscf object to generate k points
if not molecule:
    mycell = gto.loads(cell)
    # Use ase to generate the kpath
    if wannier:
        a_vecs = np.genfromtxt(
            mycell.a.replace(',', ' ').splitlines(), dtype=float
        )
        points = sc_special_points[celltype]
        kptlist = []
        for kchar in bandpath_str:
            kptlist.append(points[kchar])
        band_kpts, kpath, sp_points, mesh_grid = mesh_bandpath(
            cell, kptlist, a_vecs, bandpts
        )
        print(mesh_grid)

print("Reading sim file")
f = h5py.File(sim_path, 'r')
if it == -1:
    it = f["iter"][()]
rSk = f["/S-k"][()]
rFk = f["iter" + str(it) + "/Sigma1"][()]
# Fk should be recalculated from H0 (core) and G.
# Just the static J + K for now (Sigma1). H will be added later.
rGk = f["iter" + str(it) + "/G_tau/data"][()]
rSigmak = f["iter" + str(it) + "/Selfenergy/data"][()]
tau_mesh = f["iter" + str(it) + "/G_tau/mesh"][()]
mu = f["iter"+str(it)+"/mu"][()]
# rFk = rFk.reshape(rFk.shape[:-1])
# rGk = rGk.reshape(rGk.shape[:-1])
# rSigmak = rSigmak.reshape(rSigmak.shape[:-1])
# rSk = rSk.reshape(rSk.shape[:-1])
nao = rFk.shape[-1]
nts = rSigmak.shape[0]
f.close()

if debug:
    print("The dimensions of F, S, and Sigma read from input and sim files.")
    print(rFk.shape)
    print(rSk.shape)
    print(rSigmak.shape)

print("===============================================================")
print("  Transfrom quantities to full BZ...  ")
Fk = mb.to_full_bz(rFk, conj_list, ir_list, index, 1)
# Add core Hamiltonian to the static Fock after transfromation to the FBZ.
Fk += rHk
Sk = mb.to_full_bz(rSk, conj_list, ir_list, index, 1)
G_tk = mb.to_full_bz(rGk, conj_list, ir_list, index, 2)
Sigma_tk = mb.to_full_bz(rSigmak, conj_list, ir_list, index, 2)

if debug:
    print("The dimensions of F, S, and Sigma transformed to the FBZ.")
    print(Fk.shape)
    print(Sk.shape)
    print(Sigma_tk.shape)

del rFk, rSk, rSigmak


#
# Wannier interpolation
#

# Initialize mbanalysis post processing
mbo = mb.MB_post(
    fock=Fk, gtau=G_tk, sigma=Sigma_tk, mu=mu, S=Sk, kmesh=kmesh_scaled,
    beta=T_inv, ir_file=ir_file
)

num_len = mesh_grid[0]
num_wid = mesh_grid[1]
n_real = 1001
A_w_total = np.zeros((n_real, 2, num_len*num_wid, nao), dtype=G_tk.dtype)

# loop over row by row.
print("===============================================================")
print("  Starting interpolation...")
if orth_ao == 'sao':
    print("  Transforming interpolated Gtau to SAO basis...")
elif orth_ao == 'co':
    print("  Transforming interpolated Gtau to Canonical basis...")

for i in range(num_len):
    band_kpts_seg = band_kpts[i*num_wid:(i+1)*num_wid, :]
    band_kpts_abs = mycell.get_abs_kpts(band_kpts_seg)
    print(band_kpts_abs)
    # Wannier interpolation
    if wannier:
        t1 = time.time()
        G_tk_int, Sigma_tk_int, tau_mesh, Fk_int, Sk_int = \
            mbo.wannier_interpolation(
                band_kpts_seg, hermi=True, debug=debug
            )
        nk_cbrt = int(np.cbrt(nk))
        kmesh = mycell.make_kpts([nk_cbrt, nk_cbrt, nk_cbrt])
        mf = pbcdft.KUKS(mycell, kmesh)
        Sk_int_tmp = mf.get_ovlp(mycell, band_kpts_abs)
        Sk_int = np.array([Sk_int_tmp, Sk_int_tmp])
        if debug:
            print("  The shape of Sk_int is ", Sk_int.shape)
        t2 = time.time()
        print("Time consumed for Wannier interpolation: ", t2 - t1)

    else:
        G_tk_int = mbo.gtau
        Sigma_tk_int = mbo.sigma
        Fk_int = mbo.fock
        Sk_int = mbo.S

    #
    # Orthogonalization and Nevanlinna
    #

    if orth_ao == 'sao':
        Gt_ortho = orth.sao_orth(G_tk_int, Sk_int, type='g')
    elif orth_ao == 'co':
        Gt_ortho = orth.canonical_orth(G_tk_int, Sk_int, type='g')

    Gt_ortho_diag = np.einsum('tskii -> tski', Gt_ortho)

    # NOTE: The user can now control parameters that go into analytic cont.
    #       such as no. of real freqs (n_real), w_min, w_max, and eta.
    print("Starting Nevanlinna...")
    t3 = time.time()
    freqs, A_w = mbo.AC_nevanlinna(
        outdir=nev_outdir, gtau_orth=Gt_ortho_diag,
        n_real=n_real, w_min=-0.7, w_max=0.3, eta=0.005
    )
    t4 = time.time()
    print("Time required for Nevanlinna AC: ", t4 - t3)

    print("Writing to ", str(i*num_wid), " to ", str((i+1)*num_wid))
    A_w_total[:, :, i*num_wid:(i+1)*num_wid, :] += A_w


#
# Save interpolated data to h5 file.
#

f = h5py.File(output, 'w')
if wannier:
    f["kpts_interpolate"] = band_kpts
    f["kpath"] = kpath
    f["sp_points"] = sp_points
f["mu"] = mu
f["nevanlinna/freqs"] = freqs
f["nevanlinna/dos"] = A_w_total
f["nevanlinna/mesh_grid"] = mesh_grid
f.close()
