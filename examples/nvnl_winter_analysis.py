import h5py
import time
import argparse
import numpy as np
from ase.dft.kpoints import sc_special_points, get_bandpath
from pyscf.pbc import gto, dft
from green_mbtools.pesto import mb, orth, winter, dyson


#
# Example
# Perform wannier interpolation + Nevanlinna analytic continuation
# of the correlated Green's function, to obtaiin correlated
# band structure
# This is more of a script, rather than an example, that can be
# used directly for applications
#
# for usage, run
# python nvnl_winter_analysis.py --help
#

if __name__ == "__main__":
    #
    # Input data for Wannier interpolation and analytic continuation
    #

    # Default parameters
    parser = argparse.ArgumentParser(
        description="Wannier interpolaotion and Nevanlinna continuation"
    )
    parser.add_argument(
        "--beta", type=float, default=100, help="Inverse temperature"
    )
    parser.add_argument(
        "--debug", type=bool, default=False, help="Debug mode (True/False)"
    )
    parser.add_argument(
        "--wannier", type=int, default=1, help="Toggle Wannier interpolation"
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
        "--input", type=str, default="input.h5",
        help="Input file used in GW calculation"
    )
    parser.add_argument(
        "--sim", type=str, default="sim.h5",
        help="Output of UGF2 code, i.e., sim.h5"
    )
    parser.add_argument(
        "--iter", type=int, default=-1,
        help="Iteration number of the scGW cycle to use for continuation"
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
        "--orth", type=str, default='sao',
        help="Choice of orthonormal basis (sao or co)"
    )
    args = parser.parse_args()

    #
    # First perform Wannier interpolation, then Nevanlinna to get the bands
    #

    # Parameters in the calculation
    T_inv = args.beta
    debug = args.debug
    wannier = args.wannier
    celltype = args.celltype
    bandpath_str = args.bandpath.replace(' ', '')  # clean up spaces
    bandpts = args.bandpts
    input_path = args.input
    sim_path = args.sim
    it = args.iter
    ir_file = args.ir_file
    output = args.out
    orth_ao = args.orth

    #
    # Read Input
    #

    print("Reading input file")
    f = h5py.File(input_path, 'r')
    cell = f["Cell"][()]
    Hk = f['HF/H-k'][()].view(complex)
    Sk = f['HF/S-k'][()].view(complex)
    Hk = Hk.reshape(Hk.shape[:-1])
    Sk = Sk.reshape(Sk.shape[:-1])
    kmesh_abs = f["/grid/k_mesh"][()]
    kmesh_scaled = f["/grid/k_mesh_scaled"][()]
    index = f["/grid/index"][()]
    ir_list = f["/grid/ir_list"][()]
    conj_list = f["/grid/conj_list"][()]
    reduced_kmesh_scaled = kmesh_scaled[ir_list]
    nk = index.shape[0]
    ink = ir_list.shape[0]
    f.close()

    # Pyscf object to generate k points
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
        band_kpts, kpath, sp_points = get_bandpath(
            kptlist, a_vecs, npoints=bandpts
        )

        # ASE will give scaled band_kpts. We need to transform them to
        # absolute values using mycell.get_abs_kpts
        band_kpts_abs = mycell.get_abs_kpts(band_kpts)

    print("Reading sim file")
    f = h5py.File(sim_path, 'r')
    if it == -1:
        it = f["iter"][()]
    rFk = f["iter" + str(it) + "/Sigma1"][()]
    rGk = f["iter" + str(it) + "/G_tau/data"][()]
    rSigmak = f["iter" + str(it) + "/Selfenergy/data"][()]
    tau_mesh = f["iter" + str(it) + "/G_tau/mesh"][()]
    mu = f["iter"+str(it)+"/mu"][()]
    nao = rFk.shape[-1]
    nts = rSigmak.shape[0]
    f.close()

    print(rFk.shape)
    print(rSigmak.shape)

    print("Transfrom quantities to full BZ")
    Fk = mb.to_full_bz(rFk, conj_list, ir_list, index, 1)
    Fk += Hk  # add Hcore contribution fron imput file
    G_tk = mb.to_full_bz(rGk, conj_list, ir_list, index, 2)
    Sigma_tk = mb.to_full_bz(rSigmak, conj_list, ir_list, index, 2)
    print("Fock shape: ", Fk.shape)
    print("Overlap shape: ", Sk.shape)
    print("Self-energy shape: ", Sigma_tk.shape)
    del rFk, rSigmak

    #
    # Wannier interpolation
    #

    # Initialize mbanalysis post processing
    mbo = mb.MB_post(
        fock=Fk, gtau=G_tk, sigma=Sigma_tk, mu=mu, S=Sk, kmesh=kmesh_scaled,
        beta=T_inv, ir_file=ir_file
    )

    # Wannier interpolation
    if wannier:
        print("Starting interpolation")
        t1 = time.time()
        # interpolate Sk
        kmf = dft.KUKS(mycell, kmesh_abs)
        Sk_int = kmf.get_ovlp(kpt=band_kpts_abs)
        # interpolate Fk and Sigma_tk
        Fk_int = winter.interpolate(
            Fk, kmesh_scaled, band_kpts, dim=3, hermi=True, debug=debug
        )
        Sigma_tk_int = winter.interpolate_tk_object(
            Sigma_tk, kmesh_scaled, band_kpts, dim=3, hermi=True, debug=debug
        )
        # form G_tk by solving dyson
        G_tk_int = dyson.solve_dyson(
            Fk_int, Sk_int, Sigma_tk_int, mu, mbo.ir
        )
        t2 = time.time()
        print("Time required for Wannier interpolation: ", t2 - t1)
    else:
        G_tk_int = mbo.gtau
        Sigma_tk_int = mbo.sigma
        Fk_int = mbo.fock
        Sk_int = mbo.S

    #
    # Orthogonalization and Nevanlinna
    #

    if orth_ao == 'sao':
        print("Transforming interpolated Gtau to SAO basis")
        Gt_ortho = orth.sao_orth(G_tk_int, Sk_int, type='g')
    elif orth_ao == 'co':
        print("Transforming interpolated Gtau to Canonical basis")
        Gt_ortho = orth.canonical_orth(G_tk_int, Sk_int, type='g')

    Gt_ortho_diag = np.einsum('tskii -> tski', Gt_ortho)

    # NOTE: The user can now control parameters that go into analytic cont.
    #       such as no. of real freqs (n_real), w_min, w_max, and eta.
    print("Starting Nevanlinna")
    t3 = time.time()
    freqs, A_w = mbo.AC_nevanlinna(
        gtau_orth=Gt_ortho_diag,
        n_real=10001, w_min=-10, w_max=10, eta=0.01
    )
    t4 = time.time()
    print("Time required for Nevanlinna AC: ", t4 - t3)

    # Save interpolated data to HDF5
    f = h5py.File(output, 'w')
    if wannier:
        f["kpts_interpolate"] = band_kpts
        f["kpath"] = kpath
        f["sp_points"] = sp_points
    f["mu"] = mu
    f["nevanlinna/freqs"] = freqs
    f["nevanlinna/dos"] = A_w
    f.close()
