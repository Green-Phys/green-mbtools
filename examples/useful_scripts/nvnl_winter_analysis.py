import h5py
import time
import argparse
import numpy as np
from ase.dft.kpoints import bandpath
# sc_special_points, get_bandpath
from pyscf.pbc import gto, dft
from green_mbtools.pesto import mb, orth, winter, dyson, analyt_cont, spectral


def main():
    #
    # Input data for Wannier interpolation and analytic continuation
    #
    
    # Default parameters
    parser = argparse.ArgumentParser(
        description="Wannier interpolaotion and Nevanlinna analytic continuation"
    )
    parser.add_argument("--x2c", type=int, default=0, help="Level of relativity")
    parser.add_argument("--beta", type=float, default=100, help="Inverse temperature")
    parser.add_argument("--eta", type=float, default=0.001, help="Broadening")
    parser.add_argument("--wannier", type=int, default=1, help="Toggle Wannier interpolation")
    parser.add_argument("--celltype", type=str, default="cubic", help="Type of lattice: cubic, diamond, etc.")
    parser.add_argument(
        "--bandpath", type=str, nargs="+", default=["L", "G", "X", "K", "G"],
        help="High symmetry path for the band structure, e.g. L G X K G"
    )
    parser.add_argument("--bandpts", type=int, default=50, help="Number of k-points used in the band path")
    parser.add_argument("--input", type=str, default="input.h5", help="Input file used in GW calculation")
    parser.add_argument("--sim", type=str, default="sim.h5", help="Output of UGF2 code, i.e., sim.h5")
    parser.add_argument(
        "--iter", type=int, default=-1, help="Iteration number of the scGW cycle to use for continuation"
    )
    parser.add_argument(
        "--ir_file", type=str, default=None, help="HDF5 file that contains information about the IR grid."
    )
    parser.add_argument(
        "--out", type=str, default='winter_out.h5', help="Name for output file (should be .h5 format)"
    )
    parser.add_argument("--orth", type=str, default='sao', help="Choice of orthonormal basis (sao or co)")
    parser.add_argument("--mo_range", type=int, nargs=2, default=[0, -1], help="MO range for analytic continuation")
    parser.add_argument(
        "--nw_real", type=int, default=1001,
        help="Number of real frequency points."
    )
    parser.add_argument(
        "--e_max", type=float, default=2.0,
        help="Maximum energy limit for extrapolation in eV."
    )
    parser.add_argument(
        "--e_min", type=float, default=-2.0,
        help="Minimal energy limit for extrapolation in eV."
    )
    
    args = parser.parse_args()
    
    #
    # First perform Wannier interpolation, then Nevanlinna to get the bands
    #
    
    # Parameters in the calculation
    T_inv = args.beta
    wannier = args.wannier
    bandpath_str = args.bandpath  # clean up spaces
    bandpts = args.bandpts
    input_path = args.input
    sim_path = args.sim
    it = args.iter
    ir_file = args.ir_file
    output = args.out
    orth_ao = args.orth
    nw_real = args.nw_real
    e_max = args.e_max
    e_min = args.e_min
    eta = args.eta
    
    
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
    reduced_kmesh_scaled = kmesh_scaled[ir_list]
    nk = index.shape[0]
    ink = ir_list.shape[0]
    Sk = f['HF/S-k'][()]
    Hk = f['HF/H-k'][()].view(complex)
    Hk = Hk.reshape(Hk.shape[:-1])
    f.close()
    
    # Pyscf object to generate k points
    mycell = gto.loads(cell)
    
    # Use ase to generate the kpath
    a_vecs = np.genfromtxt(mycell.a.replace(',', ' ').splitlines(), dtype=float)
    path = bandpath(bandpath_str, a_vecs, npoints=bandpts)
    band_kpts = path.kpts
    kpath, sp_points, labels = path.get_linear_kpoint_axis()
    
    # ASE will give scaled band_kpts. We need to transform them to absolute values
    # using mycell.get_abs_kpts
    band_kpts_abs = mycell.get_abs_kpts(band_kpts)
    
    print("Reading sim file")
    f = h5py.File(sim_path, 'r')
    if it == -1:
        it = f["iter"][()]
    rGk = f["iter" + str(it) + "/G_tau/data"][()]
    rSigma_inf = f["iter" + str(it) + "/Sigma1"][()]
    rSigmak = f["iter" + str(it) + "/Selfenergy/data"][()]
    mu = f["iter"+str(it)+"/mu"][()]
    f.close()
    print("Sigmak shape: ", rSigmak.shape)
    
    print("Transfrom quantities to full BZ")
    if args.x2c:
        Sigma_inf_k = mb.to_full_bz_TRsym(rSigma_inf, conj_list, ir_list, index, 1)
        G_tk = mb.to_full_bz_TRsym(rGk, conj_list, ir_list, index, 2)
        Sigma_tk = mb.to_full_bz_TRsym(rSigmak, conj_list, ir_list, index, 2)
    else:
        Sigma_inf_k = mb.to_full_bz(rSigma_inf, conj_list, ir_list, index, 1)
        G_tk = mb.to_full_bz(rGk, conj_list, ir_list, index, 2)
        Sigma_tk = mb.to_full_bz(rSigmak, conj_list, ir_list, index, 2)
    
    # Build Fock
    Fk = Hk + Sigma_inf_k
    
    print(Fk.shape)
    print(Sigma_tk.shape)
    del rSigmak, rSigma_inf
    
    
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
        if args.x2c == 0:
            kmf = dft.KUKS(mycell, kmesh_abs)
            Sk_int = kmf.get_ovlp(mycell, band_kpts_abs)
            Sk_int = np.array((Sk_int, Sk_int))
        else:
            kmf = dft.KGKS(mycell, kmesh_abs).x2c1e()
            Sk_int = kmf.get_ovlp(mycell, band_kpts_abs)
            Sk_int = Sk_int.reshape((1, ) + Sk_int.shape)
        # interpolate Fk and Sigma_tk
        Fk_int = winter.interpolate(
            Fk, kmesh_scaled, band_kpts, dim=3, hermi=True
        )
        Sigma_tk_int = winter.interpolate_tk_object(
            Sigma_tk, kmesh_scaled, band_kpts, dim=3, hermi=True
        )
        # form G_tk by solving dyson
        print('Number of iw: ', mbo.ir.wsample.shape[0])
        G_tk_int = dyson.solve_dyson(
            Fk_int, Sk_int, Sigma_tk_int, mu, mbo.ir
        )
        t2 = time.time()
        print("Time required for Wannier interpolation: ", t2 - t1)
        print('Sk_int shape: ', Sk_int.shape)
        print('Fk_int shape: ', Fk_int.shape)
        print('Sigma_tk_int shape: ', Sigma_tk_int.shape)
        print('G_tk_int shape: ', G_tk_int.shape)
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
    elif orth_ao == 'mo':
        print("Transforming interpolated Gtau to MO basis")
        fk_eigs, mo_vecs = spectral.compute_mo(Fk_int, Sk_int)
        mo_vecs_adj = np.einsum('skba -> skab', mo_vecs.conj())
        s_c = np.einsum('skab, skbc -> skac', Sk_int, mo_vecs)
        cdag_s = np.einsum('skab, skbc -> skac', mo_vecs_adj, Sk_int)
        Gt_ortho = np.einsum('skab, wskbc, skcd -> wskad', cdag_s, G_tk_int, s_c, optimize=True)
    
    noccs_orth = -np.einsum('skaa -> ska', Gt_ortho[-1])
    Gt_ortho_diag = np.einsum('tskii -> tski', Gt_ortho)
    if orth_ao == 'mo':
        orb_min = args.mo_range[0]
        orb_max = args.mo_range[1]
        Gt_ortho_diag = Gt_ortho_diag[:, :, :, orb_min:orb_max]
    
    
    # NOTE: The user can now control parameters that go into analytic continuation
    #       such as no. of real freqs (n_real), w_min, w_max, and eta.
    print("Starting Nevanlinna")
    t3 = time.time()
    nw = mbo.ir.wsample.shape[0]
    iw_inp = mbo.ir.wsample[nw//2::1]
    Gw_inp = mbo.ir.tau_to_w(Gt_ortho_diag)[nw//2::1]
    freqs, A_w = analyt_cont.nevan_run(
        Gw_inp, iw_inp, n_real=nw_real, w_min=e_min, w_max=e_max, eta=eta
    )
    t4 = time.time()
    print("Time required for Nevanlinna AC: ", t4 - t3)
    
    # Save interpolated data to HDF5
    # This file contains the Green's function, Fock matrix, etc. at the k-points
    # along the selected path
    f = h5py.File(output, 'w')
    # f["S-k"] = Sk_int
    f["kpts_interpolate"] = band_kpts
    f["kpath"] = kpath
    f["sp_points"] = sp_points
    f["sp_labels"] = labels
    f["iter"+str(it)+"/mu"] = mu
    f["nevanlinna/freqs"] = freqs
    f["nevanlinna/dos"] = A_w
    f["occs"] = noccs_orth
    f.close()

    # Traced DOS
    f = h5py.File('traced_' + output, 'w')
    f["kpts_interpolate"] = band_kpts
    f["kpath"] = kpath
    f["sp_points"] = sp_points
    f["sp_labels"] = labels
    f["iter"+str(it)+"/mu"] = mu
    f["nevanlinna/freqs"] = freqs
    f["nevanlinna/dos"] = np.einsum('wska -> wk', A_w)
    f["occs"] = noccs_orth
    f.close()


if __name__ == "__main__":
    main()

