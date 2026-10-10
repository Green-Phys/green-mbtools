import h5py
import time
import argparse
import numpy as np
import scipy.linalg as LA
from ase.dft.kpoints import bandpath
# sc_special_points, get_bandpath
from pyscf.pbc import gto, dft
from green_mbtools.pesto import mb, winter, dyson, analyt_cont, spectral
from green_mbtools.pesto.ir import IR_factory


def wannier_interpolation():
    pass


def get_minor(arr, row, col):
    """
    Calculates the minor of a NumPy array at the specified row and column.

    Args:
      arr: The input NumPy array.
      row: The row index to exclude.
      col: The column index to exclude.

    Returns:
      The determinant of the submatrix (minor).
    """
    sub_matrix = np.delete(np.delete(arr, row, axis=0), col, axis=1)
    return sub_matrix


def main():
    #
    # Input data for Wannier interpolation and analytic continuation
    #

    # Default parameters
    parser = argparse.ArgumentParser(
        description="Wannier interpolaotion and Nevanlinna analytic continuation for zeros of Green's function"
    )
    # parser.add_argument("--x2c", type=int, default=0, help="Level of relativity")
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
    parser.add_argument("--mo_range", type=int, nargs=2, default=[0, -1], help="MO range for analytic continuation")
    args = parser.parse_args()

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

    #
    # Read input
    #

    print("Reading input file")
    f = h5py.File(input_path, 'r')
    cell = f["Cell"][()]
    nso = f["params/nso"][()]
    nao = f["params/nao"][()]
    ns = f["params/ns"][()]
    nk = f["params/nk"][()]
    is_ghf = nso == nao
    kmesh_abs = f["/grid/k_mesh"][()]
    kmesh_scaled = f["/grid/k_mesh_scaled"][()]
    index = f["/grid/index"][()]
    ir_list = f["/grid/ir_list"][()]
    conj_list = f["/grid/conj_list"][()]
    Sk = f['HF/S-k'][()]
    Hk = f['HF/H-k'][()].view(complex)
    Hk = Hk.reshape(Hk.shape[:-1])
    nelec = f["params/nel_cell"][()]
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
    
    print("Transform quantities to full BZ")
    if is_ghf:
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
        if not is_ghf:
            kmf = dft.KUKS(mycell, kmesh_abs)
            Sk_int = kmf.get_ovlp(mycell, band_kpts_abs)
            Sk_int = np.array((Sk_int, Sk_int))
        else:
            kmf = dft.KGKS(mycell, kmesh_abs)  # .x2c1e()
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
    # Transform to MO basis
    #
    
    print("Transforming interpolated Gtau to MO basis")
    fk_eigs, mo_vecs = spectral.compute_mo(Fk_int, Sk_int)
    mo_vecs_adj = np.einsum('skba -> skab', mo_vecs.conj())
    s_c = np.einsum('skab, skbc -> skac', Sk_int, mo_vecs)
    cdag_s = np.einsum('skab, skbc -> skac', mo_vecs_adj, Sk_int, optimize=True)
    Sigma_tk_ortho = np.einsum('skab, tskbc, skcd -> tskad', mo_vecs_adj, Sigma_tk_int, mo_vecs, optimize=True)
    Gt_ortho = np.einsum('skab, wskbc, skcd -> wskad', cdag_s, G_tk_int, s_c, optimize=True)

    # # Fix MO range
    # # -- NOTE: This is dangerous as we are not keeping track of hybridization.
    # original_nao = Gt_ortho.shape[-1]
    # if args.mo_range[1] == -1:
    #     args.mo_range[1] = original_nao
    # if args.mo_range[0] < 0 or args.mo_range[1] > original_nao or args.mo_range[0] >= args.mo_range[1]:
    #     raise ValueError(f"Invalid MO range {args.mo_range} for orbital space of size {original_nao}")
    
    # Gt_ortho = Gt_ortho[:, :, :, args.mo_range[0]:args.mo_range[1]]
    # Sigma_tk_ortho = Sigma_tk_ortho[:, :, :, args.mo_range[0]:args.mo_range[1]]
    # _, ns, nk, nao = Gt_ortho.shape[:4]
    # print("New number of orbitals for continuation: ", nao)

    # Fix MO range
    if nao > 16:
        if is_ghf:
            # for x2c1e
            orb_min = np.min((0, nelec - 8))
            orb_max = np.min((nao, nelec + 8))
            fermi_orbs = np.arange(orb_min, orb_max)
        else:
            # for non-relativistic, non-ghf
            orb_min = np.min((0, nelec//2 - 4))
            orb_max = np.min((nao, nelec//2 + 4))
            fermi_orbs = np.arange(orb_min, orb_max)
    else:
        fermi_orbs = np.arange(nao)

    #
    # Nevanlinna
    #

    w_min = -0.5
    w_max = 0.5
    nfreqs = 10001

    # NOTE: The user can now control parameters that go into analytic continuation
    #       such as no. of real freqs (n_real), w_min, w_max, and eta.
    myir = IR_factory(args.beta, args.ir_file)
    nw = myir.nw
    iw_pos = myir.wsample[nw//2:]

    print("IR transform sigma and Gtau from tau to iw")
    t3 = time.time()
    Sigma_iw_ortho = myir.tau_to_w(Sigma_tk_ortho)
    Giw_ortho = myir.tau_to_w(Gt_ortho)
    t4 = time.time()
    print("Time required for Fourier transform: ", t4 - t3)

    # Continuation for full GF
    Giw_full_inp_ao = Giw_ortho[nw//2:]
    Giw_full_inp = np.einsum('wskaa -> wska', Giw_full_inp_ao)[:, :, :, fermi_orbs[0]:fermi_orbs[-1]+1]
    # Perform analytic continuation
    freqs, Gw_full = analyt_cont.nevan_run(
        Giw_full_inp, iw_pos, n_real=nfreqs, w_min=w_min, w_max=w_max, eta=args.eta, spectral=False
    )
    t5 = time.time()
    print("Time for full analytic continuation: ", t5 - t4)

    # Continuation for zeros
    # Ensure fermi_orbs are within the valid range [0, nao)
    print("Orbitals to be considered: ", fermi_orbs)
    Aw = np.zeros((len(fermi_orbs), nfreqs, nk))
    eye_n1 = np.eye(nao - 1)
    G_i = np.zeros((nw, ns, nk, nao - 1, nao - 1), dtype=complex)
    for i, ao_i in enumerate(fermi_orbs):
        print("\n\nOrbital index: ", ao_i)
        print("Perform Dyson equation to get Green's function")
        G_i *= 0
        t6 = time.time()
        for s_idx in range(ns):
            for k_idx in range(nk):
                # Remove orbital in H1 and sigma infinity
                fock_i = get_minor(np.diag(fk_eigs[s_idx, k_idx]), ao_i, ao_i)
                # Remove orbital in Sigma_iw and form G_iw
                for w_idx, w_val in enumerate(myir.wsample):
                    sig_w = get_minor(Sigma_iw_ortho[w_idx, s_idx, k_idx], ao_i, ao_i)
                    G_i[w_idx, s_idx, k_idx] = LA.inv((1j * w_val + mu) * eye_n1 - fock_i - sig_w)
        t7 = time.time()
        print("Time required for Dyson transform: ", t7 - t6)
        # Do analytic continuation
        t8 = time.time()
        Gw_inp = np.einsum('wskaa -> wska', G_i[nw//2:])[:, :, :, fermi_orbs[0]:fermi_orbs[-1]+1]
        # Analytic continuation
        print("Starting Nevanlinna continuation")
        freqs, A_w_i = analyt_cont.nevan_run(
            Gw_inp, iw_pos, n_real=nfreqs, w_min=w_min, w_max=w_max, eta=args.eta, spectral=True
        )
        t9 = time.time()
        print("Time for analytic continuation: ", t9 - t8)
        # trace out spectral function
        Aw[i, :, :] = np.einsum('wska -> wk', A_w_i)

    # Save interpolated data to HDF5
    f = h5py.File(output, 'w')
    f["nevanlinna/freqs"] = freqs
    f["nevanlinna/Gw"] = Gw_full
    f["sp_labels"] = labels
    f["sp_points"] = sp_points
    f["kpath"] = kpath
    f.close()


if __name__ == "__main__":
    main()
