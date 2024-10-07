import h5py
import time
import argparse
import numpy as np
import scipy
from ase.dft.kpoints import sc_special_points, get_bandpath
from pyscf.pbc import gto
from mbanalysis import mb, orth, analyt_cont as ac

np.set_printoptions(precision=5, linewidth=200, suppress=True)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Nevanlinna analytic continuation for molecule.")
    parser.add_argument("--beta", type=float, default=1000, help="Inverse temperature")
    parser.add_argument("--input", type=str, default="input.h5", help="Input file used in GW calculation")
    parser.add_argument("--sim", type=str, default="sim.h5", help="Output of UGF2 code, i.e., sim.h5")
    parser.add_argument("--iter", type=int, default=-1, help="Iteration number of the scGW cycle to use for continuation")
    parser.add_argument("--grid_file", type=str, help="HDF5 file with IR grid information (new format).")
    parser.add_argument("--legacy_ir", type=bool, default=False, help="Toggles to old format of IR-grid file.")
    parser.add_argument("--out", type=str, default='ac_out.h5', help="Name for output file (should be .h5 format)")
    parser.add_argument("--e_min", type=float, default=-4.0, help="Minimum energy (in Hartree) for real axis")
    parser.add_argument("--e_max", type=float, default=4.0, help="Maximum energy (in Hartree) for real axis")
    parser.add_argument("--n_omega", type=int, default=8000, help="Number of points in the real axis")
    parser.add_argument("--eta", type=float, default=0.005, help="Broadening parameter for real axis")
    return parser.parse_args()

def read_h5_data(file_path):
    with h5py.File(file_path, 'r') as f:
        kmesh_scaled = f["/grid/k_mesh_scaled"][()]
        index = f["/grid/index"][()]
        ir_list = f["/grid/ir_list"][()]
        conj_list = f["/grid/conj_list"][()]

        scfFk = f["/HF/Fock-k"][()].view(complex)
        scfFk = scfFk.reshape(scfFk.shape[:-1])
        rSk = f["/HF/S-k"][()].view(complex)
        rSk = rSk.reshape(rSk.shape[:-1])
        rH0k = f["/HF/H-k"][()].view(complex)
        rH0k = rH0k.reshape(rH0k.shape[:-1])

        print(rH0k.shape)
        print(rSk.shape)
        print(scfFk.shape)
    return kmesh_scaled, index, ir_list, conj_list, scfFk, rSk, rH0k

def read_sim_data(sim_path, it):
    with h5py.File(sim_path, 'r') as f:
        if it == -1:
            it = f["iter"][()]
        rVeffk = f[f"iter{it}/Sigma1"][()].view(complex)
        rGk = f[f"iter{it}/G_tau/data"][()].view(complex)
        rSigmak = f[f"iter{it}/Selfenergy/data"][()].view(complex)
        tau_mesh = f[f"iter{it}/G_tau/mesh"][()]
        mu = f[f"iter{it}/mu"][()]
    return it, rVeffk, rGk, rSigmak, tau_mesh, mu

def main():
    args = parse_arguments()

    print("Reading input file")
    kmesh_scaled, index, ir_list, conj_list, scfFk, rSk, rH0k = read_h5_data(args.input)
    nk = index.shape[0]

    print("Reading sim file")
    it, rVeffk, rGk, rSigmak, tau_mesh, mu = read_sim_data(args.sim, args.iter)

    rFk = rH0k + rVeffk  # updated mean-field Fock

    print("Transforming quantities to full BZ")
    Fk = np.copy(rFk)
    Sk = np.copy(rSk)
    Sigma_tk = np.copy(rSigmak)
    G_tk = np.copy(rGk)

    ns = Fk.shape[0]
    mo_coeff = np.zeros_like(Fk)

    for s in range(ns):
        for k in range(nk):
            e, v = scipy.linalg.eigh(Fk[s, k, :], Sk[s, k, :])
            idx = np.argsort(e)
            mo_coeff[s, k, :] = v[:, idx]

    mo_coeff_adj = np.einsum('skpq -> skqp', mo_coeff.conj())

    print("Transforming quantities to MO basis")
    Fk_mo = np.einsum('skpr, skrt, sktq -> skpq', mo_coeff_adj, Fk, mo_coeff, optimize=True)
    Sk_mo = np.einsum('skpr, skrt, sktq -> skpq', mo_coeff_adj, Sk, mo_coeff, optimize=True)
    Sigma_tk_mo = np.einsum('skab, tskbc, skcd -> tskad', mo_coeff_adj, Sigma_tk, mo_coeff, optimize=True)

    # Initialize mbanalysis post processing
    mbo = mb.MB_post(
        fock=Fk_mo, sigma=Sigma_tk_mo, mu=mu, S=Sk_mo, kmesh=kmesh_scaled,
        beta=args.beta, ir_file=args.grid_file, legacy_ir=args.legacy_ir
    )

    Gt = mbo.gtau
    Gt_diag = np.einsum('tskii -> tski', Gt, optimize=True)
    nelec = int(np.round(-np.einsum('sa->', Gt_diag[-1, :, 0, :])).real)

    print("Starting Nevanlinna")
    start_time = time.time()

    nw = mbo.ir.wsample.shape[0]
    Gw = mbo.ir.tau_to_w(Gt_diag)
    w_pos = mbo.ir.wsample[nw // 2:]
    Gw_pos = Gw[nw // 2:]

    nskip = 2
    if nw // 2 < 40:
        idx = np.arange(0, nw // 2)
    else:
        idx1 = np.arange(0, 40, nskip)
        idx2 = np.arange(idx1[-1] + nskip, nw // 2)
        idx = np.concatenate((idx1, idx2))

    iw_inp = w_pos[idx]
    Gw_inp = Gw_pos[idx]

    freqs_aw, A_w = ac.nevan_run(
        Gw_inp, iw_inp, n_real=args.n_omega, w_min=args.e_min, w_max=args.e_max, eta=args.eta, prec=128,
    )
    
    elapsed_time = time.time() - start_time
    print("Time required for Nevanlinna AC:", elapsed_time)

    # Save interpolated data to HDF5
    with h5py.File(args.out, 'w') as f:
        f["S-k"] = Sk
        f["iter"] = it
        f[f"iter{it}/Fock-k"] = Fk_mo
        f[f"iter{it}/Selfenergy/data"] = Sigma_tk_mo
        f[f"iter{it}/Selfenergy/mesh"] = tau_mesh
        f[f"iter{it}/G_tau/data"] = G_tk
        f[f"iter{it}/G_tau/mesh"] = tau_mesh
        f[f"iter{it}/mu"] = mu
        f["HF/nelec"] = nelec
        f["nevanlinna/freqs"] = freqs_aw
        f["nevanlinna/dos"] = A_w
        f["HF/mo_coeff"] = mo_coeff  # after scGW iterations

if __name__ == "__main__":
    main()

