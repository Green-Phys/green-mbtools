#!/usr/bin/env python3
"""Transform G(tau) and Sigma(tau) from IR tau grid to DLR Matsubara frequencies."""

import sys
import os
import numpy as np
import h5py
import argparse
from pydlr import dlr

# Ensure the parent directory (AC/) is on the path for util.ir2newgrid imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="Transform G/Sigma from IR tau to DLR Matsubara frequencies")
    parser.add_argument("--data_file", type=str, required=True, help="Path to seet_results.h5")
    parser.add_argument("--ir_file", type=str, required=True, help="Path to IR basis HDF5 (e.g. 1e4.h5)")
    parser.add_argument("--output", type=str, required=True, help="Output HDF5 file path")
    parser.add_argument("--Euv", type=float, default=1e2, help="DLR energy cutoff")
    parser.add_argument("--eps", type=float, default=1e-12, help="DLR accuracy")
    parser.add_argument("--use_ir2newgrid", action="store_true",
                        help="Use Method 1 (IR->DLR via TransformIR); otherwise use Method 2 (direct pydlr lstsq)")
    args = parser.parse_args()

    # Load data
    with h5py.File(args.data_file, 'r') as f:
        last_iter = f['iter'][()]
        grp = f[f'iter{last_iter}']
        G_tau_full = grp['G_tau/data'][:]
        tau_mesh = grp['G_tau/mesh'][:]
        Sigma_tau_full = grp['Selfenergy/data'][:]

    beta = tau_mesh[-1]
    print(f"Last iteration: {last_iter}")
    print(f"beta = {beta}")
    print(f"Data shape: {G_tau_full.shape}")

    # Setup DLR
    dlrGrid = dlr(lamb=beta * args.Euv, eps=args.eps, xi=-1)
    iw_dlr = dlrGrid.get_matsubara_frequencies(beta)
    omega_dlr = iw_dlr.imag
    nw_dlr = len(dlrGrid.dlrrf)
    print(f"DLR rank: {nw_dlr}")

    if args.use_ir2newgrid:
        # Method 1: IR -> DLR via TransformIR
        from util.ir2newgrid import TransformIR

        tr = TransformIR(args.ir_file, beta)

        # Strip endpoints (interior 108 points)
        G_tau_ir = G_tau_full[1:-1]
        Sigma_tau_ir = Sigma_tau_full[1:-1]

        G_dlr = tr.transform_tau_to_new_omega(G_tau_ir, omega_dlr)
        Sigma_dlr = tr.transform_tau_to_new_omega(Sigma_tau_ir, omega_dlr)
        method = "ir2newgrid"
        print(f"Method 1 (IR->DLR) done: G shape = {G_dlr.shape}")

    else:
        # Method 2: Direct pydlr lstsq
        ns, nk, no1, no2 = G_tau_full.shape[1:]

        G_dlr = np.zeros((nw_dlr, ns, nk, no1, no2), dtype=complex)
        Sigma_dlr = np.zeros_like(G_dlr)

        for s in range(ns):
            for k in range(nk):
                G_coeff = dlrGrid.lstsq_dlr_from_tau(tau_mesh, G_tau_full[:, s, k, :, :], beta)
                Sig_coeff = dlrGrid.lstsq_dlr_from_tau(tau_mesh, Sigma_tau_full[:, s, k, :, :], beta)

                G_dlr[:, s, k, :, :] = beta * np.tensordot(dlrGrid.T_qx, G_coeff, axes=(1, 0))
                Sigma_dlr[:, s, k, :, :] = beta * np.tensordot(dlrGrid.T_qx, Sig_coeff, axes=(1, 0))

        method = "pydlr"
        print(f"Method 2 (pydlr lstsq) done: G shape = {G_dlr.shape}")

    # Save output
    with h5py.File(args.output, 'w') as f:
        f['G_dlr'] = G_dlr
        f['Sigma_dlr'] = Sigma_dlr
        f['omega'] = omega_dlr
        f['dlrrf'] = dlrGrid.dlrrf
        f['beta'] = beta
        f['Euv'] = args.Euv
        f['eps'] = args.eps
        f['method'] = method

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
