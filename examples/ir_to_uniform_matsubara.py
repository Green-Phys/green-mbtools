#!/usr/bin/env python3
"""Transform IR-grid G(tau)/Sigma(tau) to a uniformly spaced Matsubara grid.

This script initializes green_mbtools.pesto.mb.MB_post from a GREEN
simulation file and its corresponding input file, then transforms
G_tau and/or Sigma_tau to user-defined fermionic Matsubara frequencies.
"""

import argparse
import numpy as np
import h5py

from green_mbtools.pesto import mb
from ir2newgrid import TransformIR


def build_uniform_fermionic_mesh(beta, n_iw, n_start=0):
    """Return a uniformly spaced fermionic Matsubara mesh.

    Frequencies are omega_n = (2n + 1) pi / beta for n in
    [n_start, n_start + n_iw).
    """
    n = np.arange(n_start, n_start + n_iw, dtype=int)
    omega = (2 * n + 1) * np.pi / beta
    return n, omega


def main():
    parser = argparse.ArgumentParser(
        description="Transform IR-grid G/Sigma to uniformly spaced Matsubara frequencies"
    )
    parser.add_argument(
        "--sim",
        required=True,
        help="Path to GW/GF2 simulation file (sim.h5)",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to corresponding input file (input.h5)",
    )
    parser.add_argument(
        "--ir-file",
        required=True,
        help="Path to IR basis file used in the simulation",
    )
    parser.add_argument(
        "--output",
        default="matsubara_uniform.h5",
        help="Output HDF5 path",
    )
    parser.add_argument(
        "--quantity",
        choices=["G", "Sigma", "both"],
        default="both",
        help="Which quantity to transform",
    )
    parser.add_argument(
        "--n-iw",
        type=int,
        default=512,
        help="Number of Matsubara frequencies to generate",
    )
    parser.add_argument(
        "--n-start",
        type=int,
        default=0,
        help="Starting Matsubara index n in omega_n=(2n+1)pi/beta",
    )
    parser.add_argument(
        "--legacy-ir",
        action="store_true",
        help="Use legacy IR file format",
    )
    args = parser.parse_args()

    if args.n_iw <= 0:
        raise ValueError("--n-iw must be a positive integer")
    if args.n_start < 0:
        raise ValueError("--n-start must be non-negative")

    mb_obj = mb.initialize_MB_post(
        sim_path=args.sim,
        input_path=args.input,
        ir_file=args.ir_file,
        legacy_ir=args.legacy_ir,
    )

    beta = mb_obj.beta
    n_indices, omega = build_uniform_fermionic_mesh(beta, args.n_iw, args.n_start)

    transformer = TransformIR(args.ir_file, beta, statistics="fermi")

    # TransformIR expects data on the interior IR tau nodes (without endpoints 0 and beta).
    gtau_ir = mb_obj.gtau[1:-1]
    sigma_ir = mb_obj.sigma[1:-1]

    out_data = {
        "beta": beta,
        "matsubara_n": n_indices,
        "omega": omega,
    }

    if args.quantity in ("G", "both"):
        out_data["G_iw"] = transformer.transform_tau_to_new_omega(gtau_ir, omega)
    if args.quantity in ("Sigma", "both"):
        out_data["Sigma_iw"] = transformer.transform_tau_to_new_omega(sigma_ir, omega)

    with h5py.File(args.output, "w") as fout:
        for key, value in out_data.items():
            fout[key] = value

    print("Done.")
    print(f"beta = {beta}")
    print(f"n_iw = {args.n_iw}, n_start = {args.n_start}")
    print(f"omega range = [{omega[0]}, {omega[-1]}]")
    if "G_iw" in out_data:
        print(f"G_iw shape: {out_data['G_iw'].shape}")
    if "Sigma_iw" in out_data:
        print(f"Sigma_iw shape: {out_data['Sigma_iw'].shape}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()