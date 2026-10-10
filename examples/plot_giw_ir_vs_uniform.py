#!/usr/bin/env python3
"""Plot G(iw) on IR Matsubara grid vs uniform Matsubara grid.

This script reads GREEN test data via MB_post to get G(iw) on the IR
Matsubara mesh, reads a converted uniform-grid G(iw) file, and plots
both for a selected matrix element.
"""

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt

from green_mbtools.pesto import mb


def main():
    parser = argparse.ArgumentParser(description="Compare G(iw) on IR and uniform Matsubara grids")
    parser.add_argument("--sim", required=True, help="Path to sim.h5")
    parser.add_argument("--input", required=True, help="Path to input.h5")
    parser.add_argument("--ir-file", required=True, help="Path to IR basis file")
    parser.add_argument("--uniform-file", required=True, help="Path to output HDF5 from ir_to_uniform_matsubara.py")
    parser.add_argument("--output", default="tests/test_data/H2_GW/G_iw_ir_vs_uniform.png", help="Output plot path")
    parser.add_argument("--spin", type=int, default=0, help="Spin index")
    parser.add_argument("--k", type=int, default=0, help="k-point index")
    parser.add_argument("--i", type=int, default=0, help="Row orbital index")
    parser.add_argument("--j", type=int, default=0, help="Column orbital index")
    parser.add_argument("--legacy-ir", action="store_true", help="Use legacy IR file format")
    args = parser.parse_args()

    mb_obj = mb.initialize_MB_post(
        sim_path=args.sim,
        input_path=args.input,
        ir_file=args.ir_file,
        legacy_ir=args.legacy_ir,
    )

    # G(iw) on the IR Matsubara grid from MB_post's IR transform.
    giw_ir = mb_obj.ir.tau_to_w(mb_obj.gtau)
    w_ir = mb_obj.ir.wsample

    with h5py.File(args.uniform_file, "r") as f:
        giw_uniform = f["G_iw"][:]
        w_uniform = f["omega"][:]

    s, k, i, j = args.spin, args.k, args.i, args.j
    y_ir = giw_ir[:, s, k, i, j]
    y_uniform = giw_uniform[:, s, k, i, j]

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    axes[0].plot(w_ir, y_ir.real, "o", ms=3, label="IR grid")
    axes[0].plot(w_uniform, y_uniform.real, "-", lw=1.2, label="Uniform grid")
    axes[0].set_ylabel("Re G(iw)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(w_ir, y_ir.imag, "o", ms=3, label="IR grid")
    axes[1].plot(w_uniform, y_uniform.imag, "-", lw=1.2, label="Uniform grid")
    axes[1].set_ylabel("Im G(iw)")
    axes[1].set_xlabel("omega_n")
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"G(iw) comparison: s={s}, k={k}, i={i}, j={j}")
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    plt.show()

    print(f"Saved plot: {args.output}")
    print(f"IR points: {len(w_ir)}, Uniform points: {len(w_uniform)}")


if __name__ == "__main__":
    main()
