#!/usr/bin/env python3
"""
Scaling benchmark: compute_ewald_correction CC-forced (old) vs RS-aware (new).

Measures wall-clock time and max EW-tensor deviation for a periodic cell
as a function of k-mesh size.

Usage
-----
    # H2, 1x1x1 through 3x3x3 (fast, ~10 s)
    python tests/ewald_scaling_bench.py

    # Si with gth-szv, 2x2x2 through 4x4x4
    python tests/ewald_scaling_bench.py --system si

    # Si with gth-dzvp (larger auxbasis, more realistic)
    python tests/ewald_scaling_bench.py --system si --basis gth-dzvp

    # extend by one mesh (4x4x4 for H2, 5x5x5 for Si)
    python tests/ewald_scaling_bench.py --system si --large

    # repeat each measurement N times and report median
    python tests/ewald_scaling_bench.py --nreps 3

Output
------
    ewald_scaling_<system>.png  (log-log timing + deviation)
    table to stdout
"""

import argparse
import os
import sys
import time
import tempfile
import shutil

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyscf.pbc import gto, df

sys.path.insert(0, os.path.dirname(__file__))
from ewald_rsgdf_test import _run_ewald_old, _run_ewald_new


# ── cell factories ────────────────────────────────────────────────────────────

def _make_h2_cell(basis="gth-szv"):
    cell = gto.Cell()
    cell.atom = "H 0 0 0; H 0 0 1.4"
    cell.basis = basis
    cell.pseudo = "gth-pade"
    cell.a = np.diag([4.0, 4.0, 5.0])
    cell.verbose = 0
    cell.build()
    return cell


def _make_si_cell(basis="gth-szv"):
    # Diamond-cubic Si, a = 5.43 Å = 10.263 bohr
    # FCC primitive vectors (in Å)
    a = 2.715  # a/2
    cell = gto.Cell()
    cell.atom = "Si 0 0 0; Si 1.3575 1.3575 1.3575"
    cell.a = np.array([[0., a, a], [a, 0., a], [a, a, 0.]])
    cell.basis = basis
    cell.pseudo = "gth-pade"
    cell.verbose = 0
    cell.build()
    return cell


SYSTEMS = {
    "h2": {
        "factory":       _make_h2_cell,
        "default_meshes": [(1, 1, 1), (2, 2, 2), (3, 3, 3)],
        "large_mesh":    (4, 4, 4),
        "label":         "H₂",
    },
    "si": {
        "factory":       _make_si_cell,
        "default_meshes": [(2, 2, 2), (3, 3, 3), (4, 4, 4)],
        "large_mesh":    (5, 5, 5),
        "label":         "Si",
    },
}


# ── per-mesh benchmark ────────────────────────────────────────────────────────

def _bench_one(cell, kmesh_dims, nreps=1, tmpdir=None):
    """Time old and new Ewald paths; return timing and max EW-tensor deviation.

    tmpdir: directory for large intermediate HDF5 files (default: local ./bench_tmp).
    Using a local directory avoids /var/folders space exhaustion on macOS.
    """
    if tmpdir is None:
        tmpdir = os.path.join(os.getcwd(), "bench_tmp")
    os.makedirs(tmpdir, exist_ok=True)

    kmesh = cell.make_kpts(list(kmesh_dims))
    Nk    = len(kmesh)
    nao   = cell.nao_nr()

    # pre-build maindf — shared cost, not timed
    f_main = tempfile.mktemp(dir=tmpdir, suffix="_main.h5")
    maindf = df.GDF(cell)
    maindf.kpts = kmesh
    maindf.verbose = 0
    maindf._cderi_to_save = f_main
    maindf._cderi = f_main
    maindf.build()

    t_old_runs, t_new_runs = [], []
    max_dev = 0.0

    try:
        for _ in range(nreps):
            f_old = tempfile.mktemp(dir=tmpdir, suffix="_old.h5")
            f_new = tempfile.mktemp(dir=tmpdir, suffix="_new.h5")
            try:
                t0 = time.perf_counter()
                _run_ewald_old(cell, kmesh, maindf, nao, f_old)
                t_old_runs.append(time.perf_counter() - t0)

                t0 = time.perf_counter()
                _run_ewald_new(cell, kmesh, maindf, nao, f_new)
                t_new_runs.append(time.perf_counter() - t0)

                with h5py.File(f_old, "r") as fold, h5py.File(f_new, "r") as fnew:
                    for i in range(Nk):
                        ew_old = fold["EW"][str(i)][...].view(np.complex128)
                        ew_new = fnew["EW"][str(i)][...].view(np.complex128)
                        max_dev = max(max_dev, np.abs(ew_old - ew_new).max())
            finally:
                for f in (f_old, f_new):
                    if os.path.exists(f):
                        os.remove(f)
    finally:
        if os.path.exists(f_main):
            os.remove(f_main)

    return {
        "label":   "×".join(str(d) for d in kmesh_dims),
        "Nk":      Nk,
        "t_old":   np.median(t_old_runs),
        "t_new":   np.median(t_new_runs),
        "max_dev": max_dev,
    }


# ── table printer ─────────────────────────────────────────────────────────────

def _print_table(rows, system_label, basis):
    title = f"compute_ewald_correction — {system_label} / {basis}"
    hdr = (f"{'k-mesh':>10}  {'Nk':>5}  {'t_old (s)':>10}  {'t_new (s)':>10}"
           f"  {'speedup':>8}  {'max|ΔEW|':>12}  note")
    sep = "─" * len(hdr)
    print(f"\n{title}")
    print(sep)
    print(hdr)
    print(sep)
    for r in rows:
        sp = r["t_old"] / r["t_new"] if r["t_new"] > 0 else float("nan")
        # large deviation means old (CC-forced) method was inconsistent
        note = "old-method error" if r["max_dev"] > 1e-3 else "✓ consistent"
        print(f"{r['label']:>10}  {r['Nk']:>5d}  {r['t_old']:>10.3f}"
              f"  {r['t_new']:>10.3f}  {sp:>8.2f}x  {r['max_dev']:>12.2e}  {note}")
    print(sep)
    print("  note: max|ΔEW| = max absolute deviation between CC-forced (old) and")
    print("        RS-aware (new) EW tensors.  Large values indicate the old CC")
    print("        lattice-sum j2c metric was inconsistent; the new RS path is correct.")


# ── plot ──────────────────────────────────────────────────────────────────────

def _fit_slope(xvals, yvals):
    """Power-law fit log(y) = slope*log(x) + c; returns (slope, c)."""
    logx = np.log(xvals.astype(float))
    logy = np.log(yvals.astype(float))
    return np.polyfit(logx, logy, 1)


def _plot(rows, system_label, basis, outfile):
    Nk_vals  = np.array([r["Nk"]      for r in rows], dtype=float)
    t_old    = np.array([r["t_old"]   for r in rows])
    t_new    = np.array([r["t_new"]   for r in rows])
    max_devs = np.array([r["max_dev"] for r in rows])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(f"compute_ewald_correction — {system_label} / {basis}", fontsize=13)

    # ── left: timing ─────────────────────────────────────────────────────────
    ax1.loglog(Nk_vals, t_old, "o-", color="tab:red",  lw=2, ms=7,
               label="CC-forced (old)")
    ax1.loglog(Nk_vals, t_new, "s-", color="tab:blue", lw=2, ms=7,
               label="RS-aware (new)")

    xfit = np.linspace(Nk_vals.min(), Nk_vals.max(), 80)
    for t_arr, color in [(t_old, "tab:red"), (t_new, "tab:blue")]:
        if len(Nk_vals) >= 3:
            slope, intercept = _fit_slope(Nk_vals, t_arr)
            ax1.loglog(xfit, np.exp(intercept) * xfit**slope, "--",
                       color=color, alpha=0.45,
                       label=f"  fit: $N_k^{{{slope:.2f}}}$")

    ax1.set_xlabel("Number of k-points $N_k$")
    ax1.set_ylabel("Wall-clock time (s)")
    ax1.set_title("Timing vs $N_k$")
    ax1.legend(fontsize=9)
    ax1.grid(True, which="both", ls=":", alpha=0.5)

    # ── right: deviation ─────────────────────────────────────────────────────
    ax2.loglog(Nk_vals, max_devs, "D-", color="tab:green", lw=2, ms=7,
               label=r"max$|$EW$_{\rm CC}$ − EW$_{\rm RS}|$")
    ax2.axhline(1e-8, color="gray", ls="--", lw=1,
                label="H₂ tolerance (1e-8)")
    ax2.axhline(1e-6, color="orange", ls=":", lw=1,
                label="old-method error threshold")
    ax2.set_xlabel("Number of k-points $N_k$")
    ax2.set_ylabel("Max|EW$_{\\rm CC}$ − EW$_{\\rm RS}$|")
    ax2.set_title("CC-forced vs RS-aware inconsistency\n"
                  r"(large = old CC j2c was wrong)")
    ax2.legend(fontsize=8)
    ax2.grid(True, which="both", ls=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    print(f"Plot saved to {outfile}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--system", choices=list(SYSTEMS), default="h2",
                        help="periodic system (default: h2)")
    parser.add_argument("--basis",  default=None,
                        help="override basis set (default: gth-szv)")
    parser.add_argument("--large",  action="store_true",
                        help="add one extra k-mesh beyond the default range")
    parser.add_argument("--nreps",  type=int, default=1,
                        help="timing repetitions per mesh (default: 1)")
    parser.add_argument("--output", default=None,
                        help="output plot filename (default: ewald_scaling_<system>.png)")
    parser.add_argument("--tmpdir", default=None,
                        help="directory for large scratch HDF5 files (default: ./bench_tmp)")
    args = parser.parse_args()

    cfg    = SYSTEMS[args.system]
    basis  = args.basis or "gth-szv"
    meshes = list(cfg["default_meshes"])
    if args.large:
        meshes.append(cfg["large_mesh"])
    outfile = args.output or f"ewald_scaling_{args.system}.png"

    cell = cfg["factory"](basis=basis)
    nao  = cell.nao_nr()
    print(f"\nSystem : {cfg['label']}  |  basis: {basis}  |  nao: {nao}")
    print(f"Meshes : {meshes}")
    print(f"Reps   : {args.nreps}\n")

    rows = []
    for dims in meshes:
        Nk = dims[0] * dims[1] * dims[2]
        print(f"  {'×'.join(str(d) for d in dims)}  ({Nk} k-pts) ...", flush=True)
        row = _bench_one(cell, dims, nreps=args.nreps, tmpdir=args.tmpdir)
        rows.append(row)
        sp = row["t_old"] / row["t_new"]
        print(f"    old={row['t_old']:.2f}s  new={row['t_new']:.2f}s  "
              f"speedup={sp:.2f}x  max|ΔEW|={row['max_dev']:.2e}")

    _print_table(rows, cfg["label"], basis)
    _plot(rows, cfg["label"], basis, outfile)


if __name__ == "__main__":
    main()
