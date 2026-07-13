#!/usr/bin/env python3
"""Self-contained reproduction of the k-space symmetry AO Bloch-phase bug.

No data files needed -- it builds a small cell from scratch.

Background
----------
The AO symmetry operators ``k_sym_transform_ao`` (U_k) must satisfy the exact
one-electron identity for any space-group operation g mapping an irreducible
k-point k_ir to a full-BZ point k:

    X(k) = U_k @ X(k_ir) @ U_k^dagger        for X in {S(k), H(k), ...}

Before the fix, ``get_representation`` built the Bloch phase from atom
coordinates folded into [-0.5, 0.5) instead of the as-input positions the
integrals use, so any atom with input fractional coordinate >= 0.5 acquired a
spurious e^{i 2*pi k . L} factor and the identity was violated by O(1).

Why LiH (and not an H chain)?
-----------------------------
The error is a *relative* phase between atoms. In a single-species cell whose
atoms are all symmetry-equivalent (e.g. H at frac 0.25 and 0.75) it collapses
to a global phase and cancels -- that case round-trips to ~2e-16 even with the
bug. To expose it you need (i) an atom at fractional coordinate >= 0.5 and
(ii) *inequivalent* sublattices, so that some atoms stay fixed while others are
folded/swapped. Rock-salt LiH doubled along [111] (2 formula units) is the
smallest such system. A primitive cell (all atoms in [0, 0.5)) never exercises
the fold and was always correct -- which is why Ge/Si/diamond never broke.

Run
---
    python repro_ksym_phase_bug.py

Prints the round-trip residual for S(k) and the kinetic H(k). With the fix it is
~1e-14; without it, O(1) (~10).
"""
import sys
from pathlib import Path

import numpy as np
from pyscf.pbc import gto
from pyscf.pbc.lib import kpts as libkpts

# This script lives in <repo>/docs; prefer the in-repo source so the demo
# reflects the checked-out code rather than any separately installed copy.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from green_mbtools.mint.symmetry_utils import get_representation


def build_lih_supercell(a=4.0, shift=0.03):
    """Rock-salt LiH doubled along [111]; shift keeps atoms off the exact-0.5
    fold boundary (which is numerically fragile), while the second Li/H still
    sit at fractional >= 0.5 to exercise the folding path."""
    amat = np.array([[a, a / 2, a / 2], [a / 2, a, a / 2], [a / 2, a / 2, a]])
    frac = np.array([[0, 0, 0], [0.25] * 3, [0.5] * 3, [0.75] * 3]) + shift
    cart = frac @ amat
    cell = gto.Cell()
    cell.a = amat.tolist()
    cell.atom = [["Li", cart[0]], ["H", cart[1]], ["Li", cart[2]], ["H", cart[3]]]
    cell.basis = "gth-szv"
    cell.pseudo = "gth-pbe"
    cell.verbose = 0
    cell.space_group_symmetry = True
    cell.symmorphic = False
    cell.build()
    return cell


def max_roundtrip_residual(cell, kstruct, X):
    """max | U_k X(k_ir) U_k^dagger - X(k) | over the full BZ."""
    ibz_of = kstruct.ibz2bz[kstruct.bz2ibz]     # full-BZ index of each k's IBZ rep
    tr = kstruct.time_reversal_symm_bz
    worst = 0.0
    for ik in range(kstruct.nkpts):
        uop = get_representation(ik, kstruct.stars_ops_bz[ik], cell, kstruct)
        recon = uop @ X[ibz_of[ik]] @ uop.conj().T
        if tr[ik]:
            recon = recon.conj()
        worst = max(worst, np.max(np.abs(recon - X[ik])))
    return worst


def main():
    cell = build_lih_supercell()
    kstruct = libkpts.make_kpts(
        cell, cell.make_kpts([4, 4, 4]),
        space_group_symmetry=True, time_reversal_symmetry=True,
    )
    assert kstruct.nkpts_ibz < kstruct.nkpts, "supercell did not reduce; test is vacuous"

    overlap = np.asarray(cell.pbc_intor("int1e_ovlp", kpts=kstruct.kpts))
    kinetic = np.asarray(cell.pbc_intor("int1e_kin", kpts=kstruct.kpts))

    res_s = max_roundtrip_residual(cell, kstruct, overlap)
    res_h = max_roundtrip_residual(cell, kstruct, kinetic)

    print(f"LiH rock-salt 2-f.u. supercell: nao={cell.nao_nr()} "
          f"nk={kstruct.nkpts} ink={kstruct.nkpts_ibz}")
    print(f"  overlap S(k) round-trip residual = {res_s:.3e}")
    print(f"  kinetic H(k) round-trip residual = {res_h:.3e}")
    ok = max(res_s, res_h) < 1e-9
    print("  => symmetry identity holds" if ok
          else "  => BROKEN: k_sym_transform_ao Bloch phase is wrong")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
