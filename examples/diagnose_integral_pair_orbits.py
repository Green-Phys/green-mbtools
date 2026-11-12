"""Diagnose the actual input mesh/crystal without building integral factors.

Run on GREEN_workstation:
  python examples/diagnose_integral_pair_orbits.py --input INPUT.h5
No input data or legacy symmetry maps are modified.
"""

import argparse
import json

import h5py
from pyscf.pbc import gto

from green_mbtools.mint.integral_symmetry import (
    IntegerMesh,
    build_pair_orbits,
    validated_cell_operations,
    integral_kstruct,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--time-reversal", action="store_true")
    parser.add_argument("--exchange", action="store_true")
    args = parser.parse_args()
    with h5py.File(args.input, "r") as source:
        cell_dump = source["Cell"][()]
        if isinstance(cell_dump, bytes):
            cell_dump = cell_dump.decode()
        cell = gto.loads(cell_dump)
        scaled = source["symmetry/k/mesh_scaled"][()]
        actual_k = source["symmetry/k/mesh"][()]
        legacy_representatives = len(source["symmetry/pairs/kpair_irre_list"])
        if int(source["params/nao"][()]) != int(source["params/nso"][()]):
            raise ValueError("SG diagnostics initially require scalar orbitals")
    kstruct = integral_kstruct(cell, actual_k)
    operations = validated_cell_operations(cell, kstruct)
    orbits = build_pair_orbits(
        IntegerMesh.from_scaled(scaled),
        operations,
        time_reversal=args.time_reversal,
        exchange=args.exchange,
    )
    summary = orbits.summary()
    summary["legacy_representatives"] = legacy_representatives
    summary["candidate_crystal_operations"] = len(operations)
    summary["scope"] = "momentum orbit diagnostics; no omitted tensors"
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
