"""Report how crystal symmetry reduces momentum pairs in a GREEN input.

Read the cell and k mesh from a saved GREEN input file. Compare the spatial
orbit count with the baseline for the same time-reversal and pair exchange
flags. This diagnostic reads metadata and leaves the input file intact.

    python examples/diagnose_integral_pair_orbits.py --input input.h5 --time-reversal --exchange
"""

import argparse
import json

import h5py
from pyscf.pbc import gto

from green_mbtools.mint.integral_symmetry import (
    IDENTITY,
    build_pair_orbits,
    validated_cell_operations,
    integral_kstruct,
    validate_input_kmesh,
)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
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
    mesh = validate_input_kmesh(cell, actual_k, scaled)
    kstruct = integral_kstruct(cell, actual_k)
    operations = validated_cell_operations(cell, kstruct)
    orbits = build_pair_orbits(
        mesh,
        operations,
        time_reversal=args.time_reversal,
        exchange=args.exchange,
    )
    summary = orbits.summary()
    summary["legacy_representatives"] = legacy_representatives
    baseline = build_pair_orbits(
        mesh, [IDENTITY], time_reversal=args.time_reversal, exchange=args.exchange
    )
    summary["matched_baseline_representatives"] = len(baseline.representatives)
    summary["legacy_count_comparison"] = (
        "Stored legacy flags are not inferred; use matched_baseline_representatives for like-for-like reduction"
    )
    summary["candidate_crystal_operations"] = len(operations)
    summary["scope"] = "momentum orbit diagnostics; no omitted tensors"
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
