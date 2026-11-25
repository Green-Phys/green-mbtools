"""Offline conversion of a complete captured-frame reference; never overwrites."""

import argparse
import json
import h5py
from pyscf.pbc import gto
from green_mbtools.mint.integral_symmetry_gauge import CapturedGauges
from green_mbtools.mint.integral_symmetry_geometry import PhysicalPairMap
from green_mbtools.mint.integral_symmetry_io import write_archive

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--source",
    required=True,
    help="Complete reference with Cell, AuxCell, pair_q, captured_C, factors",
)
parser.add_argument("--out", required=True)
parser.add_argument("--chunk-size", type=int, default=32)
parser.add_argument("--producer-revision", required=True)
args = parser.parse_args()
with h5py.File(args.source) as f:
    cell, auxcell = (gto.loads(f[name][()].decode()) for name in ("Cell", "AuxCell"))
    kpts, qpts = f["kpts"][()], f["qpts"][()]
    pair_q = {
        (i, j): int(f[f"pair_q/{i}_{j}"][()])
        for i in range(len(kpts))
        for j in range(len(kpts))
    }
    kind = f.attrs.get("set_kind", "hf")
    kernel = f.attrs.get("kernel_id", "ordinary-Coulomb")
    gauges = CapturedGauges(args.source, len(qpts), kernel, dataset_prefix="captured_C")

    def read_factor(i, j):
        dataset = f[f"factors/{i}_{j}"]
        if dataset.size * dataset.dtype.itemsize > 128 * 1024**2:
            raise ValueError("Complete source factor exceeds the 128 MiB buffer bound")
        return dataset[()]

    geometry = PhysicalPairMap(
        cell, auxcell, kpts, qpts, pair_q, gauges, set_id=args.source
    )
    result = write_archive(
        geometry,
        read_factor,
        args.out,
        set_kind=kind,
        finite_size_kind="ewald" if kernel == "ewald-Coulomb" else "none",
        producer_revision=args.producer_revision,
        chunk_size=args.chunk_size,
    )
print(json.dumps(result, indent=2))
