"""Compare a representative-only build with a separate complete physical reference."""

import argparse
import json
from pathlib import Path
import h5py
from validate_complete_integral_reference import load_reference
from green_mbtools.mint.integral_symmetry_builder import build_representative_archive
from green_mbtools.mint.integral_symmetry_io import ArchiveReader
from green_mbtools.mint.integral_symmetry_transform import residual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    cell, aux, kpts, qpts, pair_q, gauges, factors = load_reference(args.reference)
    corrected = gauges[0].kernel_id == "ewald-Coulomb"
    summary = build_representative_archive(
        cell,
        kpts,
        aux._basis,
        args.out,
        args.work,
        corrected=corrected,
        producer_revision=args.revision,
        chunk_size=5,
        reference_get_factor=lambda i, j: factors[i, j],
        mesh=cell.mesh,
    )
    ordinary_count = corrected_count = 0
    ordinary = Path(args.work) / "representatives-cderi.h5"
    if ordinary.exists():
        with h5py.File(ordinary) as file:
            ordinary_count = len(file["j3c"])
    corrected_file = Path(args.work) / "legacy-ewald-cderi.h5"
    if corrected_file.exists():
        with h5py.File(corrected_file) as file:
            corrected_count = len(file["j3c"])
    assert ordinary_count == summary["ordinary_three_center_pairs"]
    assert corrected_count == summary["corrected_three_center_pairs"]
    assert ordinary_count + corrected_count == summary["representatives"]
    maximum = {"max_absolute": 0.0}
    with ArchiveReader(args.out) as reader:
        for i in range(len(kpts)):
            for j in range(len(kpts)):
                check = residual(reader.read(i, j), factors[i, j])
                if check["max_absolute"] > maximum["max_absolute"]:
                    maximum = {"pair": [i, j], **check}
    summary["complete_reference_residual"] = maximum
    Path(args.work, "validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
