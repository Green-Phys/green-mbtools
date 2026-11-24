"""Prepare a bounded physical solver case, direct legacy, SG and expanded files.

All calculations run on GREEN_workstation. Original complete references remain
untouched. The input uses unreduced one-body k points so frozen G may break the
crystal symmetry independently of integral compression.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import h5py
import numpy as np
from pyscf.pbc import df, scf, tools
from pyscf.pbc.lib import kpts as libkpts
from importlib.metadata import version

from green_mbtools.mint import common_utils as comm
from green_mbtools.mint.integral_symmetry_geometry import PhysicalPairMap
from green_mbtools.mint.integral_symmetry_io import (
    write_archive,
    expand_archive,
    input_fingerprint,
    pack,
)
from validate_complete_integral_reference import load_reference


def direct_legacy(output, pairs, factors, chunk=5):
    output.mkdir()
    starts = np.arange(0, len(pairs), chunk, dtype=np.int64)
    for start in starts:
        first = factors[tuple(pairs[0])]
        buffer = np.zeros((chunk, *first.shape), dtype=complex)
        for i, pair in enumerate(pairs[start : start + chunk]):
            buffer[i] = factors[tuple(pair)]
        with h5py.File(output / f"VQ_{start}.h5", "w") as f:
            f[str(start)] = pack(buffer)
    with h5py.File(output / "meta.h5", "w") as f:
        f.attrs["__green_version__"] = version("green-mbtools")
        f["chunk_size"] = chunk
        f["chunk_indices"] = starts
        f["chunk_valid_count"] = np.minimum(chunk, len(pairs) - starts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hf", required=True)
    p.add_argument("--correlation", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument(
        "--hf-only",
        action="store_true",
        help="Prepare a bounded HF-only solver fixture",
    )
    args = p.parse_args()
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=False)
    hf_path = Path(args.hf).resolve()
    correlation_path = Path(args.correlation).resolve()
    cell, aux, kpts, qpts, pair_q, gauges, factors = load_reference(hf_path)
    os.chdir(output)
    shutil.copy2(hf_path.parent / "cderi.h5", output / "cderi.h5")
    provider = df.GDF(cell, kpts)
    provider.auxcell = aux
    provider._cderi = str(output / "cderi.h5")
    mean_field = scf.KRHF(cell, kpts)
    mean_field.with_df = provider
    mean_field.conv_tol = 1e-9
    mean_field.max_cycle = 40
    mean_field.max_memory = 4000
    mean_field.verbose = 3
    mean_field.kernel()
    identity = libkpts.make_kpts(
        cell, kpts, space_group_symmetry=False, time_reversal_symmetry=False
    )
    scaled = np.mod(cell.get_scaled_kpts(kpts), 1)
    mesh_shape = [
        len(np.unique(np.round(scaled[:, axis], 10) % 1)) for axis in range(3)
    ]
    params = SimpleNamespace(
        output_path=str(output / "input.h5"),
        nk=mesh_shape,
        x2c=0,
        orth="none",
        space_symm=False,
        tr_symm=False,
    )
    density = np.array(mean_field.make_rdm1(), dtype=complex)
    overlap = np.array(mean_field.get_ovlp(), dtype=complex)
    hcore = np.array(mean_field.get_hcore(), dtype=complex)
    fock = np.array(mean_field.get_fock(dm=density), dtype=complex)
    comm.save_data(
        params,
        cell,
        mean_field,
        kpts,
        np.arange(len(kpts)),
        np.ones(len(kpts)),
        len(kpts),
        np.arange(len(kpts)),
        np.zeros(len(kpts), dtype=int),
        32,
        len(kpts),
        aux.nao_nr(),
        fock[None],
        overlap[None],
        hcore[None],
        density[None],
        tools.pbc.madelung(cell, kpts),
        np.asarray(cell.atom_charges()),
        cell.aoslice_by_atom()[:, 3],
    )
    comm.store_kstruct_ops_info(params, cell, kpts, identity)
    qstruct = comm.init_q_mesh(params, aux, kpts, save_data=False)
    qscaled = cell.get_scaled_kpts(qpts)
    with h5py.File(output / "cderi.h5", "a") as f:
        for iq, q in enumerate(qstruct.kpts):
            delta = qscaled - cell.get_scaled_kpts(q)
            candidates = np.flatnonzero(
                np.max(np.abs(delta - np.rint(delta)), axis=1) < 1e-9
            )
            if len(candidates) != 1:
                raise ValueError("Cannot align captured and legacy q frames")
            c = gauges[int(candidates[0])].unwhitening
            f[f"j2c/{iq}"] = c @ c.conj().T
        f["j2c"].attrs["j2c_decomposition"] = "cholesky"
    comm.store_auxcell_kstruct_ops_info(params, aux.copy(), kpts)
    fingerprint = input_fingerprint(cell, kpts)
    with h5py.File(output / "input.h5", "a") as f:
        f["integral_symmetry/input_fingerprint"] = fingerprint
        legacy_pairs = f["symmetry/pairs/kpair_idx"][()][
            f["symmetry/pairs/kpair_irre_list"][()]
        ]
    results = {
        "mean_field_converged": bool(mean_field.converged),
        "mean_field_energy": float(mean_field.e_tot),
        "one_body_symmetry": "unreduced",
        "legacy_representatives": len(legacy_pairs),
    }
    for name, reference in (("hf", hf_path), ("correlation", correlation_path)):
        if args.hf_only and name == "correlation":
            continue
        c, a, k, q, pq, gg, v = load_reference(reference)
        if input_fingerprint(c, k) != fingerprint:
            raise ValueError("HF/correlation input identities disagree")
        geometry = PhysicalPairMap(c, a, k, q, pq, gg, set_id=str(reference))
        direct_legacy(output / f"{name}_direct", legacy_pairs, v)
        results[name] = write_archive(
            geometry,
            lambda i, j: v[i, j],
            output / f"{name}_sg",
            set_kind=name,
            finite_size_kind="none" if name == "hf" else "ewald",
            producer_revision=args.revision,
            fingerprint=fingerprint,
            chunk_size=5,
        )
        if name == "correlation":
            with h5py.File(output / "input.h5", "a") as file:
                file["integral_symmetry/correlation_gauge_id"] = results[name][
                    "auxiliary_gauge_id"
                ]
        expand_archive(
            output / f"{name}_sg",
            output / "input.h5",
            output / f"{name}_expanded",
            chunk_size=5,
        )
    (output / "preparation.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
