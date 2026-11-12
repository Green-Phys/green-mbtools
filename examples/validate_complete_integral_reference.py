"""Validate complete physical factors, cross-pair ERIs, stored basis, and group actions.

Run on GREEN_workstation; reads a saved complete reference and writes a fresh
JSON report. It never changes reference integrals.
"""

import argparse
import json
from pathlib import Path
import time

import h5py
import numpy as np
from pyscf.pbc import gto

from green_mbtools.mint.integral_symmetry_geometry import PhysicalPairMap
from green_mbtools.mint.integral_symmetry_transform import CholeskyGauge, residual


def load_reference(path):
    with h5py.File(path) as f:
        cell, auxcell = (
            gto.loads(f[name][()].decode()) for name in ("Cell", "AuxCell")
        )
        kpts, qpts = f["kpts"][()], f["qpts"][()]
        nk = len(kpts)
        factors = {
            (i, j): f[f"factors/{i}_{j}"][()] for i in range(nk) for j in range(nk)
        }
        pair_q = {
            (i, j): int(f[f"pair_q/{i}_{j}"][()]) for i in range(nk) for j in range(nk)
        }
        kernel = f.attrs.get("kernel_id", "ordinary-Coulomb")
        gauges = {
            i: CholeskyGauge(f[f"captured_C/{i}"][()], f"{path}:q{i}", kernel)
            for i in range(len(qpts))
        }
    return cell, auxcell, kpts, qpts, pair_q, gauges, factors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("--out", required=True)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--rtol", type=float, default=1e-10)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise ValueError("Validation report already exists")
    started = time.perf_counter()
    cell, auxcell, kpts, qpts, pair_q, gauges, factors = load_reference(args.reference)
    geometry = PhysicalPairMap(
        cell,
        auxcell,
        kpts,
        qpts,
        pair_q,
        gauges,
        set_id=str(Path(args.reference).resolve()),
        atol=args.atol,
        rtol=args.rtol,
    )
    nk = len(kpts)
    overlap = cell.pbc_intor("int1e_ovlp", kpts=kpts)
    x, inverse = [], []
    for s in overlap:
        eig, vectors = np.linalg.eigh(s)
        if eig[0] <= 1e-9:
            raise ValueError("Orthogonal fixture has a truncated overlap space")
        x.append((vectors / np.sqrt(eig)) @ vectors.conj().T)
        inverse.append((vectors * np.sqrt(eig)) @ vectors.conj().T)
    ortho = {
        pair: np.einsum("im,Qmn,jn->Qij", x[pair[0]], v, x[pair[1]].conj())
        for pair, v in factors.items()
    }
    reconstructed, maxima = (
        {},
        {
            key: {"max_absolute": -1}
            for key in (
                "AO",
                "orthogonal",
                "slice",
                "group",
                "inverse",
                "composition",
                "cross_pair_ERI",
            )
        },
    )
    checks = {key: 0 for key in maxima}
    failures = []

    def compare(kind, actual, expected, context):
        measure = residual(actual, expected)
        if measure["max_absolute"] > maxima[kind]["max_absolute"]:
            maxima[kind] = dict(measure, context=context)
        checks[kind] += 1
        if not np.allclose(actual, expected, atol=args.atol, rtol=args.rtol):
            failures.append(dict(kind=kind, context=context, **measure))

    for i in range(nk):
        for j in range(nk):
            rep, maps = geometry.pair(i, j)
            v = geometry.reconstruct(factors[rep], maps)
            reconstructed[i, j] = v
            compare("AO", v, factors[i, j], [i, j])
            compare(
                "slice",
                geometry.reconstruct(factors[rep], maps, output_slice=(2, 3)),
                factors[i, j][2:5],
                [i, j],
            )
            vo = geometry.reconstruct(
                ortho[rep],
                maps,
                source_x_inverse=(inverse[rep[0]], inverse[rep[1]]),
                target_x=(x[i], x[j]),
            )
            compare("orthogonal", vo, ortho[i, j], [i, j])
    o = geometry.orbits
    group_samples = sorted(
        {(0, k) for k in range(nk)}
        | {(k, k) for k in range(nk)}
        | {(k, (k + 1) % nk) for k in range(nk)}
    )
    for opid, operation in enumerate(o.operations):
        for pair in group_samples:
            target, maps = geometry.spatial(pair, operation)
            v = geometry.reconstruct(factors[pair], maps)
            compare("group", v, factors[target], [opid, *pair])
            back, back_maps = geometry.spatial(target, o.operations[o.inverses[opid]])
            compare(
                "inverse",
                geometry.reconstruct(v, back_maps),
                factors[back],
                [opid, *pair],
            )
            # Test another actual group element, including lattice-translation carry.
            second_id = (opid + 1) % len(o.operations)
            target2, maps2 = geometry.spatial(target, o.operations[second_id])
            sequential = geometry.reconstruct(v, maps2)
            combined_id = o.multiplication[second_id, opid]
            combined_target, combined_maps = geometry.spatial(
                pair, o.operations[combined_id]
            )
            if combined_target != target2:
                raise ValueError("Group composition has incompatible momentum action")
            compare(
                "composition",
                sequential,
                geometry.reconstruct(factors[pair], combined_maps),
                [opid, second_id, *pair],
            )
    wrong_map_detected = False
    for iq in range(len(qpts)):
        pairs = sorted(pair for pair in factors if pair_q[pair] == iq)
        direct = np.stack(
            [factors[p].reshape(auxcell.nao_nr(), -1) for p in pairs], axis=1
        ).reshape(auxcell.nao_nr(), -1)
        rebuilt = np.stack(
            [reconstructed[p].reshape(auxcell.nao_nr(), -1) for p in pairs], axis=1
        ).reshape(auxcell.nao_nr(), -1)
        # All orbital cross-products at a fixed q, including diagonal/offdiagonal
        # pairs. Bound temporary ERI storage by comparing rows in blocks.
        for row in range(0, direct.shape[1], 128):
            compare(
                "cross_pair_ERI",
                rebuilt[:, row : row + 128].conj().T @ rebuilt,
                direct[:, row : row + 128].conj().T @ direct,
                [iq, row],
            )
        if len(pairs) > 1:
            cross = factors[pairs[0]].reshape(auxcell.nao_nr(), -1).conj().T @ factors[
                pairs[1]
            ].reshape(auxcell.nao_nr(), -1)
            wrong = np.exp(0.3j) * cross
            wrong_map_detected |= not np.allclose(
                wrong, cross, atol=args.atol, rtol=args.rtol
            )
    if not wrong_map_detected:
        failures.append(
            {
                "kind": "negative_cross_pair_test",
                "reason": "Wrong pair-dependent frame was not detected",
            }
        )
    summary = o.summary()
    summary.update(
        status="PASS" if not failures else "FAIL",
        tensor_reconstruction_validated=not failures,
        reference=args.reference,
        checks=checks,
        maxima=maxima,
        failures=failures,
        wrong_pair_dependent_map_detected=wrong_map_detected,
        nonzero_translation_operations=sum(any(op.translation) for op in o.operations),
        cache_peak_bytes=geometry.cache.peak,
        cache_budget_bytes=geometry.cache.budget,
        elapsed_seconds=time.perf_counter() - started,
        atol=args.atol,
        rtol=args.rtol,
    )
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit("Complete physical reconstruction validation failed")


if __name__ == "__main__":
    main()
