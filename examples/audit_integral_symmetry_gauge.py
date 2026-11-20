"""Read-only diagnosis of an actual saved complete CCGDF gauge."""

import argparse
import json
import h5py
import numpy as np
from scipy.linalg import solve_triangular
from pyscf.pbc import gto
from pyscf.pbc.lib import kpts as libkpts
from green_mbtools.mint.integral_symmetry import (
    CrystalOperation,
    IntegerMesh,
    build_pair_orbits,
    validated_cell_operations,
)
from green_mbtools.mint.integral_symmetry_transform import residual, reconstruct_factor
from green_mbtools.mint.symmetry_utils import get_representation
from pyscf.pbc.df.gdf_builder import _CCGDFBuilder

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("reference")
args = parser.parse_args()
with h5py.File(args.reference) as f:
    cell = gto.loads(f["Cell"][()].decode())
    auxcell = gto.loads(f["AuxCell"][()].decode())
    kpts, qpts = f["kpts"][()], f["qpts"][()]
    nk = len(kpts)
    factors = {(i, j): f[f"factors/{i}_{j}"][()] for i in range(nk) for j in range(nk)}
    pair_q = {
        (i, j): int(f[f"pair_q/{i}_{j}"][()]) for i in range(nk) for j in range(nk)
    }
    gauges = {i: f[f"captured_C/{i}"][()] for i in range(len(qpts))}
mesh = IntegerMesh.from_scaled(cell.get_scaled_kpts(kpts))
kstruct = libkpts.make_kpts(
    cell, kpts, space_group_symmetry=True, time_reversal_symmetry=False
)
qstruct = libkpts.make_kpts(
    auxcell, qpts, space_group_symmetry=True, time_reversal_symmetry=False
)
orbits = build_pair_orbits(
    mesh, validated_cell_operations(cell, kstruct), time_reversal=True, exchange=True
)
op_ids = {CrystalOperation(op.rot, op.trans): i for i, op in enumerate(kstruct.ops)}
aux_ids = {CrystalOperation(op.rot, op.trans): i for i, op in enumerate(qstruct.ops)}
qmesh = IntegerMesh.from_scaled(auxcell.get_scaled_kpts(qpts))
q_ids = {tuple(row % qmesh.denominator): i for i, row in enumerate(qmesh.coordinates)}
builder = _CCGDFBuilder(cell, auxcell, kpts).build()
metrics = builder.get_2c2e(qpts)
print(
    json.dumps(
        {
            "captured_vs_rebuilt_metrics": [
                residual(gauges[i] @ gauges[i].conj().T, m)
                for i, m in enumerate(metrics)
            ]
        }
    )
)
failed = 0
for i in range(nk):
    for j in range(nk):
        rep = tuple(
            map(int, orbits.representatives[orbits.pair_to_representative[i, j]])
        )
        op = orbits.operations[orbits.pair_operation[i, j]]
        mapped, _ = mesh.action(op)
        si, sj = map(int, mapped[list(rep)])
        tr, ex = bool(orbits.time_reversal[i, j]), bool(orbits.exchange[i, j])
        conjugate = tr ^ ex
        ui = get_representation(si, op_ids[op], cell, kstruct, tr_phase=False)
        uj = get_representation(sj, op_ids[op], cell, kstruct, tr_phase=False)
        qscaled = cell.get_scaled_kpts(kpts[sj] - kpts[si])
        iq = q_ids[
            tuple(np.rint(qscaled * qmesh.denominator).astype(int) % qmesh.denominator)
        ]
        raw = get_representation(iq, aux_ids[op], auxcell, qstruct, tr_phase=False)
        source, target = gauges[pair_q[rep]], gauges[pair_q[i, j]]
        if conjugate:
            source = source.conj()
            raw = raw.conj()
        left, right = (uj, ui) if ex else (ui, uj)
        if tr:
            left, right = left.conj(), right.conj()
        metric = target @ target.conj().T
        if np.allclose(
            raw @ source @ source.conj().T @ raw.conj().T, metric, atol=1e-8, rtol=1e-8
        ):
            continue
        variants = {}
        for label, a in {
            "raw": raw,
            "conjugate": raw.conj(),
            "transpose": raw.T,
            "adjoint": raw.conj().T,
        }.items():
            d = solve_triangular(target, a @ source, lower=True)
            recon = reconstruct_factor(
                factors[rep], d, left, right, conjugate=conjugate, transpose=ex
            )
            variants[label] = {
                "metric": residual(a @ source @ source.conj().T @ a.conj().T, metric),
                "factor": residual(recon, factors[i, j]),
            }
        print(
            json.dumps(
                {
                    "pair": [i, j],
                    "rep": rep,
                    "op": int(orbits.pair_operation[i, j]),
                    "rotation": op.rotation,
                    "translation": list(map(float, op.translation)),
                    "TR": tr,
                    "EX": ex,
                    "q_source": cell.get_scaled_kpts(qpts[pair_q[rep]]).tolist(),
                    "q_target": cell.get_scaled_kpts(qpts[pair_q[i, j]]).tolist(),
                    "q_spatial": qscaled.tolist(),
                    "variants": variants,
                }
            )
        )
        failed += 1
        if failed >= 5:
            raise SystemExit(0)
