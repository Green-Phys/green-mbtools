"""Count Si pair orbits without a rebuild; diagnose existing metric rank only."""

import argparse, json, time, resource
from pathlib import Path
import h5py, numpy as np
from pyscf.pbc import gto
from green_mbtools.mint.integral_symmetry import (
    IntegerMesh,
    integral_kstruct,
    validated_cell_operations,
    build_pair_orbits,
)

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--input", required=True, type=Path)
p.add_argument("--cderi", required=True, type=Path)
p.add_argument("--out", required=True, type=Path)
args = p.parse_args()
assert not args.out.exists()
with h5py.File(args.input) as f:
    cell = gto.loads(f["Cell"][()].decode())
    nao = int(f["params/nao"][()])
    naux = int(f["params/NQ"][()])
counts = []
for n in [1, 2, 3, 6]:
    started = time.perf_counter()
    kpts = cell.make_kpts([n] * 3)
    mesh = IntegerMesh.from_scaled(cell.get_scaled_kpts(kpts))
    orbits = build_pair_orbits(
        mesh,
        validated_cell_operations(cell, integral_kstruct(cell, kpts)),
        time_reversal=True,
        exchange=True,
    )
    # The unchanged legacy pair relation uses only TR and exchange.
    legacy = build_pair_orbits(
        mesh, [orbits.operations[0]], time_reversal=True, exchange=True
    )
    nk = len(kpts)
    size = naux * nao * nao * 16
    record = {
        "nk": [n] * 3,
        "kpoints": nk,
        "full_ordered_pairs": nk * nk,
        "legacy_representatives": len(legacy.representatives),
        "sg_representatives": len(orbits.representatives),
        "operations": len(orbits.operations),
        "nao": nao,
        "NQ": naux,
        "estimated_valid_factor_bytes": len(orbits.representatives) * size,
        "estimated_full_ordered_bytes": nk * nk * size,
        "estimated_legacy_valid_bytes": len(legacy.representatives) * size,
        "estimated_captured_gauge_bytes": nk * naux * naux * 16,
        "map_seconds": time.perf_counter() - started,
        "scope": "Actual map counts; byte estimates only; no Si integral rebuild",
    }
    counts.append(record)
    print(json.dumps(record), flush=True)
metrics = []
with h5py.File(args.cderi) as f:
    for key in f["j2c/q_ibz2bz"][()][:3]:
        metric = f[f"j2c/{key}"][()]
        e = np.linalg.eigvalsh(metric)
        condition = float(np.linalg.cond(metric))
        full_rank = np.count_nonzero(e > 1e-9) == len(e)
        record = {
            "q_index": int(key),
            "min_eigenvalue": float(e[0]),
            "max_eigenvalue": float(e[-1]),
            "eigenvalues_above_1e-9": int(np.count_nonzero(e > 1e-9)),
            "raw_dimension": len(e),
            "condition_number": condition,
            "eigendecomposition_rank_at_1e-9": "full" if full_rank else "truncated",
            "captured_cholesky_support": "NOT DETERMINED from a raw metric",
            "scope": "Read-only rank diagnostic, not a captured whitening frame or new producer run",
        }
        metrics.append(record)
        print(json.dumps(record), flush=True)
args.out.write_text(
    json.dumps(
        {
            "pair_counts": counts,
            "existing_metric_diagnostics": metrics,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
        indent=2,
    )
    + "\n"
)
