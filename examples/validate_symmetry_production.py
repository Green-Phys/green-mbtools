"""Exercise the real producer entry point in a fresh, bounded workstation case."""

import argparse
import json
import os
import resource
import time
from pathlib import Path
import h5py
import numpy as np
from green_mbtools.mint import common_utils as comm
from green_mbtools.mint.pyscf_init import pyscf_pbc_init
from green_mbtools.mint.integral_symmetry_io import expand_archive

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", required=True, type=Path)
parser.add_argument("--nk", nargs=3, type=int, default=[3, 1, 1])
parser.add_argument(
    "--orth",
    choices=["none", "symmetric_lowdin", "lowdin", "mo", "natural"],
    default="none",
)
parser.add_argument("--space-symm", choices=["true", "false"], default="false")
args = parser.parse_args()
out = args.out.resolve()
out.mkdir(parents=True, exist_ok=False)
os.chdir(out)
params = [
    "--a",
    "0 1.75 1.75\n1.75 0 1.75\n1.75 1.75 0",
    "--atom",
    "C 0 0 0\nC .875 .875 .875",
    "--nk",
    *map(str, args.nk),
    "--basis",
    "sto3g",
    "--Nk",
    "31",
    "--space_symm",
    args.space_symm,
    "--tr_symm",
    "true",
    "--orth",
    args.orth,
    "--integral_symmetry",
    "space_group",
    "--integral_symmetry_work",
    str(out / "work"),
    "--output_path",
    str(out / "input.h5"),
    "--hf_int_path",
    str(out / "hf_sg"),
    "--int_path",
    str(out / "correlation_sg"),
    "--keep_cderi",
    "true",
]
settings = comm.init_pbc_params(params)
settings.basis = {"C": [[0, [1.0, 1.0]], [1, [0.6, 1.0]]]}
settings.auxbasis = {"C": [[0, [0.8, 1.0]], [0, [0.25, 1.0]], [1, [0.6, 1.0]]]}
started = time.perf_counter()
system = pyscf_pbc_init(settings)
system.cell.max_memory = 4000
system.cell.precision = 1e-10
system.cell.mesh = [31] * 3
system.cell.build(False, False)
system.mean_field_input()
assert (out / "cderi.h5").exists(), "Complete SCF cache must be preserved"
for kind in ("hf", "correlation"):
    expand_archive(
        out / f"{kind}_sg", out / "input.h5", out / f"{kind}_expanded", chunk_size=5
    )
with h5py.File(out / "input.h5") as f:
    q = f["symmetry/q/k_sym_transform_p0"][()]
    assert np.isfinite(q).all()
    inq = int(f["symmetry/q/inq"][()])
    ink = int(f["symmetry/k/ink"][()])
result = {
    "status": "PASS",
    "orth": args.orth,
    "nk": args.nk,
    "ink": ink,
    "inq": inq,
    "wall_seconds": time.perf_counter() - started,
    "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    "parameters": params,
    "complete_scf_cache_bytes": (out / "cderi.h5").stat().st_size,
}
(out / "production-validation.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
