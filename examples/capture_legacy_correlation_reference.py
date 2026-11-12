"""Fresh complete corrected reference: ordinary factors plus actual legacy q=0."""

import argparse, json
from pathlib import Path
import h5py, numpy as np
from pyscf.pbc import gto
from green_mbtools.mint.integral_symmetry_gauge import build_captured_legacy_ewald

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--hf", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
args = p.parse_args()
args.out.mkdir(parents=True, exist_ok=False)
with h5py.File(args.hf) as source:
    cell, aux = (gto.loads(source[name][()].decode()) for name in ["Cell", "AuxCell"])
    cell.max_memory = 4000
    kpts = source["kpts"][()]
    qpts = source["qpts"][()]
    nk = len(kpts)
    q0 = np.flatnonzero(np.linalg.norm(qpts, axis=1) < 1e-9)
    assert len(q0) == 1
    q0 = int(q0[0])
    reader, gauge = build_captured_legacy_ewald(
        cell,
        kpts,
        aux._basis,
        [(i, i) for i in range(nk)],
        args.out / "legacy-ewald-cderi.h5",
        cell.mesh,
    )
    with h5py.File(args.out / "complete-reference.h5", "w") as f:
        f.attrs.update(
            kernel_id="ewald-Coulomb",
            set_kind="correlation",
            corrected_producer="green_igen.df._make_j3c",
            actual_whitening_captured=True,
        )
        for name in ["Cell", "AuxCell", "kpts", "qpts"]:
            f[name] = source[name][()]
        for iq in range(len(qpts)):
            f[f"captured_C/{iq}"] = (
                gauge.unwhitening if iq == q0 else source[f"captured_C/{iq}"][()]
            )
        for i in range(nk):
            for j in range(nk):
                value = source[f"factors/{i}_{j}"][()]
                if i == j:
                    blocks = []
                    for re, im, sign in reader.sr_loop(
                        (kpts[i], kpts[j]), compact=False
                    ):
                        assert sign == 1
                        blocks.append(re + 1j * im)
                    value = np.concatenate(blocks).reshape(
                        aux.nao_nr(), cell.nao_nr(), cell.nao_nr()
                    )
                f[f"factors/{i}_{j}"] = value
                f[f"pair_q/{i}_{j}"] = source[f"pair_q/{i}_{j}"][()]
        original = source["factors/0_0"][()].reshape(aux.nao_nr(), -1)
        corrected = f["factors/0_0"][()].reshape(aux.nao_nr(), -1)
        change = float(
            np.linalg.norm(
                corrected.T @ corrected.conj() - original.T @ original.conj()
            )
        )
        assert change > 1e-5, (
            "The legacy corrected sector must actually change the interaction"
        )
    report = {
        "status": "PASS",
        "full_pairs": nk * nk,
        "corrected_diagonal_pairs": nk,
        "q0": q0,
        "q0_ERI_change_norm": change,
        "corrected_producer": "green_igen.df._make_j3c",
        "actual_whitening_captured": True,
    }
    (args.out / "capture.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
