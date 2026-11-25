"""Independent frozen-input contractions for each diagram on a tiny fixture.

Production executables expose aggregate Sigma. This separate oracle resolves
HF direct/exchange, projected GW P, and GF2 direct/exchange before that sum.
All momenta use the original full mesh; arbitrary complex G is permitted.
"""

import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from green_mbtools.mint.integral_symmetry_io import ArchiveReader


def diagrams(hf, corr, scaled, gt, gb, density):
    nk = len(scaled)
    ns = gt.shape[1]
    nao = gt.shape[-1]

    def index(point):
        delta = scaled - point
        delta -= np.rint(delta)
        found = int(np.argmin(np.linalg.norm(delta, axis=1)))
        assert np.linalg.norm(delta[found]) < 1e-9
        return found

    upper = (
        sum(np.einsum("Qab,sba->Q", hf[k, k], density[:, k]) for k in range(nk)) / nk
    )
    hartree = np.array([np.einsum("Q,Qij->ij", upper, hf[k, k]) for k in range(nk)])
    exchange = np.zeros_like(density)
    for s in range(ns):
        for k in range(nk):
            for j in range(nk):
                v = hf[k, j]
                exchange[s, k] -= (
                    np.einsum(
                        "Qib,ba,Qja->ij", v, density[s, j], v.conj(), optimize=True
                    )
                    / nk
                )
    polarization = []
    spectra = []
    for q in scaled:
        p = np.zeros((len(gt), len(upper), len(upper)), complex)
        for k in range(nk):
            j = index(scaled[k] + q)
            v = corr[k, j]
            for s in range(ns):
                p -= (
                    np.einsum(
                        "tap,Qpm,tmn,Ran->tQR",
                        gb[:, s, k],
                        v,
                        gt[:, s, j],
                        v.conj(),
                        optimize=True,
                    )
                    / nk
                )
        p = (p + p.swapaxes(-1, -2).conj()) / 2
        v = corr[0, index(q)].reshape(len(upper), -1)
        projected = np.einsum("Qa,tQR,Rb->tab", v.conj(), p, v, optimize=True)
        rng = np.random.default_rng(47)
        rotation = np.linalg.qr(
            rng.normal(size=(len(v), len(v))) + 1j * rng.normal(size=(len(v), len(v)))
        )[0]
        rv = rotation @ v
        rp = np.einsum("AQ,tQR,BR->tAB", rotation, p, rotation.conj(), optimize=True)
        np.testing.assert_allclose(
            projected,
            np.einsum("Qa,tQR,Rb->tab", rv.conj(), rp, rv, optimize=True),
            atol=1e-11,
            rtol=1e-11,
        )
        polarization.append(projected)
        spectra.append(np.linalg.eigvalsh(p))
    direct = np.zeros((len(gt), ns, nk, nao, nao), complex)
    second_exchange = np.zeros_like(direct)
    for k0 in range(nk):
        for k1 in range(nk):
            for k2 in range(nk):
                k3 = index(scaled[k0] + scaled[k2] - scaled[k1])
                v = np.einsum(
                    "Qip,Qmk->ipmk", corr[k0, k1], corr[k2, k3], optimize=True
                )
                vx = np.einsum(
                    "Qjl,Qnq->jlnq",
                    corr[k0, k3].conj(),
                    corr[k2, k1].conj(),
                    optimize=True,
                )
                for s in range(ns):
                    for sp in range(ns):
                        direct[:, s, k0] += (
                            np.einsum(
                                "ipmk,tpq,tnm,tkl,jqnl->tij",
                                v,
                                gt[:, s, k1],
                                gb[:, sp, k2],
                                gt[:, sp, k3],
                                v.conj(),
                                optimize=True,
                            )
                            / nk**2
                        )
                        if s == sp:
                            second_exchange[:, s, k0] -= (
                                np.einsum(
                                    "ipmk,tpq,tnm,tkl,jlnq->tij",
                                    v,
                                    gt[:, s, k1],
                                    gb[:, sp, k2],
                                    gt[:, sp, k3],
                                    vx,
                                    optimize=True,
                                )
                                / nk**2
                            )
    return {
        "HF_direct": hartree,
        "HF_exchange": exchange,
        "GW_projected_P": np.array(polarization),
        "GW_P_spectrum": np.array(spectra),
        "GF2_direct": direct,
        "GF2_exchange": second_exchange,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--hf-reference", required=True, type=Path)
    parser.add_argument("--correlation-reference", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    assert not args.out.exists()
    with h5py.File(args.case / "input.h5") as file:
        scaled = file["symmetry/k/mesh_scaled"][()]
    nk = len(scaled)
    assert nk <= 8, "Independent diagram oracle is a bounded tiny-case test"
    with h5py.File(args.case / "frozen_G.h5") as file:
        raw = file["G_tau"][()]
        g = raw[..., 0] + 1j * raw[..., 1]
    times = [0, len(g) // 4, len(g) // 2, len(g) - 2, len(g) - 1]
    gt, gb = g[times], g[[len(g) - 1 - t for t in times]]
    results = []
    for source in ("direct", "sg"):
        factors = {}
        for kind, reference in (
            ("hf", args.hf_reference),
            ("correlation", args.correlation_reference),
        ):
            if source == "direct":
                with h5py.File(reference) as file:
                    factors[kind] = {
                        (i, j): file[f"factors/{i}_{j}"][()]
                        for i in range(nk)
                        for j in range(nk)
                    }
            else:
                with ArchiveReader(args.case / f"{kind}_sg") as reader:
                    factors[kind] = {
                        (i, j): reader.read(i, j) for i in range(nk) for j in range(nk)
                    }
        results.append(
            diagrams(factors["hf"], factors["correlation"], scaled, gt, gb, -g[-1])
        )
    errors = {}
    for name, reference in results[0].items():
        actual = results[1][name]
        delta = actual - reference
        np.testing.assert_allclose(
            actual, reference, atol=1e-10, rtol=1e-10, err_msg=name
        )
        assert np.linalg.norm(reference) > 1e-8, name + " must be exercised"
        errors[name] = {
            "max_absolute": float(np.max(np.abs(delta))),
            "relative_frobenius": float(
                np.linalg.norm(delta) / np.linalg.norm(reference)
            ),
            "worst_index": list(
                map(int, np.unravel_index(np.argmax(np.abs(delta)), delta.shape))
            ),
        }
    report = {
        "status": "PASS",
        "times": times,
        "arbitrary_complex_frozen_G": True,
        "errors": errors,
        "scope": "Independent diagram reference; production aggregate comparison is separate",
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
