"""Bounded CCGDF covariance experiment with the builder's captured Cholesky gauge.

Run only on GREEN_workstation in a fresh /data case directory. This builds the
complete ordered-pair reference, saves every factor and metric factor, and
compares all orbit reconstructions. It does not generate compressed archives.
"""

import argparse
import json
from pathlib import Path
import time

import h5py
import numpy as np
from pyscf.pbc import df, gto
from pyscf.pbc.df import df as gdf
from green_mbtools.mint.integral_symmetry_compat import require_ccgdf_compatibility

from green_mbtools.mint.integral_symmetry import (
    CrystalOperation,
    IntegerMesh,
    build_pair_orbits,
    validated_cell_operations,
    integral_kstruct,
)
from green_mbtools.mint.integral_symmetry_transform import (
    CholeskyGauge,
    auxiliary_transform,
    reconstruct_factor,
    residual,
    validate_overlap_covariance,
)
from green_mbtools.mint.symmetry_utils import get_representation
from green_mbtools.mint.integral_symmetry_gauge import build_captured_legacy_ewald


class CapturingCCGDFBuilder(require_ccgdf_compatibility()):
    """Capture actual factors, including the builder's imposed q/-q relation."""

    def gen_uniq_kpts_groups(self, *args, **kwargs):
        for q, pairs, decomposition in super().gen_uniq_kpts_groups(*args, **kwargs):
            factor, negative, tag = decomposition
            if (
                tag != "CD"
                or negative is not None
                or factor.shape != (self.auxcell.nao_nr(),) * 2
            ):
                raise ValueError(
                    f"Unsupported metric decomposition at q={q}: tag={tag}, "
                    f"shape={factor.shape}, expected={self.auxcell.nao_nr()}, "
                    f"negative={None if negative is None else negative.shape}"
                )
            factor = np.array(factor, dtype=np.complex128, copy=True)
            self.captured.append((np.array(q), np.array(pairs), factor))
            yield q, pairs, decomposition

    def decompose_j2c(self, metric):
        values = np.linalg.eigvalsh(metric)
        result = super().decompose_j2c(metric)
        print(
            json.dumps(
                {
                    "metric_min_eigenvalue": float(values[0]),
                    "metric_max_eigenvalue": float(values[-1]),
                    "decomposition": result[2],
                    "retained_rank": len(result[0]),
                }
            )
        )
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--mesh",
        type=int,
        default=31,
        help="Finite PW grid for this convergence experiment",
    )
    parser.add_argument(
        "--precision",
        type=float,
        default=1e-10,
        help="Real-space integral cutoff for the fixture",
    )
    parser.add_argument("--nk", type=int, nargs=3, default=(3, 1, 1))
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument(
        "--aux-tight",
        action="store_true",
        help="Use a more localized full-rank auxiliary fixture",
    )
    parser.add_argument(
        "--ewald",
        action="store_true",
        help="Build a separately captured Ewald-corrected gauge",
    )
    parser.add_argument("--system", choices=["carbon", "silicon"], default="carbon")
    parser.add_argument(
        "--supercell",
        type=int,
        default=1,
        help="Bounded repetition along the first lattice vector",
    )
    parser.add_argument("--shift", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    cell = gto.Cell()
    cell.a = 3.5 * np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    cell.atom = [["C", [0, 0, 0]], ["C", [0.875, 0.875, 0.875]]]
    cell.unit = "Angstrom"
    cell.basis = {"C": [[0, [1.0, 1.0]], [1, [0.6, 1.0]]]}
    if args.system == "silicon":
        cell.a *= 5.431 / 3.5
        cell.atom = [["Si", [0, 0, 0]], ["Si", [1.35775] * 3]]
        cell.basis = "gth-dzvp-molopt-sr"
        cell.pseudo = "gth-pbe"
    if not 1 <= args.supercell <= 2:
        raise ValueError("Fixture supercell is bounded to two repetitions")
    if args.supercell == 2:
        cell.atom = list(cell.atom) + [
            [symbol, (np.asarray(position) + cell.a[0]).tolist()]
            for symbol, position in cell.atom
        ]
        cell.a[0] *= 2
    cell.precision = args.precision
    cell.verbose = 4
    cell.mesh = [args.mesh] * 3
    cell.max_memory = 4000
    cell.space_group_symmetry = True
    cell.symmorphic = False
    cell.build()
    kpts = cell.make_kpts(args.nk) + cell.get_abs_kpts(np.asarray(args.shift))
    kstruct = integral_kstruct(cell, kpts)
    mesh = IntegerMesh.from_scaled(cell.get_scaled_kpts(kpts))
    operations = validated_cell_operations(cell, kstruct)
    orbits = build_pair_orbits(mesh, operations, time_reversal=True, exchange=True)
    auxiliary_basis = (
        {"C": [[0, [2.0, 1.0]], [1, [1.2, 1.0]]]}
        if args.aux_tight
        else {"C": [[0, [0.8, 1.0]], [0, [0.25, 1.0]], [1, [0.6, 1.0]]]}
    )
    if args.system == "silicon":
        auxiliary_basis = {"Si": auxiliary_basis["C"]}
    # CCGDF's compensation/fuse contract requires normalized multipole charges.
    # An ordinary incore.make_auxcell gives inconsistent compensated metrics.
    auxcell = gdf.make_modrho_basis(cell, auxiliary_basis)
    builder = CapturingCCGDFBuilder(cell, auxcell, kpts)
    builder.mesh = np.array(cell.mesh)
    builder.j2c_eig_always = False
    builder.captured = []
    pairs = np.array([(ki, kj) for ki in kpts for kj in kpts])
    started = time.perf_counter()
    builder.make_j3c(str(out / "cderi.h5"), kptij_lst=pairs)
    provider = df.GDF(cell, kpts)
    provider.auxcell = auxcell
    provider._cderi = str(out / "cderi.h5")
    nk, nao = len(kpts), cell.nao_nr()
    corrected_provider = q0_gauge = None
    if args.ewald:
        corrected_provider, q0_gauge = build_captured_legacy_ewald(
            cell,
            kpts,
            auxiliary_basis,
            [(i, i) for i in range(nk)],
            out / "legacy-ewald-cderi.h5",
            cell.mesh,
        )
    build_seconds = time.perf_counter() - started
    factors = {}
    gauges = {}
    pair_q = {}
    qpts = []
    for iq, (q, group_pairs, c) in enumerate(builder.captured):
        qpts.append(q)
        gauges[iq] = (
            q0_gauge
            if args.ewald and np.linalg.norm(q) < 1e-9
            else CholeskyGauge(
                c,
                f"captured-ccgdf-q{iq}",
                "ewald-Coulomb" if args.ewald else "ordinary-Coulomb",
            )
        )
        for pair in group_pairs:
            pair_q[divmod(int(pair), nk)] = iq
    if len(pair_q) != nk * nk:
        raise ValueError("Captured gauges do not cover every ordered reference pair")
    for i in range(nk):
        for j in range(nk):
            blocks = []
            reader = corrected_provider if args.ewald and i == j else provider
            for real, imag, sign in reader.sr_loop((kpts[i], kpts[j]), compact=False):
                if sign != 1:
                    raise ValueError("Negative-metric sectors are unsupported")
                blocks.append(real + 1j * imag)
            factors[i, j] = np.concatenate(blocks).reshape(-1, nao, nao)
    qpts = np.array(qpts)
    qmesh = IntegerMesh.from_scaled(auxcell.get_scaled_kpts(qpts))
    auxstruct = integral_kstruct(auxcell, qpts)
    op_ids = {CrystalOperation(op.rot, op.trans): i for i, op in enumerate(kstruct.ops)}
    aux_ids = {
        CrystalOperation(op.rot, op.trans): i for i, op in enumerate(auxstruct.ops)
    }
    q_ids = {
        tuple(row % qmesh.denominator): i for i, row in enumerate(qmesh.coordinates)
    }
    overlap = cell.pbc_intor("int1e_ovlp", kpts=kpts)
    # Preserve the complete reference/gauges even if a covariance check fails.
    with h5py.File(out / "complete-reference.h5", "w") as archive:
        archive.attrs["kernel_id"] = (
            "ewald-Coulomb" if args.ewald else "ordinary-Coulomb"
        )
        archive.attrs["set_kind"] = "correlation" if args.ewald else "hf"
        archive.attrs["corrected_producer"] = (
            "green_igen.df._make_j3c" if args.ewald else "none"
        )
        archive["Cell"] = cell.dumps()
        archive["AuxCell"] = auxcell.dumps()
        archive["kpts"] = kpts
        archive["qpts"] = qpts
        for (i, j), factor in factors.items():
            archive[f"factors/{i}_{j}"] = factor
            archive[f"pair_q/{i}_{j}"] = pair_q[i, j]
        for iq, gauge in gauges.items():
            archive[f"captured_C/{iq}"] = gauge.unwhitening
    results = []
    maximum = {"max_absolute": -1}
    for i in range(nk):
        for j in range(nk):
            rep = tuple(
                map(int, orbits.representatives[orbits.pair_to_representative[i, j]])
            )
            operation = orbits.operations[orbits.pair_operation[i, j]]
            spatial_points, _ = mesh.action(operation)
            spatial_i, spatial_j = map(int, spatial_points[list(rep)])
            tr, ex = bool(orbits.time_reversal[i, j]), bool(orbits.exchange[i, j])
            conjugate = tr ^ ex
            ui = get_representation(
                spatial_i, op_ids[operation], cell, kstruct, tr_phase=False
            )
            uj = get_representation(
                spatial_j, op_ids[operation], cell, kstruct, tr_phase=False
            )
            validate_overlap_covariance(
                overlap[rep[0]], overlap[spatial_i], ui, atol=args.atol, rtol=args.rtol
            )
            validate_overlap_covariance(
                overlap[rep[1]], overlap[spatial_j], uj, atol=args.atol, rtol=args.rtol
            )
            q_spatial = cell.get_scaled_kpts(kpts[spatial_j] - kpts[spatial_i])
            q_key = tuple(
                np.rint(q_spatial * qmesh.denominator).astype(np.int64)
                % qmesh.denominator
            )
            aq = get_representation(
                q_ids[q_key], aux_ids[operation], auxcell, auxstruct, tr_phase=False
            )
            if conjugate:
                aq = aq.conj()
            d = auxiliary_transform(
                gauges[pair_q[rep]],
                gauges[pair_q[i, j]],
                aq,
                conjugate=conjugate,
                atol=args.atol,
                rtol=args.rtol,
            )
            left, right = (uj, ui) if ex else (ui, uj)
            if tr:
                left, right = left.conj(), right.conj()
            reconstructed = reconstruct_factor(
                factors[rep], d, left, right, conjugate=conjugate, transpose=ex
            )
            measure = residual(reconstructed, factors[i, j])
            measure.update(
                pair=[i, j],
                representative=list(rep),
                operation=int(orbits.pair_operation[i, j]),
                time_reversal=tr,
                exchange=ex,
                pass_entries=bool(
                    np.allclose(
                        reconstructed, factors[i, j], atol=args.atol, rtol=args.rtol
                    )
                ),
            )
            results.append(measure)
            if measure["max_absolute"] > maximum["max_absolute"]:
                maximum = measure
    summary = orbits.summary()
    summary.update(
        build_seconds=build_seconds,
        nao=nao,
        naux=auxcell.nao_nr(),
        plane_wave_mesh=args.mesh,
        kernel_id="ewald-Coulomb" if args.ewald else "ordinary-Coulomb",
        physical_fixture=f"{args.system} diamond; supercell={args.supercell}",
        backend="CCGDF",
        maximum=maximum,
        all_pairs_pass=all(r["pass_entries"] for r in results),
        results=results,
        atol=args.atol,
        rtol=args.rtol,
        status="PASS" if all(r["pass_entries"] for r in results) else "FAIL",
    )
    (out / "validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
    if not summary["all_pairs_pass"]:
        raise SystemExit(
            "Directly generated factors do not satisfy the reconstruction contract"
        )


if __name__ == "__main__":
    main()
