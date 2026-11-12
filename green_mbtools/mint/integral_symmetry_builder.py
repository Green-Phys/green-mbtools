"""Representative-only CCGDF producer with captured, disk-backed metric gauges.

The full SCF DF cache is independent and is never opened for writing here.
Only representative three-center pairs are supplied to PySCF's outcore builder.
All q metrics remain necessary; each is computed/factorized by that builder's
actual routines, stored on disk and loaded through a bounded validation cache.
"""

from pathlib import Path
from types import SimpleNamespace
import json
import time

import h5py
import numpy as np
from pyscf.pbc import df
from pyscf.pbc.df.gdf_builder import _CCGDFBuilder
from pyscf.pbc.lib.kpts_helper import kk_adapted_iter

from .integral_symmetry import (
    IntegerMesh,
    build_pair_orbits,
    integral_kstruct,
    validated_cell_operations,
)
from .integral_symmetry_geometry import PhysicalPairMap
from .integral_symmetry_transform import CholeskyGauge
from .integral_symmetry_gauge import CapturedGauges, build_captured_legacy_ewald
from .integral_symmetry_io import write_archive


class RepresentativeCCGDFBuilder(_CCGDFBuilder):
    def weighted_coulG(self, kpt, exx, mesh, omega=None):
        return super().weighted_coulG(
            kpt, "ewald" if self.corrected else exx, mesh, omega
        )

    def gen_uniq_kpts_groups(self, j_only, h5swap, kk_idx=None):
        if kk_idx is None and len(self.kpts) == 1:
            kk_idx = np.array([0], dtype=np.int32)
        if (j_only and len(self.kpts) > 1) or kk_idx is None:
            raise ValueError(
                "Representative build requires an explicit ordered-pair subset"
            )
        nk = len(self.kpts)
        selected = set(map(int, kk_idx))
        # Explicit full kk_idx disables PySCF's independent q/-q gauge shortcut,
        # matching the complete captured reference contract exactly.
        groups = list(
            kk_adapted_iter(
                self.cell, self.kpts, np.arange(nk * nk, dtype=np.int32), False
            )
        )
        # Reuse the real-space metric pass within a bounded q batch. The
        # conservative estimate includes compensation functions (3*naux).
        batch = max(1, min(32, 64 * 1024**2 // (16 * (3 * self.auxcell.nao_nr()) ** 2)))
        with h5py.File(self.gauge_file, "w") as file:
            for start in range(0, len(groups), batch):
                started = time.perf_counter()
                metrics = self.get_2c2e(
                    np.asarray([group[0] for group in groups[start : start + batch]])
                )
                self.metric_seconds += time.perf_counter() - started
                for offset, (q, ki, kj, self_conjugate) in enumerate(
                    groups[start : start + batch]
                ):
                    iq = start + offset
                    started = time.perf_counter()
                    metric = metrics[offset].real if self_conjugate else metrics[offset]
                    factor, negative, tag = self.decompose_j2c(metric)
                    if tag != "CD" or negative is not None:
                        raise ValueError(
                            "Initial SG producer requires positive full-rank Cholesky metrics"
                        )
                    gauge = CholeskyGauge(factor, f"captured-q{iq}", self.kernel_id)
                    file[f"C/{iq}"] = gauge.unwhitening
                    self.metric_seconds += time.perf_counter() - started
                    self.qpts.append(np.array(q))
                    pairs = np.asarray(ki) * nk + np.asarray(kj)
                    for pair in pairs:
                        self.pair_q[divmod(int(pair), nk)] = iq
                    retained = np.array(
                        [pair for pair in pairs if int(pair) in selected],
                        dtype=np.int32,
                    )
                    if len(retained):
                        self.three_center_pairs += len(retained)
                        yield q, retained, (factor, negative, tag)


def build_representative_archive(
    cell,
    kpts,
    auxbasis,
    output,
    work,
    *,
    corrected=False,
    stored_x=None,
    producer_revision,
    chunk_size=32,
    reference_get_factor=None,
    mesh=None,
):
    """Build a fresh ordinary or Ewald set without replacing a complete cache."""
    output, work = Path(output).resolve(), Path(work).resolve()
    if output.exists() or work.exists():
        raise ValueError("SG output and work directories must be fresh")
    if cell.dimension != 3 or cell.cart or cell.omega != 0:
        raise ValueError(
            "Initial SG producer supports scalar spherical ordinary 3D cells"
        )
    kpts = np.asarray(kpts)
    if stored_x is not None and np.asarray(stored_x).shape != (
        len(kpts),
        cell.nao_nr(),
        cell.nao_nr(),
    ):
        raise ValueError("Initial SG producer requires a square retained orbital basis")
    kstruct = integral_kstruct(cell, kpts)
    orbits = build_pair_orbits(
        IntegerMesh.from_scaled(cell.get_scaled_kpts(kpts)),
        validated_cell_operations(cell, kstruct),
        time_reversal=True,
        exchange=True,
    )
    auxcell = df.df.make_modrho_basis(cell, auxbasis)
    work.mkdir(parents=True)
    builder = RepresentativeCCGDFBuilder(cell, auxcell, kpts)
    builder.corrected = (
        False  # The default corrected sector is GREEN's legacy routine below.
    )
    builder.kernel_id = "ewald-Coulomb" if corrected else "ordinary-Coulomb"
    builder.gauge_file = str(work / "captured-gauges.h5")
    builder.qpts, builder.pair_q = [], {}
    builder.metric_seconds, builder.three_center_pairs = 0.0, 0
    builder.j2c_eig_always = False
    if mesh is not None:
        builder.mesh = np.asarray(mesh)
    diagonal = [(int(i), int(j)) for i, j in orbits.representatives if i == j]
    selected = [
        (int(i), int(j)) for i, j in orbits.representatives if not corrected or i != j
    ]
    pairs = np.array([(kpts[i], kpts[j]) for i, j in selected])
    started = time.perf_counter()
    if selected:
        builder.make_j3c(
            str(work / "representatives-cderi.h5"), kptij_lst=pairs, aosym="s1"
        )
    corrected_provider = None
    corrected_seconds = 0.0
    if corrected:
        corrected_started = time.perf_counter()
        corrected_provider, q0_gauge = build_captured_legacy_ewald(
            cell, kpts, auxbasis, diagonal, work / "legacy-ewald-cderi.h5", mesh
        )
        corrected_seconds = time.perf_counter() - corrected_started
        if not selected:
            builder.qpts = [np.zeros(3)]
            builder.pair_q = {(0, 0): 0}
        q0 = [i for i, q in enumerate(builder.qpts) if np.linalg.norm(q) < 1e-9]
        if len(q0) != 1:
            raise ValueError("Captured mesh must contain one q=0 frame")
        with h5py.File(builder.gauge_file, "a") as file:
            key = f"C/{q0[0]}"
            if key in file:
                del file[key]
            file[key] = q0_gauge.unwhitening
    build_seconds = time.perf_counter() - started
    if builder.three_center_pairs + (len(diagonal) if corrected else 0) != len(
        orbits.representatives
    ):
        raise ValueError(
            "Builder did not restrict three-center production to representatives"
        )
    gauges = CapturedGauges(builder.gauge_file, len(builder.qpts), builder.kernel_id)
    geometry = PhysicalPairMap(
        cell,
        auxcell,
        kpts,
        np.asarray(builder.qpts),
        builder.pair_q,
        gauges,
        set_id=str(work),
    )
    if not np.array_equal(geometry.orbits.representatives, orbits.representatives):
        raise ValueError("Auxiliary and orbital orbit enumeration disagree")
    provider = df.GDF(cell, kpts)
    provider.auxcell = auxcell
    provider._cderi = str(work / "representatives-cderi.h5")

    def get_factor(i, j):
        blocks = []
        reader = corrected_provider if corrected and i == j else provider
        for real, imag, sign in reader.sr_loop((kpts[i], kpts[j]), compact=False):
            if sign != 1:
                raise ValueError("Negative metric sectors are unsupported")
            blocks.append(real + 1j * imag)
        value = np.concatenate(blocks).reshape(
            auxcell.nao_nr(), cell.nao_nr(), cell.nao_nr()
        )
        if stored_x is not None:
            value = stored_x[i] @ value @ stored_x[j].conj().T
        return value

    export_started = time.perf_counter()
    summary = write_archive(
        geometry,
        get_factor,
        output,
        set_kind="correlation" if corrected else "hf",
        finite_size_kind="ewald" if corrected else "none",
        stored_x=stored_x,
        producer_revision=producer_revision,
        chunk_size=chunk_size,
        source_mode="representatives",
        reference_get_factor=reference_get_factor,
    )
    summary.update(
        representative_only_build=True,
        three_center_pairs=len(orbits.representatives),
        ordinary_three_center_pairs=builder.three_center_pairs,
        corrected_three_center_pairs=len(diagonal) if corrected else 0,
        corrected_build_seconds=corrected_seconds,
        corrected_producer="green_igen.df._make_j3c" if corrected else None,
        build_seconds=build_seconds,
        metric_seconds=builder.metric_seconds,
        export_seconds=time.perf_counter() - export_started,
        temporary_file_bytes=sum(file.stat().st_size for file in work.glob("*.h5")),
        gauge_cache_peak_bytes=gauges.cache.peak,
        transform_cache_peak_bytes=geometry.cache.peak,
    )
    (work / "build-manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def store_captured_q_transforms(args, cell, kpts, auxbasis, gauge_file, archive):
    """Preserve q stars/weights and identify their actual correlation DF frame."""
    from .common_utils import init_q_mesh
    from .symmetry_utils import get_representation
    from .integral_symmetry_transform import auxiliary_transform

    aux = df.df.make_modrho_basis(cell, auxbasis)
    qstruct = init_q_mesh(args, aux, kpts, save_data=False)
    with h5py.File(Path(archive) / "meta.h5") as meta:
        scaled = meta["sg/q_mesh"][()] / int(meta["sg/q_denominator"][()])
        kernel = meta.attrs["kernel_id"]
    gauges = CapturedGauges(gauge_file, len(scaled), kernel)

    def captured_index(point):
        delta = scaled - aux.get_scaled_kpts(point)
        indices = np.flatnonzero(np.max(np.abs(delta - np.rint(delta)), axis=1) < 1e-9)
        if len(indices) != 1:
            raise ValueError("Input q point is absent from captured metric space")
        return int(indices[0])

    matrices = []
    for iq, point in enumerate(qstruct.kpts):
        representative = int(qstruct.ibz2bz[qstruct.bz2ibz[iq]])
        if iq == representative:
            matrices.append(np.eye(aux.nao_nr(), dtype=complex))
            continue
        source = gauges[captured_index(qstruct.kpts[representative])]
        target = gauges[captured_index(point)]
        if qstruct.time_reversal_symm_bz[iq]:
            target = SimpleNamespace(
                unwhitening=target.unwhitening.conj(), kernel_id=kernel
            )
        raw = get_representation(iq, int(qstruct.stars_ops_bz[iq]), aux, qstruct)
        matrices.append(auxiliary_transform(source, target, raw))
    with h5py.File(args.output_path, "a") as file:
        dataset = file["symmetry/q/k_sym_transform_p0"]
        if dataset.shape != np.asarray(matrices).shape:
            raise ValueError(
                "Captured q frame would change the existing star structure"
            )
        dataset[...] = np.asarray(matrices)
