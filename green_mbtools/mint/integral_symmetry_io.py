"""Versioned SG archives with bounded, set-specific host reconstruction.

See integral_symmetry_format.md. Legacy datasets and filenames are deliberately
absent. Offline conversion validates every omitted factor in its actual gauge.
"""

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from .integral_symmetry import CrystalOperation, IntegerMesh, validate_group
from .integral_symmetry_geometry import MatrixCache, PhysicalPairMap
from .integral_symmetry_transform import CholeskyGauge, auxiliary_transform

FORMAT = "green.df.space_group.v1"
PACKING = "complex128_interleaved_f64"
MAX_FACTOR_BUFFER_BYTES = 128 * 1024**2
REQUIRED_ATTRIBUTES = (
    "integral_format",
    "format_version",
    "set_kind",
    "finite_size_kind",
    "orbital_basis",
    "producer_revision",
    "input_fingerprint",
    "auxiliary_gauge_id",
    "complex_storage",
    "transform_direction",
    "operation_order",
    "kernel_id",
)


def pack(value):
    return np.ascontiguousarray(value, dtype=np.complex128).view(np.float64)


def unpack(value, shape):
    value = np.asarray(value)
    if value.dtype != np.dtype("float64") or value.shape != (
        *shape[:-1],
        2 * shape[-1],
    ):
        raise ValueError("Invalid complex128 interleaved float64 layout")
    if not np.all(np.isfinite(value)):
        raise ValueError("Nonfinite archive matrix/factor")
    return np.ascontiguousarray(value).view(np.complex128).reshape(shape)


def input_fingerprint(cell, kpts, stored_x=None):
    payload = {
        "lattice": cell.lattice_vectors(),
        "atoms": cell._atom,
        "basis": cell._basis,
        "pseudo": cell._pseudo,
        "ecp": cell._ecp,
        "dimension": cell.dimension,
        "cart": cell.cart,
        "charge": cell.charge,
        "spin": cell.spin,
        "omega": cell.omega,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x),
    )
    digest = hashlib.sha256(encoded.encode())
    mesh = IntegerMesh.from_scaled(cell.get_scaled_kpts(kpts))
    digest.update(np.asarray([mesh.denominator], dtype="<i8").tobytes())
    digest.update(np.asarray(mesh.coordinates, dtype="<i8").tobytes())
    if stored_x is not None:
        digest.update(pack(stored_x).astype("<f8").tobytes())
    return digest.hexdigest()


def _sparse_operations(cell, kstruct, operations, ids):
    from .symmetry_utils import generate_permutation_info, get_representation

    rows, cols, values, phases, offsets = [], [], [], [], [0]
    slices = cell.aoslice_by_atom()
    scaled = np.asarray(kstruct.kpts_scaled[0])
    for operation in operations:
        opid = ids[operation]
        partners, shifts = generate_permutation_info(cell, kstruct.ops[opid])
        phase_rows = np.empty((cell.nao_nr(), 3))
        for atom, target in enumerate(partners):
            phase_rows[slices[target, 2] : slices[target, 3]] = shifts[atom]
        if not np.allclose(phase_rows, np.rint(phase_rows), atol=1e-6, rtol=0):
            raise ValueError("Operation phase is not a validated lattice translation")
        phased = get_representation(0, opid, cell, kstruct, tr_phase=False)
        base = phased / np.exp(2j * np.pi * (phase_rows @ scaled))[:, None]
        r, c = np.nonzero(base)
        rows.extend(r.tolist())
        cols.extend(c.tolist())
        values.extend(base[r, c].tolist())
        phases.append(phase_rows)
        offsets.append(len(rows))
    return {
        "offsets": np.array(offsets, dtype=np.int64),
        "rows": np.array(rows, dtype=np.int64),
        "columns": np.array(cols, dtype=np.int64),
        "values": pack(np.array(values)),
        "phase_vectors": np.array(phases),
    }


def write_archive(
    geometry,
    get_factor,
    output,
    *,
    set_kind,
    finite_size_kind,
    producer_revision,
    fingerprint=None,
    stored_x=None,
    chunk_size=32,
    atol=1e-10,
    rtol=1e-10,
    source_mode="complete",
    reference_get_factor=None,
):
    """Write a fresh archive from complete factors or a representative builder.

    Complete conversion verifies every omitted factor. A representative build
    verifies captured metric covariance for every pair and representative
    stabilizers; a separate physical reference can additionally be supplied.
    """
    output = Path(output)
    if output.exists():
        raise ValueError("Refusing to overwrite an integral archive")
    if (
        set_kind not in ("hf", "correlation")
        or not finite_size_kind
        or not producer_revision
    ):
        raise ValueError("Integral set/producer provenance is required")
    if finite_size_kind not in ("none", "ewald"):
        raise ValueError("Special finite-size profiles require separate SG validation")
    expected_kernel = (
        "ewald-Coulomb" if finite_size_kind == "ewald" else "ordinary-Coulomb"
    )
    if any(gauge.kernel_id != expected_kernel for gauge in geometry.gauges.values()):
        raise ValueError("Finite-size profile differs from captured Coulomb kernel")
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("Chunk capacity must be a positive integer")
    if source_mode not in ("complete", "representatives"):
        raise ValueError("Unknown integral source mode")
    o = geometry.orbits
    nk, nao, nq = len(geometry.kpts), geometry.cell.nao_nr(), len(geometry.qpts)
    naux = geometry.auxcell.nao_nr()
    factor_bytes = naux * nao * nao * 16
    if factor_bytes > MAX_FACTOR_BUFFER_BYTES:
        raise ValueError("One factor exceeds the initial 128 MiB source-buffer bound")
    chunk_size = min(chunk_size, max(1, MAX_FACTOR_BUFFER_BYTES // factor_bytes))
    fingerprint = fingerprint or input_fingerprint(
        geometry.cell, geometry.kpts, stored_x
    )
    x_inverse = None
    if stored_x is not None:
        stored_x = np.asarray(stored_x, dtype=complex)
        if stored_x.shape != (nk, nao, nao) or not np.all(np.isfinite(stored_x)):
            raise ValueError(
                "Initial archive basis transform must be square and finite"
            )
        x_inverse = np.linalg.inv(stored_x)
    if source_mode == "representatives":
        for pair in o.representatives:
            pair = tuple(map(int, pair))
            source = np.asarray(get_factor(*pair), dtype=complex)
            if source.shape != (naux, nao, nao) or not np.all(np.isfinite(source)):
                raise ValueError("Malformed representative source")
            for operation_id, operation in enumerate(o.operations):
                for tr in (False, True) if o.time_reversal_enabled else (False,):
                    points, _ = o.mesh.action(operation, tr)
                    target = tuple(map(int, points[list(pair)]))
                    for exchange in (False, True) if o.exchange_enabled else (False,):
                        if (target[::-1] if exchange else target) != pair:
                            continue
                        _, maps = geometry.spatial(
                            pair, operation, tr=tr, exchange=exchange
                        )
                        kw = (
                            {}
                            if stored_x is None
                            else {
                                "source_x_inverse": (
                                    x_inverse[pair[0]],
                                    x_inverse[pair[1]],
                                ),
                                "target_x": (stored_x[pair[0]], stored_x[pair[1]]),
                            }
                        )
                        transformed = geometry.reconstruct(source, maps, **kw)
                        if not np.allclose(transformed, source, atol=atol, rtol=rtol):
                            error = float(np.max(np.abs(transformed - source)))
                            raise ValueError(
                                f"Representative stabilizer covariance failed for {pair}: "
                                f"operation={operation_id}, TR={tr}, exchange={exchange}, "
                                f"max_absolute={error:.12g}"
                            )
    # Validate before writing any directory, and bound reference residency to
    # one representative and one requested factor at a time.
    reference_get_factor = reference_get_factor or (
        get_factor if source_mode == "complete" else None
    )
    for i in range(nk):
        for j in range(nk):
            rep, maps = geometry.pair(i, j)
            if reference_get_factor is None:
                continue  # geometry.pair already checks this target's metric covariance
            reference = np.asarray(reference_get_factor(i, j), dtype=complex)
            if reference.shape != (naux, nao, nao) or not np.all(
                np.isfinite(reference)
            ):
                raise ValueError("Complete source contains malformed factors")
            kw = (
                {}
                if stored_x is None
                else {
                    "source_x_inverse": (x_inverse[rep[0]], x_inverse[rep[1]]),
                    "target_x": (stored_x[i], stored_x[j]),
                }
            )
            reconstructed = geometry.reconstruct(get_factor(*rep), maps, **kw)
            if not np.allclose(reconstructed, reference, atol=atol, rtol=rtol):
                raise ValueError(
                    f"Cannot omit pair {(i, j)}: captured-frame reconstruction failed"
                )
    orbital = _sparse_operations(
        geometry.cell, geometry.kstruct, o.operations, geometry.orbital_ids
    )
    auxiliary = _sparse_operations(
        geometry.auxcell, geometry.qstruct, o.operations, geometry.auxiliary_ids
    )
    gauge_digest = hashlib.sha256()
    for iq in range(nq):
        gauge = geometry.gauges[iq]
        gauge_digest.update(gauge.kernel_id.encode())
        gauge_digest.update(pack(gauge.unwhitening).tobytes())
    gauge_id = gauge_digest.hexdigest()
    nrep = len(o.representatives)
    starts = np.arange(0, nrep, chunk_size, dtype=np.int64)
    counts = np.minimum(chunk_size, nrep - starts)
    output.mkdir(parents=True)
    with h5py.File(output / "meta.h5", "w") as f:
        f.attrs.update(
            integral_format=FORMAT,
            format_version=1,
            set_kind=set_kind,
            finite_size_kind=finite_size_kind,
            orbital_basis="ao" if stored_x is None else "square_X",
            producer_revision=producer_revision,
            input_fingerprint=fingerprint,
            auxiliary_gauge_id=gauge_id,
            complex_storage=PACKING,
            transform_direction="representative_to_requested",
            operation_order="spatial,time_reversal,exchange",
            source_mode=source_mode,
            time_reversal_enabled=o.time_reversal_enabled,
            exchange_enabled=o.exchange_enabled,
            validation_kind="complete_reference"
            if reference_get_factor
            else "metrics_and_stabilizers",
        )
        arrays = {
            "dimensions": [nk, nao, naux, nrep, nq, len(o.operations)],
            "mesh": o.mesh.coordinates,
            "mesh_denominator": o.mesh.denominator,
            "q_mesh": geometry.qmesh.coordinates,
            "q_denominator": geometry.qmesh.denominator,
            "representative_pairs": o.representatives,
            "pair_to_representative": o.pair_to_representative,
            "pair_operation": o.pair_operation,
            "time_reversal": o.time_reversal.astype(np.int64),
            "exchange": o.exchange.astype(np.int64),
            "reciprocal_wraps": o.reciprocal_wraps,
            "pair_q": np.array(
                [[geometry.pair_q[i, j] for j in range(nk)] for i in range(nk)]
            ),
            "rotations": [op.rotation for op in o.operations],
            "translation_numerator": [
                [v.numerator for v in op.translation] for op in o.operations
            ],
            "translation_denominator": [
                [v.denominator for v in op.translation] for op in o.operations
            ],
            "multiplication": o.multiplication,
            "translation_carries": o.composition_translations,
            "inverses": o.inverses,
            "chunk_size": chunk_size,
            "chunk_start": starts,
            "chunk_valid_count": counts,
        }
        for key, value in arrays.items():
            f[f"sg/{key}"] = np.asarray(value, dtype=np.int64)
        for name, sparse in (("orbital", orbital), ("auxiliary", auxiliary)):
            for key, value in sparse.items():
                f[f"sg/{name}/{key}"] = value
        metric = f.create_dataset(
            "sg/captured_C", shape=(nq, naux, 2 * naux), dtype="float64"
        )
        for iq in range(nq):
            metric[iq] = pack(geometry.gauges[iq].unwhitening)
        if stored_x is not None:
            f["sg/stored_x"] = pack(stored_x)
            f["sg/stored_x_inverse"] = pack(x_inverse)
        f.attrs["kernel_id"] = geometry.gauges[0].kernel_id
    for start, count in zip(starts, counts):
        with h5py.File(output / f"SGVQ_{start}.h5", "w") as f:
            data = f.create_dataset(
                "factors", (int(count), naux, nao, 2 * nao), dtype="float64"
            )
            for position in range(int(count)):
                data[position] = pack(
                    get_factor(*map(int, o.representatives[int(start) + position]))
                )
    # Verification uses the serialized sparse phase blocks, not in-memory
    # PySCF maps, and includes final-chunk and complex-packing behavior.
    with ArchiveReader(
        output, expected_fingerprint=fingerprint, expected_set_kind=set_kind
    ) as reader:
        requests = (
            ((i, j) for i in range(nk) for j in range(nk))
            if reference_get_factor
            else map(tuple, o.representatives)
        )
        for i, j in requests:
            reference = (
                reference_get_factor(i, j) if reference_get_factor else get_factor(i, j)
            )
            if not np.allclose(reader.read(i, j), reference, atol=atol, rtol=rtol):
                raise ValueError(f"Serialized reconstruction failed for {(i, j)}")
    return {
        "input_fingerprint": fingerprint,
        "auxiliary_gauge_id": gauge_id,
        "full_pairs": nk * nk,
        "representatives": nrep,
        "chunks": len(starts),
        "file_bytes": sum(p.stat().st_size for p in output.iterdir()),
        "source_build_time_saved": False,
    }


class ArchiveReader:
    def __init__(
        self,
        path,
        *,
        expected_fingerprint=None,
        expected_set_kind=None,
        cache_bytes=256 * 1024**2,
    ):
        self.path = Path(path)
        self.file = h5py.File(self.path / "meta.h5", "r")
        try:
            self._load(expected_fingerprint, expected_set_kind, cache_bytes)
        except Exception:
            self.file.close()
            raise

    def _load(self, expected_fingerprint, expected_set_kind, cache_bytes):
        from fractions import Fraction

        f = self.file
        if any(name not in f.attrs for name in REQUIRED_ATTRIBUTES):
            raise ValueError("Missing required SG archive attribute")
        self.attributes = dict(f.attrs)
        a = self.attributes
        if a["integral_format"] != FORMAT or a["format_version"] != 1:
            raise ValueError("Unknown integral format/version")
        if (
            a["complex_storage"] != PACKING
            or a["transform_direction"] != "representative_to_requested"
            or a["operation_order"] != "spatial,time_reversal,exchange"
        ):
            raise ValueError("Unsupported packing/operation contract")
        if a["set_kind"] not in ("hf", "correlation") or a["orbital_basis"] not in (
            "ao",
            "square_X",
        ):
            raise ValueError("Unsupported integral set/orbital representation")
        if a["kernel_id"] not in ("ordinary-Coulomb", "ewald-Coulomb"):
            raise ValueError("Unsupported Coulomb kernel")
        if (
            a["finite_size_kind"] not in ("none", "ewald")
            or (self.path / "df_ewald.h5").exists()
            or (self.path / "AqQ.h5").exists()
        ):
            raise ValueError(
                "Special finite-size profiles require separate SG validation"
            )
        if a["kernel_id"] != (
            "ewald-Coulomb" if a["finite_size_kind"] == "ewald" else "ordinary-Coulomb"
        ):
            raise ValueError("Finite-size profile differs from captured Coulomb kernel")
        if (
            expected_fingerprint is not None
            and a["input_fingerprint"] != expected_fingerprint
        ):
            raise ValueError("Integral archive/input fingerprint mismatch")
        if expected_set_kind is not None and a["set_kind"] != expected_set_kind:
            raise ValueError("Wrong HF/correlation integral set")
        for name in (
            "input_fingerprint",
            "auxiliary_gauge_id",
            "producer_revision",
            "finite_size_kind",
            "kernel_id",
        ):
            if not a[name]:
                raise ValueError("Missing integral-set provenance")

        def integer(name, shape=None):
            data = f[f"sg/{name}"]
            if data.dtype.kind not in "iu" or (
                shape is not None and data.shape != shape
            ):
                raise ValueError(f"Invalid descriptor shape/type: {name}")
            if data.size * data.dtype.itemsize > 128 * 1024**2:
                raise ValueError("Descriptor array exceeds the initial 128 MiB bound")
            return data[()].astype(np.int64)

        dimensions = integer("dimensions", (6,))
        if np.any(dimensions <= 0) or np.any(dimensions > 2**31):
            raise ValueError("Invalid SG dimensions")
        self.nk, self.nao, self.naux, self.nrep, self.nq, self.nops = map(
            int, dimensions
        )
        nk, nao, naux, nrep, nq, nops = map(int, dimensions)
        self.factor_bytes = naux * nao * nao * 16
        if self.factor_bytes > MAX_FACTOR_BUFFER_BYTES:
            raise ValueError(
                "One factor exceeds the initial 128 MiB source-buffer bound"
            )
        self.mesh = IntegerMesh(
            integer("mesh", (nk, 3)), int(integer("mesh_denominator", ()))
        )
        self.qmesh = IntegerMesh(
            integer("q_mesh", (nq, 3)), int(integer("q_denominator", ()))
        )
        rotations = integer("rotations", (nops, 3, 3))
        tn, td = (
            integer("translation_numerator", (nops, 3)),
            integer("translation_denominator", (nops, 3)),
        )
        if np.any(td <= 0):
            raise ValueError("Invalid fractional translation denominator")
        operations = tuple(
            CrystalOperation(r, tuple(Fraction(int(n), int(d)) for n, d in zip(ns, ds)))
            for r, ns, ds in zip(rotations, tn, td)
        )
        canonical, mult, carries, inverses = validate_group(operations)
        if (
            operations != canonical
            or not np.array_equal(mult, integer("multiplication", (nops, nops)))
            or not np.array_equal(
                carries, integer("translation_carries", (nops, nops, 3))
            )
            or not np.array_equal(inverses, integer("inverses", (nops,)))
        ):
            raise ValueError("Corrupt affine group tables")
        self.representatives = integer("representative_pairs", (nrep, 2))
        self.pair_rep = integer("pair_to_representative", (nk, nk))
        self.pair_op = integer("pair_operation", (nk, nk))
        self.tr, self.ex = (
            integer("time_reversal", (nk, nk)),
            integer("exchange", (nk, nk)),
        )
        self.pair_q = integer("pair_q", (nk, nk))
        wraps = integer("reciprocal_wraps", (nk, nk, 2, 3))
        for value, bound in (
            (self.representatives, nk),
            (self.pair_rep, nrep),
            (self.pair_op, nops),
            (self.tr, 2),
            (self.ex, 2),
            (self.pair_q, nq),
        ):
            if np.any(value < 0) or np.any(value >= bound):
                raise ValueError("Out-of-range SG operation/pair/gauge ID")
        if len(set(map(tuple, self.representatives))) != nrep or set(
            self.pair_rep.flat
        ) != set(range(nrep)):
            raise ValueError("Invalid representative coverage")
        actions = {(i, False): self.mesh.action(op) for i, op in enumerate(operations)}
        self.spatial_points = np.array([actions[i, False][0] for i in range(nops)])
        for i in range(nk):
            for j in range(nk):
                rep = self.representatives[self.pair_rep[i, j]]
                key = (int(self.pair_op[i, j]), bool(self.tr[i, j]))
                if key not in actions:
                    actions[key] = self.mesh.action(operations[key[0]], key[1])
                points, reciprocal = actions[key]
                legs = rep[::-1] if self.ex[i, j] else rep
                if tuple(points[legs]) != (i, j) or not np.array_equal(
                    reciprocal[legs], wraps[i, j]
                ):
                    raise ValueError(
                        "SG pair operation/wrapping does not land on requested pair"
                    )
                delta = (
                    self.mesh.coordinates[j] / self.mesh.denominator
                    - self.mesh.coordinates[i] / self.mesh.denominator
                    - self.qmesh.coordinates[self.pair_q[i, j]] / self.qmesh.denominator
                )
                if not np.allclose(delta, np.rint(delta), atol=1e-10, rtol=0):
                    raise ValueError("Invalid captured gauge q convention")
        self.chunk_size = int(integer("chunk_size", ()))
        if self.chunk_size <= 0:
            raise ValueError("Invalid representative chunk capacity")
        if self.chunk_size * self.factor_bytes > MAX_FACTOR_BUFFER_BYTES:
            raise ValueError("Representative chunk capacity exceeds the 128 MiB bound")
        self.starts = integer("chunk_start")
        self.counts = integer("chunk_valid_count")
        if not np.array_equal(
            self.starts, np.arange(0, nrep, self.chunk_size)
        ) or not np.array_equal(
            self.counts, np.minimum(self.chunk_size, nrep - self.starts)
        ):
            raise ValueError("Invalid representative chunk coverage/final count")
        if f["sg/captured_C"].shape != (nq, naux, 2 * naux) or f[
            "sg/captured_C"
        ].dtype != np.dtype("float64"):
            raise ValueError("Unsupported retained rank/metric precision layout")
        self.sparse = {}
        for name, n in (("orbital", nao), ("auxiliary", naux)):
            offsets = integer(f"{name}/offsets", (nops + 1,))
            rows, cols = integer(f"{name}/rows"), integer(f"{name}/columns")
            values = unpack(f[f"sg/{name}/values"][()], (len(rows),))
            phases = f[f"sg/{name}/phase_vectors"][()]
            if (
                rows.ndim != 1
                or cols.shape != rows.shape
                or offsets[0] != 0
                or offsets[-1] != len(rows)
                or np.any(np.diff(offsets) < 0)
                or np.any(rows < 0)
                or np.any(rows >= n)
                or np.any(cols < 0)
                or np.any(cols >= n)
            ):
                raise ValueError("Invalid sparse operation indices")
            if (
                phases.shape != (nops, n, 3)
                or not np.all(np.isfinite(phases))
                or not np.allclose(phases, np.rint(phases), atol=1e-6, rtol=0)
            ):
                raise ValueError("Invalid operation Bloch phases")
            self.sparse[name] = (offsets, rows, cols, values, phases)
        self.x = self.xinv = None
        if a["orbital_basis"] == "square_X":
            self.x = unpack(f["sg/stored_x"][()], (nk, nao, nao))
            self.xinv = unpack(f["sg/stored_x_inverse"][()], (nk, nao, nao))
            if not np.allclose(self.x @ self.xinv, np.eye(nao), atol=1e-10, rtol=1e-10):
                raise ValueError("Invalid stored-basis inverse/retained subspace")
        self.q_ids = {
            tuple(row % self.qmesh.denominator): i
            for i, row in enumerate(self.qmesh.coordinates)
        }
        self.cache = MatrixCache(cache_bytes)
        self.current_chunk, self.chunk = -1, None
        resident = (
            self.mesh.coordinates,
            self.qmesh.coordinates,
            self.representatives,
            self.pair_rep,
            self.pair_op,
            self.tr,
            self.ex,
            self.pair_q,
            self.spatial_points,
            self.starts,
            self.counts,
        )
        self.descriptor_bytes = sum(array.nbytes for array in resident)
        self.descriptor_bytes += sum(
            array.nbytes for blocks in self.sparse.values() for array in blocks
        )
        if self.x is not None:
            self.descriptor_bytes += self.x.nbytes + self.xinv.nbytes
        # Metric storage is on disk and dense maps live in the separate LRU cache.
        self.metric_storage_bytes = (
            f["sg/captured_C"].size * f["sg/captured_C"].dtype.itemsize
        )

    def _operation(self, name, opid, scaled):
        offsets, rows, cols, values, phases = self.sparse[name]
        size = self.nao if name == "orbital" else self.naux
        result = np.zeros((size, size), dtype=complex)
        start, end = offsets[opid : opid + 2]
        selected_rows = rows[start:end]
        result[selected_rows, cols[start:end]] = values[start:end] * np.exp(
            2j * np.pi * (phases[opid, selected_rows] @ scaled)
        )
        return result

    def _gauge(self, iq):
        from types import SimpleNamespace

        key = f"{self.attributes['auxiliary_gauge_id']}:q{iq}"
        kernel = self.attributes["kernel_id"]

        def load():
            matrix = unpack(self.file["sg/captured_C"][iq], (self.naux, self.naux))
            return CholeskyGauge(matrix, key, kernel).unwhitening

        matrix = self.cache.get((self.attributes["auxiliary_gauge_id"], "C", iq), load)
        # This immutable view refers to a factor validated when loaded into the
        # bounded cache; do not repeat a cubic condition-number test per pair.
        return SimpleNamespace(unwhitening=matrix, gauge_id=key, kernel_id=kernel)

    def representative(self, position):
        if not 0 <= position < self.nrep:
            raise ValueError("Representative ID out of range")
        start = position // self.chunk_size * self.chunk_size
        if self.current_chunk != start:
            count = int(self.counts[start // self.chunk_size])
            with h5py.File(self.path / f"SGVQ_{start}.h5") as f:
                data = f["factors"]
                expected = (count, self.naux, self.nao, 2 * self.nao)
                if (
                    not isinstance(data, h5py.Dataset)
                    or data.shape != expected
                    or data.dtype != np.dtype("float64")
                ):
                    raise ValueError("Invalid factor chunk shape/precision layout")
                if data.size * data.dtype.itemsize > MAX_FACTOR_BUFFER_BYTES:
                    raise ValueError(
                        "Factor chunk exceeds the 128 MiB source-buffer bound"
                    )
                self.chunk = unpack(data[()], (count, self.naux, self.nao, self.nao))
            self.current_chunk = start
        return self.chunk[position - start]

    def read(self, i, j, *, output_slice=None):
        if not 0 <= i < self.nk or not 0 <= j < self.nk:
            raise ValueError("Requested k pair is out of range")
        position, opid = int(self.pair_rep[i, j]), int(self.pair_op[i, j])
        rep = tuple(map(int, self.representatives[position]))
        si, sj = map(int, self.spatial_points[opid, list(rep)])
        tr, ex = bool(self.tr[i, j]), bool(self.ex[i, j])
        conjugate = tr ^ ex
        qscaled = (
            self.mesh.coordinates[sj] - self.mesh.coordinates[si]
        ) / self.mesh.denominator
        iq = self.q_ids[
            tuple(
                np.rint(qscaled * self.qmesh.denominator).astype(np.int64)
                % self.qmesh.denominator
            )
        ]
        identity = (
            self.attributes["set_kind"],
            self.attributes["auxiliary_gauge_id"],
            "complex128",
            "v1",
        )

        def whitened():
            raw = self._operation(
                "auxiliary", opid, self.qmesh.coordinates[iq] / self.qmesh.denominator
            )
            return auxiliary_transform(
                self._gauge(int(self.pair_q[rep])),
                self._gauge(int(self.pair_q[i, j])),
                raw.conj() if conjugate else raw,
                conjugate=conjugate,
            )

        d = self.cache.get(
            (
                *identity,
                "D",
                opid,
                iq,
                int(self.pair_q[rep]),
                int(self.pair_q[i, j]),
                conjugate,
            ),
            whitened,
        )
        ui = self.cache.get(
            (*identity, "AO", opid, si),
            lambda: self._operation(
                "orbital", opid, self.mesh.coordinates[si] / self.mesh.denominator
            ),
        )
        uj = self.cache.get(
            (*identity, "AO", opid, sj),
            lambda: self._operation(
                "orbital", opid, self.mesh.coordinates[sj] / self.mesh.denominator
            ),
        )
        left, right = (uj, ui) if ex else (ui, uj)
        if tr:
            left, right = left.conj(), right.conj()
        kw = (
            {}
            if self.x is None
            else {
                "source_x_inverse": (self.xinv[rep[0]], self.xinv[rep[1]]),
                "target_x": (self.x[i], self.x[j]),
            }
        )
        return PhysicalPairMap.reconstruct(
            self.representative(position),
            (d, left, right, conjugate, ex),
            output_slice=output_slice,
            **kw,
        )

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def expand_archive(archive, input_path, output, *, chunk_size=32):
    """Expand in the exact original legacy irreducible-pair order."""
    from importlib.metadata import version

    output = Path(output)
    if output.exists() or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError(
            "Expansion requires a fresh directory and positive chunk capacity"
        )
    with h5py.File(input_path) as f:
        dimensions = tuple(int(f[f"params/{name}"][()]) for name in ("nk", "nso", "NQ"))
        gauge_id = (
            f["integral_symmetry/correlation_gauge_id"][()]
            if "integral_symmetry/correlation_gauge_id" in f
            else None
        )
        if isinstance(gauge_id, bytes):
            gauge_id = gauge_id.decode()
        pairs = f["symmetry/pairs/kpair_idx"][()][
            f["symmetry/pairs/kpair_irre_list"][()]
        ]
        fingerprint = (
            f["integral_symmetry/input_fingerprint"][()].decode()
            if "integral_symmetry/input_fingerprint" in f
            else None
        )
        if fingerprint is None:
            raise ValueError(
                "Input lacks the SG archive identity; create an identified input copy first"
            )
    with ArchiveReader(archive, expected_fingerprint=fingerprint) as reader:
        if dimensions != (reader.nk, reader.nao, reader.naux):
            raise ValueError("Legacy input/archive dimensions disagree (nk, nso, NQ)")
        if (
            reader.attributes["set_kind"] == "correlation"
            and gauge_id is not None
            and gauge_id != reader.attributes["auxiliary_gauge_id"]
        ):
            raise ValueError(
                "Correlation auxiliary gauge differs from identified input"
            )
        if np.any(pairs < 0) or np.any(pairs >= reader.nk):
            raise ValueError("Legacy input pair order is incompatible with archive")
        chunk_size = min(
            chunk_size, max(1, MAX_FACTOR_BUFFER_BYTES // reader.factor_bytes)
        )
        output.mkdir(parents=True)
        starts = np.arange(0, len(pairs), chunk_size, dtype=np.int64)
        for start in starts:
            block = np.zeros(
                (chunk_size, reader.naux, reader.nao, reader.nao), dtype=complex
            )
            for index, pair in enumerate(pairs[start : start + chunk_size]):
                block[index] = reader.read(*map(int, pair))
            with h5py.File(output / f"VQ_{start}.h5", "w") as f:
                f[str(start)] = pack(block)
        with h5py.File(output / "meta.h5", "w") as f:
            f.attrs["__green_version__"] = version("green-mbtools")
            f["chunk_size"] = chunk_size
            f["chunk_indices"] = starts
            f["chunk_valid_count"] = np.minimum(chunk_size, len(pairs) - starts)
            f.attrs["expanded_from"] = FORMAT
