"""Actual crystal operations in a captured integral-set-specific auxiliary gauge.

This producer/validation helper depends on PySCF. Archive consumers use the
serialized geometry instead. Dense operation maps are cached with a byte limit,
never stored for every ordered pair.
"""

from collections import OrderedDict

import numpy as np

from .integral_symmetry import (
    CrystalOperation,
    IntegerMesh,
    build_pair_orbits,
    validated_cell_operations,
    integral_kstruct,
)
from .integral_symmetry_transform import (
    auxiliary_transform,
    reconstruct_factor,
    stored_basis_operation,
)
from .symmetry_utils import get_representation


class MatrixCache:
    def __init__(self, budget=64 * 1024**2):
        if not isinstance(budget, int) or budget < 0:
            raise ValueError("Cache budget must be a nonnegative integer")
        self.budget, self.bytes, self.peak = budget, 0, 0
        self.hits, self.misses, self.evictions = 0, 0, 0
        self.items = OrderedDict()

    def get(self, key, factory):
        if key in self.items:
            self.hits += 1
            self.items.move_to_end(key)
            return self.items[key]
        self.misses += 1
        value = np.array(factory(), dtype=np.complex128, copy=True)
        value.setflags(write=False)
        if value.nbytes <= self.budget:
            while self.bytes + value.nbytes > self.budget:
                _, removed = self.items.popitem(last=False)
                self.bytes -= removed.nbytes
                self.evictions += 1
            self.items[key] = value
            self.bytes += value.nbytes
            self.peak = max(self.peak, self.bytes)
        return value


class PhysicalPairMap:
    def __init__(
        self,
        cell,
        auxcell,
        kpts,
        qpts,
        pair_q,
        gauges,
        *,
        set_id,
        time_reversal=True,
        exchange=True,
        cache_bytes=64 * 1024**2,
        atol=1e-10,
        rtol=1e-10,
    ):
        if not set_id:
            raise ValueError("Integral set identity is required")
        self.cell, self.auxcell = cell, auxcell
        self.kpts, self.qpts = np.asarray(kpts), np.asarray(qpts)
        self.pair_q, self.gauges, self.set_id = pair_q, gauges, set_id
        self.atol, self.rtol = atol, rtol
        self.mesh = IntegerMesh.from_scaled(cell.get_scaled_kpts(kpts))
        self.qmesh = IntegerMesh.from_scaled(auxcell.get_scaled_kpts(qpts))
        self.kstruct = integral_kstruct(cell, kpts)
        self.qstruct = integral_kstruct(auxcell, qpts)
        self.orbits = build_pair_orbits(
            self.mesh,
            validated_cell_operations(cell, self.kstruct),
            time_reversal=time_reversal,
            exchange=exchange,
        )
        self.orbital_ids = {
            CrystalOperation(op.rot, op.trans): i
            for i, op in enumerate(self.kstruct.ops)
        }
        self.auxiliary_ids = {
            CrystalOperation(op.rot, op.trans): i
            for i, op in enumerate(self.qstruct.ops)
        }
        if not all(op in self.auxiliary_ids for op in self.orbits.operations):
            raise ValueError(
                "Auxiliary basis does not preserve the retained crystal group"
            )
        self.q_ids = {
            tuple(row % self.qmesh.denominator): i
            for i, row in enumerate(self.qmesh.coordinates)
        }
        nk = len(kpts)
        if set(pair_q) != {(i, j) for i in range(nk) for j in range(nk)}:
            raise ValueError("Captured gauges must cover the full ordered mesh")
        for (i, j), iq in pair_q.items():
            if iq not in gauges or not 0 <= iq < len(qpts):
                raise ValueError("Pair references an absent auxiliary gauge")
            delta = cell.get_scaled_kpts(kpts[j] - kpts[i] - qpts[iq])
            if not np.allclose(delta, np.rint(delta), atol=1e-9, rtol=0):
                raise ValueError("Captured pair gauge has an incompatible q convention")
        self.cache = MatrixCache(cache_bytes)

    def spatial(self, source, operation, *, tr=False, exchange=False):
        """Describe one arbitrary operation, including stabilizers and inverses."""
        spatial_points, _ = self.mesh.action(operation)
        si, sj = map(int, spatial_points[list(source)])
        points, _ = self.mesh.action(operation, tr)
        target = tuple(map(int, points[list(source)]))
        if exchange:
            target = target[::-1]
        orbital_id = self.orbital_ids[operation]
        ui = self.cache.get(
            (self.set_id, "AO", operation.key, si),
            lambda: get_representation(
                si, orbital_id, self.cell, self.kstruct, tr_phase=False
            ),
        )
        uj = self.cache.get(
            (self.set_id, "AO", operation.key, sj),
            lambda: get_representation(
                sj, orbital_id, self.cell, self.kstruct, tr_phase=False
            ),
        )
        qscaled = self.cell.get_scaled_kpts(self.kpts[sj] - self.kpts[si])
        iq = self.q_ids[
            tuple(
                np.rint(qscaled * self.qmesh.denominator).astype(np.int64)
                % self.qmesh.denominator
            )
        ]
        auxiliary_id = self.auxiliary_ids[operation]
        conjugate = tr ^ exchange

        def whitened():
            raw = get_representation(
                iq, auxiliary_id, self.auxcell, self.qstruct, tr_phase=False
            )
            if conjugate:
                raw = raw.conj()
            return auxiliary_transform(
                self.gauges[self.pair_q[source]],
                self.gauges[self.pair_q[target]],
                raw,
                conjugate=conjugate,
                atol=self.atol,
                rtol=self.rtol,
            )

        d = self.cache.get(
            (
                self.set_id,
                "DF-CD-v1",
                operation.key,
                iq,
                self.pair_q[source],
                self.pair_q[target],
                conjugate,
            ),
            whitened,
        )
        left, right = (uj, ui) if exchange else (ui, uj)
        if tr:
            left, right = left.conj(), right.conj()
        return target, (d, left, right, conjugate, exchange)

    def pair(self, i, j):
        o = self.orbits
        rep = tuple(map(int, o.representatives[o.pair_to_representative[i, j]]))
        target, maps = self.spatial(
            rep,
            o.operations[o.pair_operation[i, j]],
            tr=bool(o.time_reversal[i, j]),
            exchange=bool(o.exchange[i, j]),
        )
        if target != (i, j):
            raise ValueError("Pair action does not land on the requested pair")
        return rep, maps

    @staticmethod
    def reconstruct(
        source, maps, *, output_slice=None, source_x_inverse=None, target_x=None
    ):
        d, left, right, conjugate, transpose = maps
        if source_x_inverse is not None or target_x is not None:
            if source_x_inverse is None or target_x is None:
                raise ValueError(
                    "Both source and target stored-basis transforms are required"
                )
            # Transpose exchanges the covariant/contravariant AO legs, so the
            # effective conjugation of their stored-basis maps is TR, not TR^EX.
            indices = (1, 0) if transpose else (0, 1)
            tr = conjugate ^ transpose
            left = stored_basis_operation(
                left, target_x[0], source_x_inverse[indices[0]], conjugate=tr
            )
            right = stored_basis_operation(
                right, target_x[1], source_x_inverse[indices[1]], conjugate=tr
            )
        return reconstruct_factor(
            source,
            d,
            left,
            right,
            conjugate=conjugate,
            transpose=transpose,
            output_slice=output_slice,
        )
