"""Exact mesh actions for ordered periodic integral pairs.

This module does not change legacy pair maps or integral production.  Momentum
actions use integer arithmetic.  A pair action is spatial, then time reversal,
then exchange; its direction is always representative -> requested pair.
Translations are retained even when two operations induce the same k map.
Tensor/gauge validity must be established separately before omitting factors.
"""

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from math import gcd

import numpy as np


def _fraction(value, max_denominator=4096, tolerance=1e-10):
    result = Fraction(str(value)).limit_denominator(max_denominator)
    if abs(float(result) - float(value)) > tolerance:
        raise ValueError("Coordinate has no validated rational mesh representation")
    return result


@dataclass(frozen=True)
class CrystalOperation:
    """Direct fractional coordinates transform as r_target = R r_source + t."""

    rotation: tuple
    translation: tuple = (0, 0, 0)

    def __post_init__(self):
        rotation = np.asarray(self.rotation)
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            raise ValueError("A rotation must be a finite 3 by 3 matrix")
        if (
            not np.array_equal(rotation, np.rint(rotation))
            or np.max(np.abs(rotation)) > 10000
        ):
            raise ValueError("A fractional rotation must contain bounded integers")
        rotation = rotation.astype(np.int64)
        determinant = int(round(np.linalg.det(rotation)))
        inverse = (
            np.rint(np.linalg.inv(rotation)).astype(np.int64) if determinant else None
        )
        if abs(determinant) != 1 or not np.array_equal(
            rotation @ inverse, np.eye(3, dtype=np.int64)
        ):
            raise ValueError("A fractional rotation must be unimodular")
        if len(self.translation) != 3:
            raise ValueError("A translation must have three components")
        translation = tuple(_fraction(x, 96, 1e-6) % 1 for x in self.translation)
        object.__setattr__(
            self, "rotation", tuple(tuple(int(x) for x in row) for row in rotation)
        )
        object.__setattr__(self, "translation", translation)

    @property
    def reciprocal_rotation(self):
        return np.rint(np.linalg.inv(self.rotation).T).astype(np.int64)

    @property
    def key(self):
        return tuple(x for row in self.rotation for x in row) + self.translation

    def compose(self, right):
        """Return self after right, including the discarded lattice translation."""
        rotation = np.asarray(self.rotation) @ np.asarray(right.rotation)
        translation = tuple(
            self.translation[i]
            + sum(self.rotation[i][j] * right.translation[j] for j in range(3))
            for i in range(3)
        )
        carry = tuple(int(x // 1) for x in translation)
        return CrystalOperation(rotation, translation), carry


IDENTITY = CrystalOperation(np.eye(3, dtype=int))


@dataclass(frozen=True)
class IntegerMesh:
    """k_scaled = coordinates / denominator, retaining the input BZ wrapping."""

    coordinates: np.ndarray
    denominator: int

    def __post_init__(self):
        coords = np.asarray(self.coordinates)
        if coords.ndim != 2 or coords.shape[1] != 3 or not len(coords):
            raise ValueError("A nonempty k mesh must have shape (nk, 3)")
        if not np.all(np.isfinite(coords)) or not np.array_equal(
            coords, np.rint(coords)
        ):
            raise ValueError("Mesh coordinates must be finite integers")
        if (
            not isinstance(self.denominator, (int, np.integer))
            or not 0 < self.denominator <= 2**31
        ):
            raise ValueError("Mesh denominator must be a bounded positive integer")
        if np.max(np.abs(coords)) > 2**31:
            raise ValueError("Mesh coordinates exceed the supported integer bound")
        coords = np.array(coords, dtype=np.int64, copy=True)
        keys = {tuple(row % self.denominator) for row in coords}
        if len(keys) != len(coords):
            raise ValueError(
                "The mesh contains duplicate points modulo reciprocal vectors"
            )
        coords.setflags(write=False)
        object.__setattr__(self, "coordinates", coords)
        object.__setattr__(self, "denominator", int(self.denominator))

    @classmethod
    def from_scaled(cls, scaled, tolerance=1e-10, max_denominator=4096):
        scaled = np.asarray(scaled)
        if scaled.ndim != 2 or scaled.shape[1] != 3 or not np.all(np.isfinite(scaled)):
            raise ValueError("Scaled k coordinates must be finite with shape (nk, 3)")
        fractions = [
            [_fraction(x, max_denominator, tolerance) for x in row] for row in scaled
        ]
        denominator = 1
        for row in fractions:
            for value in row:
                denominator = (
                    denominator
                    // gcd(denominator, value.denominator)
                    * value.denominator
                )
        if denominator > 2**31:
            raise ValueError(
                "Common mesh denominator exceeds the supported integer bound"
            )
        return cls(
            np.array([[int(x * denominator) for x in row] for row in fractions]),
            denominator,
        )

    def action(self, operation, time_reversal=False):
        """Return source->target point IDs and transformed-minus-target wraps."""
        lookup = {
            tuple(row % self.denominator): i for i, row in enumerate(self.coordinates)
        }
        transformed = self.coordinates @ operation.reciprocal_rotation.T
        if time_reversal:
            transformed = -transformed
        targets = []
        wraps = []
        for row in transformed:
            key = tuple(row % self.denominator)
            if key not in lookup:
                raise ValueError("Operation does not preserve the actual k mesh")
            target = lookup[key]
            targets.append(target)
            wraps.append((row - self.coordinates[target]) // self.denominator)
        return np.array(targets, dtype=np.int64), np.array(wraps, dtype=np.int64)


@dataclass(frozen=True)
class PairOrbits:
    mesh: IntegerMesh
    operations: tuple
    multiplication: np.ndarray
    composition_translations: np.ndarray
    inverses: np.ndarray
    representatives: np.ndarray
    pair_to_representative: np.ndarray
    pair_operation: np.ndarray
    time_reversal: np.ndarray
    exchange: np.ndarray
    reciprocal_wraps: np.ndarray
    orbit_sizes: np.ndarray

    def summary(self):
        arrays = (
            self.multiplication,
            self.composition_translations,
            self.inverses,
            self.representatives,
            self.pair_to_representative,
            self.pair_operation,
            self.time_reversal,
            self.exchange,
            self.reciprocal_wraps,
            self.orbit_sizes,
        )
        return {
            "nk": len(self.mesh.coordinates),
            "full_ordered_pairs": len(self.mesh.coordinates) ** 2,
            "space_group_representatives": len(self.representatives),
            "mesh_preserving_operations": len(self.operations),
            "orbit_size_histogram": dict(
                sorted(Counter(map(int, self.orbit_sizes)).items())
            ),
            "descriptor_array_bytes": sum(x.nbytes for x in arrays)
            + self.mesh.coordinates.nbytes,
            "q_convention": "k_j_minus_k_i",
            "transform_direction": "representative_to_requested",
            "operation_order": "spatial,time_reversal,exchange",
            "tensor_reconstruction_validated": False,
        }


def validate_group(operations):
    """Validate the affine group modulo lattice translations, including carries."""
    operations = tuple(sorted(set(operations), key=lambda op: (op != IDENTITY, op.key)))
    if not operations or operations[0] != IDENTITY:
        raise ValueError("The retained group must contain the affine identity")
    lookup = {op.key: i for i, op in enumerate(operations)}
    nops = len(operations)
    multiplication = np.empty((nops, nops), dtype=np.int64)
    carries = np.empty((nops, nops, 3), dtype=np.int64)
    for i, left in enumerate(operations):
        for j, right in enumerate(operations):
            product, carry = left.compose(right)
            if product.key not in lookup:
                raise ValueError("Retained affine operations are not a closed group")
            multiplication[i, j] = lookup[product.key]
            carries[i, j] = carry
    inverses = np.empty(nops, dtype=np.int64)
    for i in range(nops):
        candidates = np.flatnonzero(
            (multiplication[i] == 0) & (multiplication[:, i] == 0)
        )
        if len(candidates) != 1:
            raise ValueError("Retained operations have no unique two-sided inverse")
        inverses[i] = candidates[0]
    return operations, multiplication, carries, inverses


def build_pair_orbits(mesh, operations, *, time_reversal=False, exchange=False):
    """Build disjoint ordered-pair orbits using the same operation on both legs.

    Only a closed mesh-preserving subgroup is retained.  A requested time
    reversal that does not preserve the mesh is rejected, never approximated.
    Exchange/TR flags describe momentum relations only at this diagnostic stage.
    """
    preserving = []
    for operation in operations:
        try:
            mesh.action(operation)
        except ValueError:
            continue
        preserving.append(operation)
    operations, multiplication, carries, inverses = validate_group(preserving)
    actions = []
    for iop, operation in enumerate(operations):
        for tr in range(2 if time_reversal else 1):
            point_map, wraps = mesh.action(operation, bool(tr))
            for ex in range(2 if exchange else 1):
                actions.append((iop, bool(tr), bool(ex), point_map, wraps))
    nk = len(mesh.coordinates)
    pair_map = np.full((nk, nk), -1, dtype=np.int64)
    op_map = np.full((nk, nk), -1, dtype=np.int64)
    tr_map = np.zeros((nk, nk), dtype=np.bool_)
    ex_map = np.zeros((nk, nk), dtype=np.bool_)
    pair_wraps = np.zeros((nk, nk, 2, 3), dtype=np.int64)
    representatives = []
    sizes = []
    for pair in range(nk * nk):
        i, j = divmod(pair, nk)
        if pair_map[i, j] >= 0:
            continue
        rep = len(representatives)
        representatives.append((i, j))
        size = 0
        for iop, tr, ex, point_map, wraps in actions:
            ti, tj = int(point_map[i]), int(point_map[j])
            source_legs = (i, j)
            if ex:
                ti, tj = tj, ti
                source_legs = (j, i)
            if pair_map[ti, tj] >= 0:
                if pair_map[ti, tj] != rep:
                    raise ValueError("Overlapping pair orbits: invalid group action")
                continue
            pair_map[ti, tj] = rep
            op_map[ti, tj] = iop
            tr_map[ti, tj] = tr
            ex_map[ti, tj] = ex
            pair_wraps[ti, tj] = wraps[list(source_legs)]
            size += 1
        sizes.append(size)
    if np.any(pair_map < 0) or sum(sizes) != nk * nk:
        raise ValueError("Pair orbits do not cover the complete mesh")
    return PairOrbits(
        mesh,
        operations,
        multiplication,
        carries,
        inverses,
        np.array(representatives, dtype=np.int64),
        pair_map,
        op_map,
        tr_map,
        ex_map,
        pair_wraps,
        np.array(sizes, dtype=np.int64),
    )


def integral_kstruct(cell, kpts):
    """Full mathematical crystal group, independent of one-body/FFT filtering.

    PySCF defaults to symmorphic operations and can additionally discard
    translations incompatible with its density FFT grid. Those are separate
    from integral-pair symmetry. Numerical covariance is validated before use.
    The supplied cell and its one-body symmetry objects are never modified.
    """
    from pyscf.pbc.lib import kpts as libkpts

    analysis_cell = cell.copy()
    analysis_cell.lattice_symmetry = None
    return libkpts.make_kpts(
        analysis_cell,
        kpts,
        space_group_symmetry=True,
        time_reversal_symmetry=False,
        symmorphic=False,
        check_mesh_symmetry=False,
    )


def validated_cell_operations(cell, kstruct, tolerance=1e-6):
    """Retain actual PySCF operations after species/basis/pseudopotential checks.

    The existing orbital helper checks angular-shell maps but does not compare
    all Gaussian exponents and contractions.  Check those here before allowing
    an operation into an integral-pair group.
    """
    from .symmetry_utils import generate_permutation_info

    if cell.dimension != 3 or cell.cart or cell.omega != 0:
        raise ValueError(
            "Initial SG support requires scalar 3D spherical AOs and ordinary Coulomb"
        )
    result = []
    for op in kstruct.ops:
        partners, _ = generate_permutation_info(cell, op, tol=tolerance)
        if len(set(map(int, partners))) != cell.natm:
            raise ValueError("Atom map is not a permutation")
        for i, j in enumerate(partners):
            si, sj = cell.atom_symbol(i), cell.atom_symbol(int(j))
            for assignment in (cell._basis, cell._pseudo, cell._ecp):
                if repr(assignment.get(si)) != repr(assignment.get(sj)):
                    raise ValueError(
                        "Operation changes species-specific basis/potential assignments"
                    )
        result.append(CrystalOperation(op.rot, op.trans))
    return result
