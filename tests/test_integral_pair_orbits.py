"""Momentum diagnostics only; physical factor covariance is tested separately."""

from itertools import product

import numpy as np
import pytest

from green_mbtools.mint.integral_symmetry import (
    CrystalOperation,
    IDENTITY,
    IntegerMesh,
    build_pair_orbits,
    validate_group,
)


def mesh(shape=(3, 3, 1), shift=(0, 0, 0)):
    return IntegerMesh.from_scaled(
        np.array(list(product(*(range(n) for n in shape)))) / shape + shift
    )


def c4():
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    return [CrystalOperation(np.linalg.matrix_power(rotation, p)) for p in range(4)]


def assert_actions(orbits):
    coords = orbits.mesh.coordinates
    denominator = orbits.mesh.denominator
    for i, j in product(range(len(coords)), repeat=2):
        rep = orbits.representatives[orbits.pair_to_representative[i, j]]
        operated = (
            coords[rep]
            @ orbits.operations[orbits.pair_operation[i, j]].reciprocal_rotation.T
        )
        if orbits.time_reversal[i, j]:
            operated = -operated
        if orbits.exchange[i, j]:
            operated = operated[::-1]
        np.testing.assert_array_equal(
            operated, coords[[i, j]] + denominator * orbits.reciprocal_wraps[i, j]
        )
        # q sign and its reciprocal carry follow the same two-leg action.
        np.testing.assert_array_equal(
            operated[1] - operated[0],
            coords[j]
            - coords[i]
            + denominator
            * (orbits.reciprocal_wraps[i, j, 1] - orbits.reciprocal_wraps[i, j, 0]),
        )


def test_identity_has_no_accidental_compression():
    orbits = build_pair_orbits(mesh(), [IDENTITY])
    assert len(orbits.representatives) == 81
    np.testing.assert_array_equal(
        orbits.pair_to_representative, np.arange(81).reshape(9, 9)
    )
    assert_actions(orbits)


@pytest.mark.parametrize("tr,ex", list(product((False, True), repeat=2)))
def test_coverage_disjointness_stabilizers_and_determinism(tr, ex):
    orbits = build_pair_orbits(mesh(), c4(), time_reversal=tr, exchange=ex)
    reverse = build_pair_orbits(
        mesh(), list(reversed(c4())), time_reversal=tr, exchange=ex
    )
    np.testing.assert_array_equal(
        orbits.pair_to_representative, reverse.pair_to_representative
    )
    np.testing.assert_array_equal(orbits.pair_operation, reverse.pair_operation)
    assert sum(orbits.orbit_sizes) == 81
    assert orbits.pair_to_representative[0, 0] == 0
    assert orbits.orbit_sizes[0] == 1  # Gamma pair has the complete stabilizer.
    np.testing.assert_array_equal(
        np.bincount(orbits.pair_to_representative.ravel()), orbits.orbit_sizes
    )
    assert_actions(orbits)


def test_inverse_group_and_affine_translation_carry():
    translated_identity = CrystalOperation(np.eye(3, dtype=int), (0.5, 0, 0))
    ops, multiplication, carries, inverses = validate_group(
        [translated_identity, IDENTITY]
    )
    assert len(ops) == 2  # Do not discard a nontrivial operation fixing every k.
    assert multiplication[1, 1] == 0
    np.testing.assert_array_equal(carries[1, 1], [1, 0, 0])
    np.testing.assert_array_equal(multiplication[np.arange(2), inverses], [0, 0])
    orbits = build_pair_orbits(mesh(), ops)
    assert len(orbits.operations) == 2
    assert len(orbits.representatives) == 81


def test_nonsymmorphic_screw_and_reciprocal_boundary():
    screw = CrystalOperation(np.diag([-1, -1, 1]), (0, 0, 0.5))
    orbits = build_pair_orbits(
        mesh((3, 3, 2)), [IDENTITY, screw], time_reversal=True, exchange=True
    )
    assert np.any(orbits.reciprocal_wraps)
    assert_actions(orbits)


def test_anisotropic_mesh_selects_closed_preserving_subgroup():
    orbits = build_pair_orbits(mesh((2, 3, 1)), c4())
    assert len(orbits.operations) == 2
    assert_actions(orbits)


def test_shifted_mesh_and_time_reversal_rejection():
    shifted = mesh((3, 3, 1), (1 / 7, 0, 0))
    orbits = build_pair_orbits(shifted, c4())
    assert len(orbits.operations) == 1
    assert len(orbits.representatives) == 81
    with pytest.raises(ValueError, match="preserve"):
        build_pair_orbits(shifted, c4(), time_reversal=True)
    assert_actions(
        build_pair_orbits(mesh((2, 2, 1), (0.25, 0.25, 0)), c4(), time_reversal=True)
    )


def test_triclinic_identity_only_arbitrary_integer_mesh():
    low_symmetry = IntegerMesh(np.array([[0, 0, 0], [1, 2, 3], [3, 5, 1]]), 11)
    orbits = build_pair_orbits(low_symmetry, [IDENTITY])
    assert len(orbits.representatives) == 9
    assert_actions(orbits)


def test_reject_nonclosed_group_missing_identity_and_bad_mesh():
    with pytest.raises(ValueError, match="closed"):
        build_pair_orbits(mesh(), c4()[:2])
    with pytest.raises(ValueError, match="identity"):
        build_pair_orbits(mesh(), c4()[1:])
    with pytest.raises(ValueError, match="duplicate"):
        IntegerMesh.from_scaled([[0, 0, 0], [1, 0, 0]])
    with pytest.raises(ValueError, match="integers"):
        CrystalOperation(np.eye(3) * 0.5)
    with pytest.raises(ValueError, match="rational"):
        IntegerMesh.from_scaled([[np.sqrt(2), 0, 0]])


def test_equivalent_input_wrappings_are_recorded_exactly():
    negative = IntegerMesh.from_scaled([[0, 0, 0], [-1 / 3, 0, 0], [1 / 3, 0, 0]])
    orbits = build_pair_orbits(negative, [IDENTITY], time_reversal=True, exchange=True)
    assert_actions(orbits)
