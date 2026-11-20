"""Independent input, affine-origin and descriptor trust-boundary regressions."""

from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace
from pathlib import Path
import json
import runpy
import sys

import h5py
import numpy as np
import pytest
from pyscf import gto as molecular_gto
from pyscf.pbc import gto

from green_mbtools.mint.integral_symmetry import (
    CrystalOperation,
    IDENTITY,
    IntegerMesh,
    build_pair_orbits,
    validate_group,
    validate_input_kmesh,
    validated_cell_operations,
)


def test_arbitrary_shifted_origin_affine_group_and_carries():
    inversion = CrystalOperation(-np.eye(3, dtype=int), (Fraction(1, 500), 0, 0))
    ops, table, carries, _ = validate_group([IDENTITY, inversion])
    assert ops[1].translation[0] == Fraction(1, 500)
    assert table[1, 1] == 0
    assert not np.any(carries[1, 1])
    origin = np.array([np.sqrt(2) / 100, 0.001, 0.002])
    rotations = [np.diag(s) for s in [(1, 1, 1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1)]]
    shifted = [CrystalOperation(r, (np.eye(3) - r) @ origin) for r in rotations]
    ops, table, carries, _ = validate_group(shifted)
    for i, left in enumerate(ops):
        for j, right in enumerate(ops):
            product, carry = left.compose(right)
            assert product == ops[table[i, j]]
            np.testing.assert_array_equal(carry, carries[i, j])


def test_rational_mesh_denominator_and_tolerance_boundaries():
    valid = IntegerMesh.from_scaled([[1 / 7, 0, 0], [1 / 7 + 1 / 3, 0, 0]])
    assert valid.denominator == 21
    with pytest.raises(ValueError, match="rational"):
        IntegerMesh.from_scaled([[1 / 101, 0, 0]], max_denominator=100, tolerance=1e-10)
    IntegerMesh.from_scaled([[1 / 3 + 0.9e-8, 0, 0]], max_denominator=3, tolerance=1e-8)
    with pytest.raises(ValueError, match="rational"):
        IntegerMesh.from_scaled(
            [[1 / 3 + 1.1e-8, 0, 0]], max_denominator=3, tolerance=1e-8
        )


@pytest.mark.parametrize(
    "field",
    [
        "representatives",
        "pair_to_representative",
        "pair_operation",
        "multiplication",
        "composition_translations",
        "inverses",
        "time_reversal",
        "exchange",
        "reciprocal_wraps",
        "orbit_sizes",
    ],
)
def test_pair_metadata_cannot_be_mutated_or_unfrozen(field):
    orbits = build_pair_orbits(
        IntegerMesh.from_scaled([[0, 0, 0], [0.5, 0, 0]]), [IDENTITY]
    )
    value = getattr(orbits, field)
    with pytest.raises(ValueError):
        value.flat[0] = 7
    with pytest.raises(ValueError):
        value.setflags(write=True)
    original = np.array(value)
    copy = replace(orbits, **{field: original})
    original.flat[0] = 7
    assert getattr(copy, field).flat[0] == value.flat[0]


def test_enabled_symmetries_are_explicit_in_diagnostic_summary():
    mesh = IntegerMesh.from_scaled([[0, 0, 0], [1 / 3, 0, 0], [2 / 3, 0, 0]])
    plain = build_pair_orbits(mesh, [IDENTITY]).summary()
    reduced = build_pair_orbits(
        mesh, [IDENTITY], time_reversal=True, exchange=True
    ).summary()
    assert plain["time_reversal_enabled"] is plain["exchange_enabled"] is False
    assert reduced["time_reversal_enabled"] is reduced["exchange_enabled"] is True
    assert plain["space_group_representatives"] > reduced["space_group_representatives"]


def test_mesh_coordinates_cannot_be_unfrozen_through_orbits():
    orbits = build_pair_orbits(IntegerMesh.from_scaled([[0, 0, 0]]), [IDENTITY])
    with pytest.raises(ValueError):
        orbits.mesh.coordinates.setflags(write=True)


def test_discovered_group_at_a_nonstandard_coordinate_origin():
    from green_mbtools.mint.integral_symmetry import (
        integral_kstruct,
        matching_operation_id,
    )

    cell = gto.Cell(
        a=np.eye(3) * 4,
        atom="He .004 .008 .012",
        unit="Bohr",
        basis={"He": [[0, [1.0, 1.0]]]},
        verbose=0,
    ).build()
    structure = integral_kstruct(cell, cell.make_kpts([1, 1, 1]))
    discovered = validated_cell_operations(cell, structure)
    operations, table, _, _ = validate_group(discovered)
    assert len(operations) == 48
    assert table.shape == (48, 48)
    for operation in operations:
        matching_operation_id(operation, structure.ops)


@pytest.mark.parametrize("error", ["values", "ordering"])
def test_input_kmesh_checks_both_values_and_order(error):
    cell = gto.Cell(
        a=np.eye(3) * 4, atom="He 0 0 0", basis={"He": [[0, [1.0, 1.0]]]}, verbose=0
    ).build()
    actual = cell.make_kpts([3, 1, 1])
    scaled = cell.get_scaled_kpts(actual)
    validate_input_kmesh(cell, actual, scaled)
    wrong = scaled[::-1].copy() if error == "ordering" else scaled + 0.01
    with pytest.raises(ValueError, match="values or ordering"):
        validate_input_kmesh(cell, actual, wrong)


@pytest.mark.parametrize("kind", ["exponent", "contraction"])
def test_actual_built_shell_changes_rejected_despite_same_species_dictionary(kind):
    cell = gto.Cell(
        a=np.eye(3) * 4,
        atom="He 0 0 0; He 2 2 2",
        unit="Bohr",
        basis={"He": [[0, [1.0, 1.0], [0.3, 0.5]]]},
        verbose=0,
    ).build()
    operation = SimpleNamespace(rot=np.eye(3, dtype=int), trans=np.array([0.5] * 3))
    struct = SimpleNamespace(ops=[operation])
    validated_cell_operations(cell, struct)
    shell = cell.atom_shell_ids(1)[0]
    pointer = molecular_gto.PTR_EXP if kind == "exponent" else molecular_gto.PTR_COEFF
    start = int(cell._bas[shell, pointer])
    count = cell.bas_nprim(shell) * (
        cell.bas_nctr(shell) if kind == "contraction" else 1
    )
    values = cell._env[start : start + count].copy()
    values[0] *= 1.1
    cell._bas[shell, pointer] = len(cell._env)
    cell._env = np.concatenate([cell._env, values])
    with pytest.raises(ValueError, match="built per-atom Gaussian"):
        validated_cell_operations(cell, struct)


def diagnostic_input(path, inconsistent=None):
    cell = gto.Cell(
        a=np.eye(3) * 4, atom="He 0 0 0", basis={"He": [[0, [1.0, 1.0]]]}, verbose=0
    ).build()
    actual = cell.make_kpts([3, 1, 1])
    scaled = cell.get_scaled_kpts(actual)
    if inconsistent == "ordering":
        scaled = scaled[::-1]
    elif inconsistent == "values":
        scaled = scaled + 0.01
    with h5py.File(path, "w") as f:
        f["Cell"] = cell.dumps()
        f["symmetry/k/mesh"] = actual
        f["symmetry/k/mesh_scaled"] = scaled
        f["symmetry/pairs/kpair_irre_list"] = [0, 1, 2, 3]
        f["params/nao"] = f["params/nso"] = 1


@pytest.mark.parametrize("inconsistent", ["values", "ordering"])
def test_diagnostic_cli_rejects_inconsistent_hdf5_kmeshes(
    tmp_path, monkeypatch, inconsistent
):
    path = tmp_path / "input.h5"
    diagnostic_input(path, inconsistent)
    monkeypatch.setattr(sys, "argv", ["diagnostic", "--input", str(path)])
    with pytest.raises(ValueError, match="values or ordering"):
        runpy.run_path(
            str(
                Path(__file__).resolve().parents[1]
                / "examples/diagnose_integral_pair_orbits.py"
            ),
            run_name="__main__",
        )


@pytest.mark.parametrize("enabled", [False, True])
def test_diagnostic_cli_reports_actual_flags(tmp_path, monkeypatch, capsys, enabled):
    path = tmp_path / "input.h5"
    diagnostic_input(path)
    args = ["diagnostic", "--input", str(path)]
    if enabled:
        args += ["--time-reversal", "--exchange"]
    monkeypatch.setattr(sys, "argv", args)
    runpy.run_path(
        str(
            Path(__file__).resolve().parents[1]
            / "examples/diagnose_integral_pair_orbits.py"
        ),
        run_name="__main__",
    )
    data = json.loads(capsys.readouterr().out)
    assert data["time_reversal_enabled"] is data["exchange_enabled"] is enabled
    assert data["matched_baseline_representatives"] == (4 if enabled else 9)
