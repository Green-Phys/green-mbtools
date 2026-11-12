import numpy as np
import pytest
from pyscf.pbc import gto
from green_mbtools.mint.integral_symmetry import (
    integral_kstruct,
    validated_cell_operations,
)
from green_mbtools.mint.integral_symmetry_geometry import MatrixCache


def test_cache_bound_eviction_set_identity_and_owned_values():
    cache = MatrixCache(128)
    supplied = np.eye(2, dtype=complex)
    first = cache.get(("hf", 0), lambda: supplied)
    supplied[0, 0] = 3
    assert first[0, 0] == 1 and not first.flags.writeable
    cache.get(("correlation", 0), lambda: 2 * supplied)
    assert cache.get(("hf", 0), lambda: None) is first
    cache.get(("hf", 1), lambda: supplied)
    assert ("correlation", 0) not in cache.items
    assert (
        cache.bytes <= cache.budget
        and cache.peak <= cache.budget
        and cache.evictions == 1
    )


def test_zero_budget_does_not_retain_maps():
    cache = MatrixCache(0)
    cache.get("matrix", lambda: np.eye(2))
    assert cache.bytes == 0 and not cache.items
    with pytest.raises(ValueError):
        MatrixCache(-1)


def test_integral_group_includes_translations_without_changing_one_body_or_fft():
    cell = gto.Cell()
    cell.a = 3.5 * np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    cell.atom = [["C", [0, 0, 0]], ["C", [0.875, 0.875, 0.875]]]
    cell.basis = {"C": [[0, [1.0, 1.0]], [1, [0.6, 1.0]]]}
    cell.mesh = [31] * 3
    cell.verbose = 0
    cell.build()
    previous = getattr(cell, "lattice_symmetry", None)
    k = cell.make_kpts([3, 3, 3])
    struct = integral_kstruct(cell, k)
    operations = validated_cell_operations(cell, struct)
    assert len(operations) == 48
    assert sum(any(op.translation) for op in operations) == 24
    assert getattr(cell, "lattice_symmetry", None) is previous
    assert np.array_equal(cell.mesh, [31] * 3)
