"""Archive trust boundaries and allocation guards before any factor read."""

import shutil

import h5py
import numpy as np
import pytest
from pyscf.pbc import df, gto

from green_mbtools.mint.integral_symmetry import IntegerMesh
from green_mbtools.mint.integral_symmetry_geometry import PhysicalPairMap
from green_mbtools.mint.integral_symmetry_transform import CholeskyGauge
from green_mbtools.mint import integral_symmetry_io as io
from test_integral_symmetry_io import synthetic_archive as synthetic_archive


def identified_input(path, summary):
    with h5py.File(path, "w") as f:
        f["params/nk"], f["params/nso"], f["params/NQ"] = 3, 1, 1
        f["integral_symmetry/input_fingerprint"] = summary["input_fingerprint"]
        f["symmetry/pairs/kpair_idx"] = [[2, 1], [0, 0], [2, 2], [1, 0]]
        f["symmetry/pairs/kpair_irre_list"] = [0, 1, 2, 3]


@pytest.mark.parametrize("name", ["nk", "nso", "NQ"])
def test_expansion_rejects_mismatched_dimensions_before_output(
    synthetic_archive, tmp_path, name
):
    path, summary, *_ = synthetic_archive
    inp, output = tmp_path / "input.h5", tmp_path / "expanded"
    identified_input(inp, summary)
    with h5py.File(inp, "a") as f:
        f[f"params/{name}"][()] = 17
    with pytest.raises(ValueError, match="dimensions disagree"):
        io.expand_archive(path, inp, output)
    assert not output.exists()


def test_expansion_rejects_wrong_correlation_gauge_before_output(
    synthetic_archive, tmp_path
):
    original, summary, *_ = synthetic_archive
    path, inp, output = (
        tmp_path / "correlation",
        tmp_path / "input.h5",
        tmp_path / "expanded",
    )
    shutil.copytree(original, path)
    with h5py.File(path / "meta.h5", "a") as f:
        f.attrs["set_kind"] = "correlation"
    identified_input(inp, summary)
    with h5py.File(inp, "a") as f:
        f["integral_symmetry/correlation_gauge_id"] = "different-frame"
    with pytest.raises(ValueError, match="Correlation auxiliary gauge"):
        io.expand_archive(path, inp, output)
    assert not output.exists()


@pytest.mark.parametrize("kind", ["shape", "precision", "oversized"])
def test_bad_chunk_rejected_without_loading(
    synthetic_archive, tmp_path, monkeypatch, kind
):
    original, *_ = synthetic_archive
    path = tmp_path / "bad"
    shutil.copytree(original, path)
    with h5py.File(path / "SGVQ_0.h5", "w") as f:
        shape = (
            (3, 1, 1, 2)
            if kind == "precision"
            else ((3, 1, 1, 3) if kind == "shape" else (3, 1, 1, 2**24))
        )
        f.create_dataset(
            "factors",
            shape=shape,
            dtype="f4" if kind == "precision" else "f8",
            chunks=True,
        )
    get = h5py.Dataset.__getitem__

    def reject_read(dataset, key):
        assert dataset.name != "/factors", "Bad factor dataset read before validation"
        return get(dataset, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", reject_read)
    with io.ArchiveReader(path) as reader:
        with pytest.raises(ValueError, match="factor chunk"):
            reader.read(0, 0)


def test_single_factor_limit_checked_from_metadata(synthetic_archive, tmp_path):
    original, *_ = synthetic_archive
    path = tmp_path / "huge-factor"
    shutil.copytree(original, path)
    with h5py.File(path / "meta.h5", "a") as f:
        f["sg/dimensions"][2] = io.MAX_FACTOR_BUFFER_BYTES // 16 + 1
    with pytest.raises(ValueError, match="One factor exceeds"):
        io.ArchiveReader(path)


def test_expansion_capacity_bounded_before_allocation(
    synthetic_archive, tmp_path, monkeypatch
):
    path, summary, *_ = synthetic_archive
    inp, output = tmp_path / "input.h5", tmp_path / "expanded"
    identified_input(inp, summary)
    monkeypatch.setattr(io, "MAX_FACTOR_BUFFER_BYTES", 64)
    zeros, sizes = np.zeros, []

    def bounded_zeros(shape, *args, **kwargs):
        if isinstance(shape, tuple) and len(shape) == 4:
            size = int(np.prod(shape)) * np.dtype(kwargs.get("dtype", float)).itemsize
            sizes.append(size)
            assert size <= 64, "Expansion allocation exceeds source-buffer limit"
        return zeros(shape, *args, **kwargs)

    monkeypatch.setattr(io.np, "zeros", bounded_zeros)
    io.expand_archive(path, inp, output)
    assert sizes and max(sizes) <= 64
    with h5py.File(output / "meta.h5") as f:
        assert int(f["chunk_size"][()]) == 4


def test_reader_reuses_distinct_mesh_actions(synthetic_archive, monkeypatch):
    path, _, factors, _ = synthetic_archive
    action, calls = IntegerMesh.action, set()

    def once(mesh, operation, time_reversal=False):
        key = (id(mesh), operation.key, time_reversal)
        assert key not in calls, "Mesh-wide action recomputed for another pair"
        calls.add(key)
        return action(mesh, operation, time_reversal)

    monkeypatch.setattr(IntegerMesh, "action", once)
    with io.ArchiveReader(path) as reader:
        for pair, expected in factors.items():
            np.testing.assert_allclose(reader.read(*pair), expected)


@pytest.mark.parametrize("exchange", [False, True])
def test_serialization_respects_disabled_time_reversal(tmp_path, exchange):
    cell = gto.Cell(
        a=np.eye(3) * 4, atom="He 0 0 0", basis={"He": [[0, [1.0, 1.0]]]}, verbose=0
    ).build()
    aux = df.df.make_modrho_basis(cell, cell.basis)
    qpts = cell.make_kpts([3, 1, 1])
    kpts = qpts + cell.get_abs_kpts([1 / 7, 0, 0])
    pair_q = {(i, j): (j - i) % 3 for i in range(3) for j in range(3)}
    gauges = {
        i: CholeskyGauge(np.eye(1), f"shifted-q{i}", "ordinary-Coulomb")
        for i in range(3)
    }
    geometry = PhysicalPairMap(
        cell,
        aux,
        kpts,
        qpts,
        pair_q,
        gauges,
        set_id="disabled-TR",
        time_reversal=False,
        exchange=exchange,
    )
    rng = np.random.default_rng(28)
    sources = {
        tuple(p): rng.normal(size=(1, 1, 1)).astype(complex)
        for p in geometry.orbits.representatives
    }
    path = tmp_path / "shifted"
    io.write_archive(
        geometry,
        lambda i, j: sources[i, j],
        path,
        set_kind="hf",
        finite_size_kind="none",
        producer_revision="test",
        source_mode="representatives",
    )
    with io.ArchiveReader(path) as reader:
        assert not reader.attributes["time_reversal_enabled"]
        assert bool(reader.attributes["exchange_enabled"]) == exchange
        for i in range(3):
            for j in range(3):
                rep, maps = geometry.pair(i, j)
                np.testing.assert_allclose(
                    reader.read(i, j), geometry.reconstruct(sources[rep], maps)
                )
