import shutil
import h5py
import numpy as np
import pytest
from pyscf.pbc import gto, df
from green_mbtools.mint.integral_symmetry_geometry import PhysicalPairMap
from green_mbtools.mint.integral_symmetry_transform import CholeskyGauge
from green_mbtools.mint.integral_symmetry_io import (
    ArchiveReader,
    write_archive,
    expand_archive,
    unpack,
)


@pytest.fixture(scope="module")
def synthetic_archive(tmp_path_factory):
    root = tmp_path_factory.mktemp("sg-schema")
    cell = gto.Cell()
    cell.a = np.eye(3) * 4
    cell.atom = "He 0 0 0"
    cell.basis = {"He": [[0, [1.0, 1.0]]]}
    cell.verbose = 0
    cell.build()
    aux = df.df.make_modrho_basis(cell, cell.basis)
    kpts = cell.make_kpts([3, 1, 1])
    qpts = kpts.copy()
    pair_q = {(i, j): (j - i) % 3 for i in range(3) for j in range(3)}
    gauges = {
        i: CholeskyGauge(np.eye(1), f"synthetic-q{i}", "ordinary-Coulomb")
        for i in range(3)
    }
    geometry = PhysicalPairMap(
        cell, aux, kpts, qpts, pair_q, gauges, set_id="synthetic-storage-only"
    )
    random = np.random.default_rng(7)
    representatives = {
        tuple(map(int, p)): random.normal(size=(1, 1, 1)).astype(complex)
        for p in geometry.orbits.representatives
    }
    factors = {}
    for i in range(3):
        for j in range(3):
            rep, maps = geometry.pair(i, j)
            factors[i, j] = geometry.reconstruct(representatives[rep], maps)
    path = root / "archive"
    summary = write_archive(
        geometry,
        lambda i, j: factors[i, j],
        path,
        set_kind="hf",
        finite_size_kind="none",
        producer_revision="synthetic",
        chunk_size=3,
    )
    return path, summary, factors, geometry


def test_roundtrip_sparse_maps_slice_eviction_final_chunk_and_no_legacy_alias(
    synthetic_archive,
):
    path, summary, factors, _ = synthetic_archive
    with h5py.File(path / "meta.h5") as f:
        assert "chunk_size" not in f and "__green_version__" not in f.attrs
        assert f["sg/chunk_valid_count"][-1] == 1
    assert not list(path.glob("VQ_*.h5"))
    with ArchiveReader(
        path,
        expected_fingerprint=summary["input_fingerprint"],
        expected_set_kind="hf",
        cache_bytes=32,
    ) as r:
        for pair, expected in reversed(list(factors.items())):
            actual = r.read(*pair, output_slice=(0, 1))
            assert np.allclose(actual, expected)
            actual[:] = 100
            assert np.allclose(r.read(*pair), expected)
        assert r.cache.bytes <= 32 and r.cache.evictions > 0


def test_representative_source_never_requests_omitted_factors(
    synthetic_archive, tmp_path
):
    _, _, factors, geometry = synthetic_archive
    representatives = set(map(tuple, geometry.orbits.representatives))

    def source(i, j):
        assert (i, j) in representatives
        return factors[i, j]

    path = tmp_path / "representatives"
    summary = write_archive(
        geometry,
        source,
        path,
        set_kind="hf",
        finite_size_kind="none",
        producer_revision="synthetic",
        source_mode="representatives",
        chunk_size=3,
    )
    assert (
        summary["source_build_time_saved"] is False
    )  # serialization alone does not establish a build saving
    with ArchiveReader(path) as reader:
        for pair, factor in factors.items():
            assert np.allclose(reader.read(*pair), factor)


@pytest.mark.parametrize(
    "field,value",
    [
        ("format_version", 2),
        ("complex_storage", "float32"),
        ("integral_format", "unknown"),
        ("operation_order", "exchange,spatial"),
        ("kernel_id", ""),
        ("kernel_id", "unknown"),
    ],
)
def test_reject_unknown_or_incomplete_contract(
    synthetic_archive, tmp_path, field, value
):
    src, *_ = synthetic_archive
    path = tmp_path / "bad"
    shutil.copytree(src, path)
    with h5py.File(path / "meta.h5", "a") as f:
        f.attrs[field] = value
    with pytest.raises(ValueError):
        with ArchiveReader(path) as reader:
            reader.read(0, 1)


@pytest.mark.parametrize(
    "dataset,index,value",
    [
        ("pair_operation", (0, 1), 999),
        ("pair_to_representative", (0, 1), 999),
        ("pair_q", (0, 1), 999),
        ("time_reversal", (0, 1), 2),
        ("reciprocal_wraps", (0, 1, 0, 0), 999),
        ("chunk_valid_count", (-1,), 3),
    ],
)
def test_reject_corrupt_ids_wraps_and_last_count(
    synthetic_archive, tmp_path, dataset, index, value
):
    src, *_ = synthetic_archive
    path = tmp_path / "bad"
    shutil.copytree(src, path)
    with h5py.File(path / "meta.h5", "a") as f:
        f[f"sg/{dataset}"][index] = value
    with pytest.raises(ValueError):
        ArchiveReader(path)


def test_reject_missing_operations_bad_rank_fingerprint_and_set(
    synthetic_archive, tmp_path
):
    path, summary, *_ = synthetic_archive
    with pytest.raises(ValueError, match="fingerprint"):
        ArchiveReader(path, expected_fingerprint="wrong-input")
    with pytest.raises(ValueError, match="Wrong HF"):
        ArchiveReader(path, expected_set_kind="correlation")
    bad = tmp_path / "bad"
    shutil.copytree(path, bad)
    with h5py.File(bad / "meta.h5", "a") as f:
        del f["sg/pair_operation"]
    with pytest.raises((ValueError, KeyError)):
        ArchiveReader(bad)
    with pytest.raises(ValueError):
        unpack(np.zeros((2, 2, 2)), (2, 2))


def test_expand_preserves_original_pair_order_and_final_padding(
    synthetic_archive, tmp_path
):
    path, summary, factors, _ = synthetic_archive
    inp = tmp_path / "input.h5"
    pairs = np.array([[2, 1], [0, 0], [2, 2], [1, 0]])
    with h5py.File(inp, "w") as f:
        f["symmetry/pairs/kpair_idx"] = pairs
        f["symmetry/pairs/kpair_irre_list"] = [0, 1, 2, 3]
        f["integral_symmetry/input_fingerprint"] = summary["input_fingerprint"]
    output = tmp_path / "expanded"
    expand_archive(path, inp, output, chunk_size=3)
    for start in (0, 3):
        with h5py.File(output / f"VQ_{start}.h5") as f:
            v = unpack(f[str(start)][()], (3, 1, 1, 1))
        for index, pair in enumerate(pairs[start : start + 3]):
            assert np.allclose(v[index], factors[tuple(pair)])
        if start == 3:
            assert np.all(v[1:] == 0)
    with pytest.raises(ValueError):
        expand_archive(path, inp, output)


def test_wrong_complete_factor_prevents_omission_and_output_creation(
    synthetic_archive, tmp_path
):
    _, _, factors, geometry = synthetic_archive
    altered = {pair: v.copy() for pair, v in factors.items()}
    altered[2, 1] += 1
    with pytest.raises(ValueError, match="Cannot omit"):
        write_archive(
            geometry,
            lambda i, j: altered[i, j],
            tmp_path / "unsafe",
            set_kind="hf",
            finite_size_kind="none",
            producer_revision="synthetic",
        )
    assert not (tmp_path / "unsafe").exists()


@pytest.mark.parametrize("sidecar", ["df_ewald.h5", "AqQ.h5"])
def test_special_sidecars_are_not_silently_dropped(
    synthetic_archive, tmp_path, sidecar
):
    source, *_ = synthetic_archive
    path = tmp_path / "sidecar"
    shutil.copytree(source, path)
    (path / sidecar).touch()
    with pytest.raises(ValueError, match="Special finite-size"):
        ArchiveReader(path)


def test_kernel_profile_mismatch_is_rejected(synthetic_archive, tmp_path):
    source, *_ = synthetic_archive
    path = tmp_path / "mismatch"
    shutil.copytree(source, path)
    with h5py.File(path / "meta.h5", "a") as file:
        file.attrs["finite_size_kind"] = "ewald"
    with pytest.raises(ValueError, match="profile differs"):
        ArchiveReader(path)
