"""Small, always-on physical PySCF integration; no external fixture files."""

from itertools import product

import numpy as np
import pytest
from pyscf.pbc import df, gto

from green_mbtools.mint.integral_symmetry import IDENTITY
from green_mbtools.mint.integral_symmetry_compat import require_ccgdf_compatibility
from green_mbtools.mint.integral_symmetry_geometry import PhysicalPairMap
from green_mbtools.mint.integral_symmetry_transform import CholeskyGauge


@pytest.fixture(scope="module")
def physical_pair_reference(tmp_path_factory):
    cell = gto.Cell(
        a=[[0, 1.75, 1.75], [1.75, 0, 1.75], [1.75, 1.75, 0]],
        atom="C 0 0 0; C .875 .875 .875",
        unit="Angstrom",
        basis={"C": [[0, [1.0, 1.0]], [1, [0.6, 1.0]]]},
        mesh=[31] * 3,
        precision=1e-12,
        verbose=0,
        max_memory=1000,
    ).build()
    aux = df.df.make_modrho_basis(
        cell, {"C": [[0, [0.8, 1.0]], [0, [0.25, 1.0]], [1, [0.6, 1.0]]]}
    )
    kpts = cell.make_kpts([3, 1, 1])

    class CompleteBuilder(require_ccgdf_compatibility()):
        def gen_uniq_kpts_groups(self, *args, **kwargs):
            for q, pairs, decomposition in super().gen_uniq_kpts_groups(
                *args, **kwargs
            ):
                factor, negative, tag = decomposition
                assert tag == "CD" and negative is None
                assert factor.shape == (aux.nao_nr(),) * 2
                self.captured.append((np.array(q), np.array(pairs), factor.copy()))
                yield q, pairs, decomposition

    builder = CompleteBuilder(cell, aux, kpts)
    builder.mesh, builder.j2c_eig_always, builder.captured = (
        np.array(cell.mesh),
        False,
        [],
    )
    path = str(tmp_path_factory.mktemp("physical-pair") / "complete.h5")
    builder.make_j3c(
        path, kptij_lst=np.array(list(product(kpts, repeat=2))), aosym="s1"
    )
    provider = df.GDF(cell, kpts)
    provider.auxcell, provider._cderi = aux, path
    factors = {}
    for i, j in product(range(3), repeat=2):
        blocks = []
        for real, imag, sign in provider.sr_loop((kpts[i], kpts[j]), compact=False):
            assert sign == 1
            blocks.append(real + 1j * imag)
        factors[i, j] = np.concatenate(blocks).reshape(
            aux.nao_nr(), cell.nao_nr(), cell.nao_nr()
        )
    pair_q, gauges, qpts = {}, {}, []
    for iq, (q, pairs, factor) in enumerate(builder.captured):
        qpts.append(q)
        gauges[iq] = CholeskyGauge(factor, f"complete-q{iq}", "ordinary-Coulomb")
        for pair in pairs:
            pair_q[divmod(int(pair), 3)] = iq
    return cell, aux, kpts, np.array(qpts), pair_q, gauges, factors


@pytest.mark.parametrize("tr,exchange", list(product((False, True), repeat=2)))
def test_physical_pair_map_full_chain_all_tr_exchange_cases(
    physical_pair_reference, tr, exchange
):
    cell, aux, kpts, qpts, pair_q, gauges, factors = physical_pair_reference
    geometry = PhysicalPairMap(
        cell,
        aux,
        kpts,
        qpts,
        pair_q,
        gauges,
        set_id="physical-integration",
        time_reversal=tr,
        exchange=exchange,
    )
    overlap = cell.pbc_intor("int1e_ovlp", kpts=kpts)
    xs, inverses = [], []
    for s in overlap:
        values, vectors = np.linalg.eigh(s)
        xs.append((vectors / np.sqrt(values)) @ vectors.conj().T)
        inverses.append((vectors * np.sqrt(values)) @ vectors.conj().T)
    spatial_seen = phase_seen = False
    for op in geometry.orbits.operations:
        spatial_seen |= op != IDENTITY
        for source, factor in factors.items():
            target, maps = geometry.spatial(source, op, tr=tr, exchange=exchange)
            assert maps[3] == (tr ^ exchange)
            phase_seen |= bool(np.max(np.abs(maps[1].imag)) > 1e-5)
            actual = geometry.reconstruct(factor, maps)
            np.testing.assert_allclose(actual, factors[target], atol=1e-10, rtol=1e-10)
            sliced = geometry.reconstruct(factor, maps, output_slice=(1, 2))
            np.testing.assert_allclose(
                sliced, factors[target][1:3], atol=1e-10, rtol=1e-10
            )
            stored = np.einsum(
                "im,Qmn,jn->Qij", xs[source[0]], factor, xs[source[1]].conj()
            )
            actual_stored = geometry.reconstruct(
                stored,
                maps,
                source_x_inverse=(inverses[source[0]], inverses[source[1]]),
                target_x=(xs[target[0]], xs[target[1]]),
            )
            expected = np.einsum(
                "im,Qmn,jn->Qij", xs[target[0]], factors[target], xs[target[1]].conj()
            )
            np.testing.assert_allclose(actual_stored, expected, atol=1e-10, rtol=1e-10)
    assert spatial_seen and phase_seen
    for target, expected in factors.items():
        representative, maps = geometry.pair(*target)
        np.testing.assert_allclose(
            geometry.reconstruct(factors[representative], maps),
            expected,
            atol=1e-10,
            rtol=1e-10,
        )


def test_physical_cross_pair_eri_detects_fault_before_reconstruction(
    physical_pair_reference,
):
    cell, aux, kpts, qpts, pair_q, gauges, factors = physical_pair_reference
    geometry = PhysicalPairMap(
        cell, aux, kpts, qpts, pair_q, gauges, set_id="fault-injection"
    )
    pairs = [p for p in factors if pair_q[p] == pair_q[0, 0]]
    left, right = pairs[:2]
    representative, maps = geometry.pair(*right)
    faulty_maps = (np.exp(0.3j) * maps[0],) + maps[1:]
    wrong = geometry.reconstruct(factors[representative], faulty_maps).reshape(
        aux.nao_nr(), -1
    )
    correct = factors[right].reshape(aux.nao_nr(), -1)
    anchor = factors[left].reshape(aux.nao_nr(), -1)
    np.testing.assert_allclose(
        wrong.conj().T @ wrong, correct.conj().T @ correct, atol=1e-10, rtol=1e-10
    )
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            anchor.conj().T @ wrong, anchor.conj().T @ correct, atol=1e-10, rtol=1e-10
        )
