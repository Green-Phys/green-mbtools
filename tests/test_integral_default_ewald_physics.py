"""Opt-in physical regression, intended for a provisioned compute workstation.

Run with GREEN_RUN_PHYSICAL_SG_TESTS=1 and a compute-node pytest --basetemp.
The complete corrected reference calls the original GREEN routine directly,
without the symmetry producer's capture wrapper or its representative subset.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import h5py
import numpy as np
import pytest
from pyscf.pbc import df, gto
from pyscf.pbc.df.gdf_builder import _CCGDFBuilder
import green_igen.df as legacy

from green_mbtools.mint.integral_symmetry_builder import build_representative_archive
from green_mbtools.mint.integral_symmetry import (
    integral_kstruct,
    validated_cell_operations,
)
from green_mbtools.mint.integral_symmetry_io import ArchiveReader, pack, unpack
from green_mbtools.mint.integral_utils import integrals_grid, weighted_coulG_ewald


def _factor(provider, left, right, naux, nao):
    blocks = []
    for real, imag, sign in provider.sr_loop((left, right), compact=False):
        assert sign == 1
        blocks.append(real + 1j * imag)
    return np.concatenate(blocks).reshape(naux, nao, nao)


@pytest.mark.skipif(
    os.environ.get("GREEN_RUN_PHYSICAL_SG_TESTS") != "1",
    reason="Explicit workstation opt-in required for physical integral builds",
)
@pytest.mark.parametrize("model", ["diamond", "identity"])
def test_default_ewald_against_independent_complete_green_reference(
    tmp_path, monkeypatch, model
):
    root = Path(
        os.environ.get("GREEN_SG_PHYSICS_FIXTURE_DIR", str(tmp_path / "physics"))
    )
    root = root / model
    root.mkdir(parents=True, exist_ok=False)
    cell = gto.Cell()
    if model == "diamond":
        cell.a = [[0, 1.75, 1.75], [1.75, 0, 1.75], [1.75, 1.75, 0]]
        cell.atom = [["C", [0, 0, 0]], ["C", [0.875, 0.875, 0.875]]]
    else:
        cell.a = [[4.1, 0.2, 0.1], [0.4, 4.4, 0.3], [0.2, 0.5, 4.7]]
        cell.atom = [["C", [0, 0, 0]], ["He", [0.83, 1.17, 1.43]]]
    cell.unit = "Angstrom"
    cell.basis = {"C": [[0, [1.0, 1.0]], [1, [0.6, 1.0]]]}
    if model == "identity":
        cell.basis["He"] = [[0, [1.2, 1.0]]]
    cell.precision = 1e-12
    cell.mesh = [31] * 3
    cell.max_memory = 4000
    cell.verbose = 0
    cell.build()
    kpts = cell.make_kpts([3, 1, 1])
    auxbasis = {"C": [[0, [0.8, 1.0]], [0, [0.25, 1.0]], [1, [0.6, 1.0]]]}
    if model == "identity":
        auxbasis["He"] = [[0, [0.9, 1.0]], [0, [0.35, 1.0]]]
    aux = df.df.make_modrho_basis(cell, auxbasis)
    nk, nao, naux = len(kpts), cell.nao_nr(), aux.nao_nr()
    operations = validated_cell_operations(cell, integral_kstruct(cell, kpts))
    legacy_representatives = int(integrals_grid(cell, kpts)[4])
    if model == "identity":
        assert len(operations) == 1, (
            "The physical low-symmetry model must have only identity spatial symmetry"
        )

    # Complete ordinary reference from the upstream builder, not the selected
    # representative builder. Supplying every ordered pair fixes the q frames.
    builder = _CCGDFBuilder(cell, aux, kpts)
    builder.mesh = np.array(cell.mesh)
    builder.j2c_eig_always = False
    ordinary_path = root / "complete-ordinary-cderi.h5"
    builder.make_j3c(
        str(ordinary_path),
        kptij_lst=np.array([(i, j) for i in kpts for j in kpts]),
        aosym="s1",
    )
    ordinary = df.GDF(cell, kpts)
    ordinary.auxcell = aux
    ordinary._cderi = str(ordinary_path)

    # This is deliberately independent of build_captured_legacy_ewald. It
    # exercises all diagonal pairs, uses the original module/scipy references,
    # and restores the temporary kernel hook before building the SG archive.
    corrected = df.GDF(cell, kpts)
    corrected.auxbasis = auxbasis
    corrected.mesh = np.array(cell.mesh)
    corrected.exxdiv = "ewald"
    corrected_aux = legacy.make_modrho_basis(cell, auxbasis, corrected.exp_to_discard)
    corrected_path = root / "complete-green-ewald-cderi.h5"
    with monkeypatch.context() as context:
        context.setattr(df.GDF, "weighted_coulG", weighted_coulG_ewald)
        legacy._make_j3c(
            corrected,
            cell,
            corrected_aux,
            np.array([(point, point) for point in kpts]),
            str(corrected_path),
        )
    corrected.auxcell = corrected_aux
    corrected._cderi = str(corrected_path)
    assert corrected_aux.nao_nr() == naux

    references = {"hf": {}, "correlation": {}}
    for i in range(nk):
        for j in range(nk):
            value = _factor(ordinary, kpts[i], kpts[j], naux, nao)
            references["hf"][i, j] = value
            references["correlation"][i, j] = (
                _factor(corrected, kpts[i], kpts[j], naux, nao) if i == j else value
            )
    bare = references["hf"][0, 0].reshape(naux, -1)
    corr = references["correlation"][0, 0].reshape(naux, -1)
    q0_change = float(np.linalg.norm(corr.conj().T @ corr - bare.conj().T @ bare))
    assert q0_change > 1e-5, "The default corrected q=0 interaction was not exercised"

    repository = Path(__file__).resolve().parents[1]
    revision = os.environ.get("GREEN_SG_SOURCE_REVISION") or subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    report = {
        "status": "PASS",
        "model": model,
        "detected_spatial_operations": len(operations),
        "legacy_representatives": legacy_representatives,
        "full_pairs": nk * nk,
        "precision": cell.precision,
        "mesh": list(map(int, cell.mesh)),
        "atol": 1e-10,
        "rtol": 1e-10,
        "reference_corrected_producer": "direct green_igen.df._make_j3c, all diagonal pairs",
        "reference_module_sha256": hashlib.sha256(
            Path(legacy.__file__).read_bytes()
        ).hexdigest(),
        "q0_ERI_change_norm": q0_change,
        "source_revision": revision,
        "sets": {},
    }
    for kind in ("hf", "correlation"):
        folder = root / kind
        folder.mkdir()
        manifest = build_representative_archive(
            cell,
            kpts,
            auxbasis,
            folder / "sg",
            folder / "work",
            corrected=kind == "correlation",
            producer_revision=revision,
            mesh=cell.mesh,
            chunk_size=2,
        )
        if model == "identity":
            assert manifest["representatives"] == legacy_representatives, (
                "Identity spatial symmetry must add no reduction beyond TR/exchange"
            )
        rebuilt = {}
        maximum = 0.0
        with ArchiveReader(folder / "sg", expected_set_kind=kind) as reader:
            for pair, reference in references[kind].items():
                value = reader.read(*pair)
                assert np.linalg.norm(reference) > 1e-10
                np.testing.assert_allclose(
                    value,
                    reference,
                    atol=1e-10,
                    rtol=1e-10,
                    err_msg=f"{kind} complete V{pair}",
                )
                maximum = max(maximum, float(np.max(np.abs(value - reference))))
                rebuilt[pair] = value
            pair_q = reader.pair_q.copy()
            attributes = dict(reader.attributes)
        eri_maximum = 0.0
        for iq in np.unique(pair_q):
            pairs = sorted(pair for pair in rebuilt if pair_q[pair] == iq)
            direct = np.concatenate(
                [references[kind][p].reshape(naux, -1) for p in pairs], axis=1
            )
            actual = np.concatenate(
                [rebuilt[p].reshape(naux, -1) for p in pairs], axis=1
            )
            expected_eri = direct.conj().T @ direct
            actual_eri = actual.conj().T @ actual
            np.testing.assert_allclose(actual_eri, expected_eri, atol=1e-10, rtol=1e-10)
            eri_maximum = max(
                eri_maximum, float(np.max(np.abs(actual_eri - expected_eri)))
            )
        packed = np.array(
            [[references[kind][i, j] for j in range(nk)] for i in range(nk)]
        )
        with h5py.File(folder / "native-reference.h5", "w") as file:
            file["factors"] = pack(packed)
        # Complete factors are from the independent reference. Captured frame
        # metadata is exported only after their entrywise agreement is checked.
        with h5py.File(folder / "sg/meta.h5") as meta:
            qpts = cell.get_abs_kpts(
                meta["sg/q_mesh"][()] / int(meta["sg/q_denominator"][()])
            )
            frames = unpack(meta["sg/captured_C"][()], (len(qpts), naux, naux))
        with h5py.File(folder / "complete-reference.h5", "w") as file:
            file.attrs.update(set_kind=kind, kernel_id=attributes["kernel_id"])
            file["Cell"], file["AuxCell"] = cell.dumps(), aux.dumps()
            file["kpts"], file["qpts"] = kpts, qpts
            for iq, frame in enumerate(frames):
                file[f"captured_C/{iq}"] = frame
            for (i, j), value in references[kind].items():
                file[f"factors/{i}_{j}"] = value
                file[f"pair_q/{i}_{j}"] = pair_q[i, j]
        if kind == "hf":
            shutil.copyfile(ordinary_path, folder / "cderi.h5")
        with h5py.File(folder / "native-input.h5", "w") as file:
            file["integral_symmetry/input_fingerprint"] = manifest["input_fingerprint"]
            file["integral_symmetry/correlation_gauge_id"] = attributes[
                "auxiliary_gauge_id"
            ]
            file["symmetry/k/mesh_scaled"] = cell.get_scaled_kpts(kpts)
            for name, value in (("nao", nao), ("nso", nao), ("NQ", naux)):
                file[f"params/{name}"] = np.int64(value)
        report["sets"][kind] = {
            "V_max_absolute": maximum,
            "cross_pair_ERI_max_absolute": eri_maximum,
            "representatives": manifest["representatives"],
            "three_center_pairs": manifest["three_center_pairs"],
        }
    (root / "physical-regression.json").write_text(json.dumps(report, indent=2) + "\n")
