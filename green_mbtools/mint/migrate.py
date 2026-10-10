"""Migrate legacy green-mbtools input files to the current native-complex format.

Source families: versioned 0.x (grid/ datagroup, float+2 matrices) and
1.0.0 (symmetry/ datagroup, float+2). Target: 1.1.0 (native complex128).
Migration is stepwise: 0.x -> 1.0.0 -> 1.1.0. VQ integral chunk data is
never rewritten; only meta.h5 version strings and, for 0.x, the synthesized
symmetry schema change. Unversioned input must first be upgraded using
green-mbpt/python/migrate.py.

Usage:
    python -m green_mbtools.mint.migrate --input in.h5 --output out.h5 \
        [--int-path DIR ...] [--dm dm.h5] [--in-place] [--target 1.1.0] [--force]
"""
import argparse
import os
import shutil
import sys

import numpy as np
import h5py

from green_mbtools.mint.integral_utils import integrals_grid

INPUT_DATASETS = ("HF/Fock-k", "HF/S-k", "HF/H-k")
DM_DATASETS = ("HF/dm-k",)


def _legacy_complex_datasets(h5file, datasets):
    """Validate legacy storage and return datasets that need conversion."""
    legacy = []
    for name in datasets:
        if name not in h5file:
            continue
        ds = h5file[name]
        if isinstance(ds, h5py.Dataset) and np.issubdtype(ds.dtype, np.complexfloating):
            continue
        if (not isinstance(ds, h5py.Dataset)
                or ds.dtype != np.dtype(np.float64)
                or not ds.shape or ds.shape[-1] != 2
                or "__complex__" not in ds.attrs):
            raise ValueError(
                f"Unsupported legacy complex storage for {name!r} in "
                f"{h5file.filename!r}: expected float64[..., 2] with "
                "a __complex__ attribute, or a native complex dataset."
            )
        legacy.append(name)
    return legacy


def _to_native_complex(h5file, datasets):
    # Validate all candidates before replacing any dataset.
    converted = _legacy_complex_datasets(h5file, datasets)
    for name in converted:
        ds = h5file[name]
        arr = ds[()].view(np.complex128).reshape(ds.shape[:-1])
        del h5file[name]
        h5file[name] = arr
        if "__complex__" in h5file[name].attrs:
            del h5file[name].attrs["__complex__"]
    return converted


def detect_version(input_file):
    with h5py.File(input_file, "r") as f:
        return f.attrs.get("__green_version__")


def _bump_meta_version(int_paths, version):
    for d in int_paths:
        meta = os.path.join(d, "meta.h5")
        if not os.path.exists(meta):
            raise ValueError(f"Integral meta.h5 not found in {d!r}")
        with h5py.File(meta, "a") as m:
            m.attrs["__green_version__"] = version


def _v100_to_110(input_file, dm_file=None, int_paths=()):
    with h5py.File(input_file, "a") as f:
        _to_native_complex(f, INPUT_DATASETS)
        if f.attrs.get("__green_version__") != "1.1.0":
            f.attrs["__green_version__"] = "1.1.0"
    if dm_file is not None and os.path.exists(dm_file):
        with h5py.File(dm_file, "a") as f:
            _to_native_complex(f, DM_DATASETS)
            f.attrs["__green_version__"] = "1.1.0"
    _bump_meta_version(int_paths, "1.1.0")


def _synthesize_pairs(input_file, kmesh):
    """Return the five symmetry/pairs arrays, recomputed from the stored Cell.

    The cell is stored under the HDF5 key ``Cell`` as the JSON produced by
    ``pyscf``'s ``cell.dumps()`` (see ``common_utils.save_data``), so it is
    reconstructed directly with ``pyscf.pbc.gto.loads``.
    """
    import pyscf.pbc.gto
    with h5py.File(input_file, "r") as f:
        if "Cell" not in f:
            raise ValueError(
                f"Cannot synthesize symmetry/pairs for {input_file!r}: no 'Cell' "
                "dataset found. Regenerate the input file with green-mbtools."
            )
        cell_dump = f["Cell"][()]
    cell_str = cell_dump.decode() if isinstance(cell_dump, bytes) else cell_dump
    cell = pyscf.pbc.gto.loads(cell_str)
    kptij_idx, kij_conj, kij_trans, kpair_irre, num_kpair, _, _ = integrals_grid(cell, kmesh)
    return {
        "conj_pairs_list": kij_conj,
        "trans_pairs_list": kij_trans,
        "kpair_irre_list": kpair_irre,
        "kpair_idx": kptij_idx,
        "num_kpair_stored": num_kpair,
    }


def _grid_to_100(input_file, int_paths=()):
    with h5py.File(input_file, "a") as f:
        grid = f["grid"]
        index = grid["index"][()]          # bz2ibz; shape (nk_full,)
        irlist = grid["ir_list"][()]       # ibz2bz
        conj = grid["conj_list"][()]
        kmesh = grid["k_mesh"][()]
        nso = f["HF/S-k"].shape[2]
        # In a genuine 1.0.0 file HF/nk holds the full-BZ count (e.g. 27 for
        # a 3x3x3 mesh), not the per-axis dimension.  Set it to len(index).
        nk = len(index)
        f["HF/nk"][()] = nk

        k = f.require_group("symmetry/k")
        k["mesh"] = grid["k_mesh"][()]
        k["mesh_scaled"] = grid["k_mesh_scaled"][()]
        k["bz2ibz"] = index
        k["ibz2bz"] = irlist
        k["tr_conj"] = conj.astype(np.int64)
        k["weight_ibz"] = grid["weight"][()]
        k["ink"] = int(grid["ink"][()])
        k["nk"] = nk
        k["n_stars"] = len(irlist)
        stars = k.require_group("stars")
        for i, rep in enumerate(irlist):
            stars[str(i)] = np.sort(np.where(index == rep)[0]).astype(np.int64)
        k["k_sym_transform_ao"] = np.broadcast_to(
            np.eye(nso, dtype=np.complex128), (nk, nso, nso)
        ).copy()

        pairs = f.require_group("symmetry/pairs")
        for name, arr in _synthesize_pairs(input_file, kmesh).items():
            pairs[name] = np.asarray(arr)

        del f["grid"]
        f.attrs["__green_version__"] = "1.0.0"
    _bump_meta_version(int_paths, "1.0.0")


STEPS = [
    ("0.x", "1.0.0", _grid_to_100),
    ("1.0.0", "1.1.0", _v100_to_110),
]


def _apply_step(fn, path, int_paths, dm_file):
    if fn is _grid_to_100:
        fn(path, int_paths=int_paths)
    else:
        fn(path, dm_file=dm_file, int_paths=int_paths)


def migrate(input_file, output=None, int_paths=(), target="1.1.0",
            in_place=False, force=False, dm_file=None):
    """Migrate input_file to target version, returning the path of the result.

    If in_place is False, output must be provided; the input file is copied to
    output first and only the copy is modified. Pass force=True to overwrite an
    existing output or companion dm file. If in_place is True, the file is
    modified in place.

    In new-file mode (not in_place), if dm_file is provided the source dm.h5 is
    never mutated: a copy is placed beside the output file and that copy is
    migrated. The output directory must differ from the directory containing
    dm_file; if they are the same a ValueError is raised (use --in-place instead).
    """
    # Reject unsupported storage before copying outputs or modifying any files.
    current = detect_version(input_file)
    if current is None:
        raise ValueError(
            "Unversioned input is unsupported. First upgrade it to 0.2.4 "
            "using green-mbpt/python/migrate.py."
        )
    stage = "0.x" if isinstance(current, str) and current.startswith("0.") else current
    targets = [step[1] for step in STEPS]
    order = [STEPS[0][0]] + targets
    if stage not in order:
        raise ValueError(
            f"Detected version {current!r} is unsupported; expected one of {order}."
        )
    if target not in targets:
        raise ValueError(f"Unknown target version {target!r}; supported targets: {targets}")
    if order.index(stage) > order.index(target):
        raise ValueError(f"Cannot downgrade from {current} to {target}")

    with h5py.File(input_file, "r") as f:
        _legacy_complex_datasets(f, INPUT_DATASETS)
    if dm_file is not None and os.path.exists(dm_file):
        with h5py.File(dm_file, "r") as f:
            _legacy_complex_datasets(f, DM_DATASETS)

    if not in_place:
        if output is None:
            raise ValueError("Provide output=... or set in_place=True")
        if os.path.exists(output) and not force:
            raise FileExistsError(
                f"Output {output!r} exists; pass force=True to overwrite"
            )
        path = output
        # Derive an output-side dm path so the source dm is never modified.
        effective_dm = None
        if dm_file is not None:
            dm_out = os.path.join(
                os.path.dirname(os.path.abspath(output)),
                os.path.basename(dm_file),
            )
            if os.path.abspath(dm_out) == os.path.abspath(dm_file):
                raise ValueError(
                    f"Cannot migrate dm.h5 non-destructively: the output "
                    f"directory is the same as the source dm directory "
                    f"({os.path.dirname(os.path.abspath(dm_file))!r}). "
                    "Choose an output path in a different directory, or use "
                    "--in-place to migrate in place."
                )
            if os.path.exists(dm_out) and not force:
                raise FileExistsError(
                    f"Output {dm_out!r} exists; pass force=True to overwrite"
                )
            effective_dm = dm_out
        # Check both destinations before copying either file.
        shutil.copy(input_file, output)
        if effective_dm is not None:
            shutil.copy(dm_file, effective_dm)
    else:
        path = input_file
        effective_dm = dm_file

    # An already-current input can still have legacy companions.
    if stage == target == "1.1.0":
        _v100_to_110(path, dm_file=effective_dm, int_paths=int_paths)
    while stage != target:
        _, to_version, fn = next(step for step in STEPS if step[0] == stage)
        _apply_step(fn, path, int_paths, effective_dm)
        stage = to_version
    return path


def _main(argv=None):
    p = argparse.ArgumentParser(prog="python -m green_mbtools.mint.migrate",
                                description="Migrate green-mbtools input files to 1.1.0.")
    p.add_argument("--input", required=True)
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--output")
    grp.add_argument("--in-place", action="store_true")
    p.add_argument("--int-path", action="append", default=[], dest="int_paths")
    p.add_argument("--dm", dest="dm_file", default=None)
    p.add_argument("--target", choices=[step[1] for step in STEPS], default="1.1.0")
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    dm = a.dm_file
    if dm is None and not a.in_place and a.output:
        cand = os.path.join(os.path.dirname(os.path.abspath(a.input)), "dm.h5")
        dm = cand if os.path.exists(cand) else None
    out = migrate(a.input, output=a.output, int_paths=tuple(a.int_paths),
                  target=a.target, in_place=a.in_place, force=a.force, dm_file=dm)
    print(f"Migrated to {a.target}: {out}")


if __name__ == "__main__":
    _main(sys.argv[1:])
