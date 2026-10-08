"""Migrate legacy green-mbtools input files to the current native-complex format.

Source families: grid-legacy (pre-1.0.0, grid/ datagroup, float+2 matrices) and
1.0.0 (symmetry/ datagroup, float+2). Target: 1.1.0 (native complex128).
Migration is stepwise: grid-legacy -> 1.0.0 -> 1.1.0. VQ integral chunk data is
never rewritten; only meta.h5 version strings and, for grid-legacy, the
synthesized symmetry schema change.

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


def _to_native_complex(h5file, datasets):
    converted = []
    for name in datasets:
        if name not in h5file:
            continue
        ds = h5file[name]
        if np.iscomplexobj(np.empty(0, dtype=ds.dtype)):
            continue
        arr = ds[()].view(np.complex128).reshape(ds.shape[:-1])
        del h5file[name]
        h5file[name] = arr
        if "__complex__" in h5file[name].attrs:
            del h5file[name].attrs["__complex__"]
        converted.append(name)
    return converted


def detect_version(input_file):
    with h5py.File(input_file, "r") as f:
        if "symmetry" in f:
            s = f["HF/S-k"]
            if np.iscomplexobj(np.empty(0, dtype=s.dtype)):
                return "1.1.0"
            return "1.0.0"
        if "grid" in f:
            return "grid-legacy"
    raise ValueError(
        f"Unrecognized input.h5 structure in {input_file!r}: "
        "no 'symmetry' or 'grid' datagroup found."
    )


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
        f.attrs["__green_version__"] = "1.1.0"
    if dm_file is not None and os.path.exists(dm_file):
        with h5py.File(dm_file, "a") as f:
            _to_native_complex(f, DM_DATASETS)
            f.attrs["__green_version__"] = "1.1.0"
    _bump_meta_version(int_paths, "1.1.0")


def _synthesize_pairs(input_file, kmesh):
    """Return the five symmetry/pairs arrays, recomputed from the stored Cell."""
    from pyscf.pbc.lib.chkfile import load_cell
    cell = None
    with h5py.File(input_file, "r") as f:
        has_cell = "Cell" in f
    if has_cell:
        # Try pyscf's load_cell first (expects 'mol' key); fall back to the
        # legacy 'Cell' key (pre-1.0.0 files store the cell JSON under 'Cell').
        try:
            cell = load_cell(input_file)
        except Exception:
            cell = None
        if cell is None:
            try:
                import pyscf.pbc.gto
                with h5py.File(input_file, "r") as f:
                    cell_bytes = f["Cell"][()]
                cell_str = cell_bytes.decode() if isinstance(cell_bytes, bytes) else cell_bytes
                cell = pyscf.pbc.gto.loads(cell_str)
            except Exception:
                cell = None
    if cell is None:
        raise ValueError(
            f"Cannot synthesize symmetry/pairs for {input_file!r}: no 'Cell' "
            "found and no meta.h5 fallback available. Regenerate the input file."
        )
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
        # HF/nk is the per-axis Monkhorst-Pack dimension (e.g. 3 for 3x3x3),
        # NOT the full-BZ count. Derive nk from grid/index which has one entry
        # per full-BZ k-point.
        nk = len(index)

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
