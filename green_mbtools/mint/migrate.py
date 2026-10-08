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
