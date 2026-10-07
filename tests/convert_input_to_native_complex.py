"""Convert legacy float+2 complex datasets in input.h5 / dm.h5 to native complex128.

Legacy green-mbtools (<= 1.0) stored HF/Fock-k, HF/S-k, HF/H-k and HF/dm-k as
float64 arrays with a trailing size-2 axis plus a ``__complex__`` attribute.
green-mbtools 1.1 stores them as native complex128 (an HDF5 compound {r,i}
type, the same representation h5py, green-h5pp and sim.h5 already use).

Run on a directory tree of fixtures or on user data::

    python convert_input_to_native_complex.py /path/to/dir
"""
import os
import sys
from typing import Sequence

import numpy as np
import h5py

INPUT_DATASETS = ("HF/Fock-k", "HF/S-k", "HF/H-k")
DM_DATASETS = ("HF/dm-k",)


def _is_legacy(ds) -> bool:
    """True if ds is stored the legacy float+2 way."""
    if np.iscomplexobj(np.empty(0, dtype=ds.dtype)):
        return False
    return bool(ds.attrs.get("__complex__", 0)) or (
        ds.ndim >= 1 and ds.shape[-1] == 2
    )


def convert_file(path: str, datasets: Sequence[str], version: str = "1.1.0") -> list:
    converted = []
    with h5py.File(path, "a") as f:
        for name in datasets:
            if name not in f:
                continue
            ds = f[name]
            if not _is_legacy(ds):
                continue
            arr = ds[()].view(np.complex128).reshape(ds.shape[:-1])
            del f[name]
            f[name] = arr
            if "__complex__" in f[name].attrs:
                del f[name].attrs["__complex__"]
            converted.append(name)
        f.attrs["__green_version__"] = version
    return converted


def main(root: str) -> None:
    for dirpath, _dirs, files in os.walk(root):
        for fname in files:
            full = os.path.join(dirpath, fname)
            if fname == "input.h5":
                print(full, convert_file(full, INPUT_DATASETS))
            elif fname == "dm.h5":
                print(full, convert_file(full, DM_DATASETS))


if __name__ == "__main__":
    main(sys.argv[1])
