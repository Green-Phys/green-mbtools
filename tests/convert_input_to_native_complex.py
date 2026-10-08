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

import h5py

from green_mbtools.mint.migrate import _to_native_complex, INPUT_DATASETS, DM_DATASETS


def convert_file(path: str, datasets: Sequence[str], version: str = "1.1.0") -> list[str]:
    with h5py.File(path, "a") as f:
        converted = _to_native_complex(f, datasets)
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
