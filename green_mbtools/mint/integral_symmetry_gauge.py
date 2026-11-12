"""Captured auxiliary frames shared by reconstruction, archives and producers.

GREEN's default corrected q=0 reference is built with its original routine.
Metric factors are captured from the actual solve and loaded with a byte budget.
"""

from collections.abc import Mapping
from types import SimpleNamespace
import threading

import h5py
import numpy as np
from pyscf.pbc import df

from .integral_symmetry_geometry import MatrixCache
from .integral_symmetry_transform import CholeskyGauge

_legacy_capture_lock = threading.Lock()


def build_captured_legacy_ewald(cell, kpts, auxbasis, pairs, path, mesh=None):
    """Use GREEN's exact corrected q=0 routine; capture its actual CD solve.

    Only this module's scipy reference is proxied. Shared scipy functions are
    never patched. Class hooks and module ownership are restored on exceptions.
    Unrelated legacy builds must not run inside this scoped producer context.
    """
    import green_igen.df as legacy
    from .integral_utils import weighted_coulG_ewald

    if not _legacy_capture_lock.acquire(blocking=False):
        raise RuntimeError("Concurrent legacy Ewald capture contexts are unsupported")
    missing = object()
    saved = [
        (cls, cls.__dict__.get("weighted_coulG", missing))
        for cls in (df.GDF, legacy.GDF)
    ]
    scipy_original = legacy.scipy
    captured = []

    class LinalgProxy:
        def __getattr__(self, name):
            return getattr(scipy_original.linalg, name)

        def cholesky(self, metric, *args, **kwargs):
            value = scipy_original.linalg.cholesky(metric, *args, **kwargs)
            captured.append(np.array(value, dtype=complex, copy=True))
            return value

    try:
        provider = df.GDF(cell, kpts)
        provider.auxbasis = auxbasis
        provider.exxdiv = "ewald"
        if mesh is not None:
            provider.mesh = np.asarray(mesh)
        auxcell = legacy.make_modrho_basis(cell, auxbasis, provider.exp_to_discard)
        df.GDF.weighted_coulG = weighted_coulG_ewald
        legacy.scipy = SimpleNamespace(linalg=LinalgProxy())
        legacy._make_j3c(
            provider,
            cell,
            auxcell,
            np.asarray([(kpts[i], kpts[j]) for i, j in pairs]),
            str(path),
        )
    finally:
        legacy.scipy = scipy_original
        for cls, value in reversed(saved):
            if value is missing:
                if "weighted_coulG" in cls.__dict__:
                    delattr(cls, "weighted_coulG")
            else:
                cls.weighted_coulG = value
        _legacy_capture_lock.release()
    if len(captured) != 1 or captured[0].shape != (auxcell.nao_nr(),) * 2:
        raise ValueError(
            "Legacy Ewald must use one positive full-rank captured Cholesky factor"
        )
    gauge = CholeskyGauge(captured[0], "actual-legacy-Ewald-q0", "ewald-Coulomb")
    provider.auxcell = auxcell
    provider._cderi = str(path)
    return provider, gauge


class CapturedGauges(Mapping):
    def __init__(self, path, count, kernel, budget=64 * 1024**2, dataset_prefix="C"):
        self.path, self.count, self.kernel = str(path), count, kernel
        self.dataset_prefix = dataset_prefix
        self.cache = MatrixCache(budget)

    def __len__(self):
        return self.count

    def __iter__(self):
        return iter(range(self.count))

    def __contains__(self, key):
        return isinstance(key, (int, np.integer)) and 0 <= key < self.count

    def __getitem__(self, key):
        if key not in self:
            raise KeyError(key)
        identity = f"{self.path}:q{key}"

        def load():
            with h5py.File(self.path) as file:
                matrix = file[f"{self.dataset_prefix}/{key}"][()]
            return CholeskyGauge(matrix, identity, self.kernel).unwhitening

        return SimpleNamespace(
            unwhitening=self.cache.get(key, load),
            gauge_id=identity,
            kernel_id=self.kernel,
        )
