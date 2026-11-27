"""Captured auxiliary frames shared by reconstruction, archives and producers.

GREEN's default corrected q=0 reference is built with its original routine.
Metric factors are captured from the actual solve and loaded with a byte budget.
"""

from collections.abc import Mapping
from types import FunctionType, SimpleNamespace
import inspect
from importlib.metadata import version

import h5py
import numpy as np
from packaging.specifiers import SpecifierSet
from pyscf.pbc import df

from .integral_symmetry_geometry import MatrixCache
from .integral_symmetry_transform import CholeskyGauge


def build_captured_legacy_ewald(cell, kpts, auxbasis, pairs, path, mesh=None):
    """Use GREEN's exact corrected q=0 routine; capture its actual CD solve.

    Each call injects its provider subclass and SciPy proxy into a private
    function-global dictionary. No shared module or class is patched, including
    when the original routine assigns its GDF class hook internally.
    """
    import green_igen.df as legacy
    from .integral_utils import weighted_coulG_ewald
    from .integral_symmetry_compat import require_ccgdf_compatibility

    require_ccgdf_compatibility()
    supported = SpecifierSet(">=0.2.8,<0.3")
    if version("green-igen") not in supported or not isinstance(
        getattr(legacy, "_make_j3c", None), FunctionType
    ):
        raise RuntimeError(
            "Integral symmetry legacy capture requires green-igen >=0.2.8,<0.3 "
            "with a Python _make_j3c"
        )
    try:
        inspect.signature(legacy._make_j3c).bind(None, None, None, None, None)
    except TypeError as error:
        raise RuntimeError(
            "Unsupported green-igen _make_j3c capture interface"
        ) from error
    scipy_original = legacy.scipy
    captured = []
    if mesh is not None:
        mesh = np.asarray(mesh)
        if (
            mesh.shape != (3,)
            or not np.issubdtype(mesh.dtype, np.integer)
            or np.any(mesh <= 0)
        ):
            raise ValueError("mesh must contain three positive integers")

    class LinalgProxy:
        def __getattr__(self, name):
            return getattr(scipy_original.linalg, name)

        def cholesky(self, metric, *args, **kwargs):
            value = scipy_original.linalg.cholesky(metric, *args, **kwargs)
            captured.append(np.array(value, dtype=complex, copy=True))
            return value

    class CapturedGDF(legacy.GDF):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # _make_j3c uses this inner GDF's grid for the reciprocal-space sum.
            if mesh is not None:
                self.mesh = mesh.copy()
            provider.mesh = np.array(self.mesh, copy=True)
            self.weighted_coulG = weighted_coulG_ewald.__get__(self, type(self))

    provider = df.GDF(cell, kpts)
    provider.auxbasis = auxbasis
    provider.exxdiv = "ewald"
    auxcell = legacy.make_modrho_basis(cell, auxbasis, provider.exp_to_discard)
    scope = dict(
        legacy._make_j3c.__globals__,
        GDF=CapturedGDF,
        scipy=SimpleNamespace(linalg=LinalgProxy()),
    )
    routine = FunctionType(
        legacy._make_j3c.__code__,
        scope,
        legacy._make_j3c.__name__,
        legacy._make_j3c.__defaults__,
        legacy._make_j3c.__closure__,
    )
    routine.__kwdefaults__ = legacy._make_j3c.__kwdefaults__
    routine(
        provider,
        cell,
        auxcell,
        np.asarray([(kpts[i], kpts[j]) for i, j in pairs]),
        str(path),
    )
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
