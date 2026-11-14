"""Explicit compatibility boundary for opt-in integral-symmetry private APIs."""

import inspect

import pyscf
from packaging.version import Version


def require_pyscf_compatibility():
    version = Version(pyscf.__version__)
    if version.is_prerelease or not Version("2.14") <= version < Version("2.15"):
        raise RuntimeError(
            f"Integral symmetry requires PySCF 2.14.x (tested 2.14.0); found {version}. "
            "Legacy integral paths keep their existing package compatibility."
        )


def require_ccgdf_compatibility():
    require_pyscf_compatibility()
    from pyscf.pbc.df.gdf_builder import _CCGDFBuilder

    expected = {
        "make_j3c": ("self", "cderi_file", "kptij_lst", "aosym"),
        "gen_uniq_kpts_groups": ("self", "j_only", "h5swap", "kk_idx"),
        "decompose_j2c": ("self",),
        "get_2c2e": ("self",),
        "weighted_coulG": ("kpt", "exx", "mesh", "omega"),
    }
    for name, required in expected.items():
        method = getattr(_CCGDFBuilder, name, None)
        if method is None or not set(required) <= set(
            inspect.signature(method).parameters
        ):
            raise RuntimeError(f"Unsupported PySCF CCGDF private interface: {name}")
    return _CCGDFBuilder
