"""Check the private PySCF interface used by the integral builders."""

import inspect


def require_ccgdf_compatibility():
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
