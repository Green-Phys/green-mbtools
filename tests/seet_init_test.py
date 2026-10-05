"""Tests for ``green_mbtools.mint.seet_init.get_input_data``.

Covers the 1.0.0 ``symmetry/k`` input.h5 layout: k-mesh and the
space-group IBZ -> full-BZ expansion (conjugation + AO symmetry rotation).
"""

import os
import types

import h5py
import numpy as np
import pytest

from green_mbtools.mint.seet_init import seet_init


@pytest.fixture
def h2_gw_dir():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_data", "H2_GW"
    )


@pytest.fixture
def args(h2_gw_dir):
    return types.SimpleNamespace(
        input_file=os.path.join(h2_gw_dir, "input.h5"),
        gf2_input_file=os.path.join(h2_gw_dir, "sim.h5"),
    )


def test_get_input_data_expands_to_full_bz(args):
    seet = seet_init(args)
    F, S, T, dm, dm_s, kmesh, kmesh_sc = seet.get_input_data()

    nk = 27
    assert kmesh.shape == (nk, 3)
    assert kmesh_sc.shape == (nk, 3)

    # Everything k-resolved must live on the full BZ after expansion.
    assert F.shape[1] == nk
    assert S.shape[1] == nk
    assert T.shape[1] == nk
    assert dm_s.shape[1] == nk
    assert dm.shape[0] == nk

    # Spin-averaged dm must be consistent with the spin-resolved one.
    assert np.allclose(dm, 0.5 * (dm_s[0] + dm_s[1]))


def test_full_bz_matches_ibz_at_representatives(args):
    """At each IBZ representative the symmetry op is identity and there is
    no conjugation, so the expanded Fock equals H0 + the raw IBZ Sigma1."""
    seet = seet_init(args)
    F, S, T, dm, dm_s, kmesh, kmesh_sc = seet.get_input_data()

    with h5py.File(args.input_file, "r") as f:
        ibz2bz = f["symmetry/k/ibz2bz"][()]
    with h5py.File(args.gf2_input_file, "r") as f:
        it = f["iter"][()]
        S1 = f["iter{}/Sigma1".format(it)][()].view(np.complex128)
        if S1.ndim == 5:
            S1 = S1.reshape(S1.shape[:-1])

    for i, ik in enumerate(ibz2bz):
        assert np.allclose(F[:, ik], S1[:, i] + T[:, ik])
