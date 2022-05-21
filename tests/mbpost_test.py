from mbanalysis import mb
import numpy as np
import pytest
import h5py

#
# Test functions for parent class MB_post
#


def test_initMBpost(mbo, data_path):
    """Test shortcut way to initialize the mbanalysis::MB_post instance.
    """
    data_dir = pytest.test_data_dir
    sim_path = data_dir + '/H2_GW/sim.h5'
    input_path = data_dir + '/H2_GW/input.h5'
    ir_path = data_dir + '/ir_grid/1e4_104.h5'
    mbo_new = mb.initialize_MB_post(
        sim_path, input_path, ir_path
    )

    # compare G
    Gk = mbo.gtau
    Gk_new = mbo_new.gtau
    assert np.linalg.norm(Gk - Gk_new) < 1e-10

    # compare S
    Sk = mbo.S
    Sk_new = mbo_new.S
    assert np.linalg.norm(Sk - Sk_new) < 1e-10

    # compare sigma
    Sigma_k = mbo.sigma
    Sigma_k_new = mbo_new.sigma
    assert np.linalg.norm(Sigma_k - Sigma_k_new) < 1e-10

    # compare fock
    Fock_k = mbo.fock
    Fock_k_new = mbo_new.fock
    assert np.linalg.norm(Fock_k - Fock_k_new) < 1e-10

    return


def test_to_full_bz(data_path):
    """Test the transformation from reduced k-points to
    full Brillouin zone.
    """
    data_dir = pytest.test_data_dir

    # data file (sim.h5)
    f = h5py.File(data_dir + '/H2_GW/sim.h5', 'r')
    Sr = f["S-k"][()].view(complex)
    Sr = Sr.reshape(Sr.shape[:-1])
    Fr = f["iter14/Fock-k"][()].view(complex)
    Fr = Fr.reshape(Fr.shape[:-1])
    Sigmar = f["iter14/Selfenergy/data"][()].view(complex)
    Sigmar = Sigmar.reshape(Sigmar.shape[:-1])
    Gr = f["iter14/G_tau/data"][()].view(complex)
    Gr = Gr.reshape(Gr.shape[:-1])
    f.close()

    # input file (input.h5)
    f = h5py.File(data_dir + '/H2_GW/input.h5', 'r')
    ir_list = f["/grid/ir_list"][()]
    index = f["/grid/index"][()]
    conj_list = f["grid/conj_list"][()]
    f.close()

    # reduced k-points (for 3x3x3)
    ink = Gr.shape[2]
    assert ink == 14

    # Transform to full BZ
    Gk = mb.to_full_bz(Gr, conj_list, ir_list, index, 2)
    fullk = Gk.shape[2]
    assert fullk == 27

    return


def test_mulliken(mbo):
    """Test Mulliken analysis functionality.
    Compute the occupation numbers of the orbitals and compare
    with reference values.
    """
    occs = mbo.mulliken_analysis()
    assert occs[0].all() == np.array([0.5, 0.5]).all()  # spin up
    assert occs[1].all() == np.array([0.5, 0.5]).all()  # spin down
