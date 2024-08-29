import h5py
import pytest
import os
from mbanalysis import mb


#
# Configure the mbanalysis::MB_post objects that will be used in all the tests.
#

@pytest.fixture
def data_path():
    """Path to all the data files for pytests.
    """
    pytest.test_data_dir = os.path.abspath(
        os.path.dirname(__file__)
    ) + '/test_data'


@pytest.fixture
def mbo(data_path):
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
    mu = f["iter14/mu"][()]
    f.close()

    # input file (input.h5)
    f = h5py.File(data_dir + '/H2_GW/input.h5', 'r')
    ir_list = f["/grid/ir_list"][()]
    index = f["/grid/index"][()]
    conj_list = f["grid/conj_list"][()]
    f.close()

    # ir grid file
    irf = data_dir + '/ir_grid/1e4_104.h5'

    # All k-dependent matrices should lie on a full Monkhorst-Pack grid.
    F = mb.to_full_bz(Fr, conj_list, ir_list, index, 1)
    S = mb.to_full_bz(Sr, conj_list, ir_list, index, 1)
    Sigma = mb.to_full_bz(Sigmar, conj_list, ir_list, index, 2)
    G = mb.to_full_bz(Gr, conj_list, ir_list, index, 2)

    # Standard way to initialize
    mbobj = mb.MB_post(
        fock=F, sigma=Sigma, mu=mu, gtau=G, S=S, beta=1000,
        ir_file=irf, legacy_ir=True
    )

    return mbobj
