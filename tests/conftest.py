import h5py
import pytest
import os
from green_mbtools.pesto import mb


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
    it = f['iter'][()]
    iter_str = "iter{}".format(it)
    Sigma1r = f[iter_str + "/Sigma1"][()].view(complex)
    Sigmar = f[iter_str + "/Selfenergy/data"][()].view(complex)
    Gr = f[iter_str + "/G_tau/data"][()].view(complex)
    mu = f[iter_str + "/mu"][()]
    f.close()

    # input file (input.h5)
    f = h5py.File(data_dir + '/H2_GW/input.h5', 'r')
    S = f['HF/S-k'][()].view(complex)
    S = S.reshape(S.shape[:-1])
    H0 = f['HF/H-k'][()].view(complex)
    H0 = H0.reshape(H0.shape[:-1])
    ir_list = f["/grid/ir_list"][()]
    index = f["/grid/index"][()]
    conj_list = f["grid/conj_list"][()]
    f.close()

    # ir grid file
    irf = data_dir + '/ir_grid/1e4.h5'

    # All k-dependent matrices should lie on a full Monkhorst-Pack grid.
    Sigma1 = mb.to_full_bz(Sigma1r, conj_list, ir_list, index, 1)
    Sigma = mb.to_full_bz(Sigmar, conj_list, ir_list, index, 2)
    G = mb.to_full_bz(Gr, conj_list, ir_list, index, 2)
    F = H0 + Sigma1

    # Standard way to initialize
    mbobj = mb.MB_post(fock=F, sigma=Sigma, mu=mu, gtau=G, S=S, beta=1000, ir_file=irf)

    return mbobj
