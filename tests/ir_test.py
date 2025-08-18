from numpy.linalg import norm
from green_mbtools.pesto.ir import IR_factory
import h5py
import pytest
from numpy import flip

#
# Test function for IR-Transform
#


def test_ir_fourier_transform(mbo):
    """Transfrom from tau to iw and back.
    """

    my_ir = mbo.ir
    gtau = mbo.gtau

    # Tranform to iw
    G_iw = my_ir.tau_to_w(gtau)

    # # Check symmetry of G_iw
    # nw = my_ir.nw
    # G_iw_neg = flip(G_iw[:nw//2], axis=0)
    # G_iw_pos = G_iw[nw//2:]

    # assert norm(G_iw_neg.conj() + G_iw_pos) < 1e-8
    # NOTE: This test fails becaue:
    #   iw -> -iw  ==>  G(k, -iw) = -G*(-k, iw)
    # We need to find a way to take k -> -k

    # Transform back to G_tau
    gtau2 = my_ir.w_to_tau(G_iw)
    assert norm(gtau2 - gtau) < 1e-10


def test_legacy_ir_fourier_transform(data_path):
    """Tests 1) initialization of new-format IR grids, and
    2) trnasforms G from tau to iw, and back to tau to check
    functionality and accuracy.
    """

    # init IR handler
    beta = 100
    ir_file = pytest.test_data_dir + '/ir_grid/1e4_104.h5'
    myir = IR_factory(beta, ir_file, legacy_ir=True)

    # load new-format GF2 data
    gf2_path = pytest.test_data_dir + '/H2_GW_legacy/sim.h5'
    fdata = h5py.File(gf2_path, 'r')
    it = fdata['iter'][()]
    g_tau = fdata['iter' + str(it) + '/G_tau/data'][()].view(complex)
    g_tau = g_tau.reshape(g_tau.shape[:-1])

    # transform
    g_iw = myir.tau_to_w(g_tau)
    g_tau2 = myir.w_to_tau(g_iw)

    assert norm(g_tau2 - g_tau) < 1e-10
