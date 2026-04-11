from green_mbtools.pesto.dyson import solve_dyson
from numpy.linalg import norm
import numpy as np

#
# Test function for Dyson equation solver
#


def test_dyson(mbo):
    """Use Fock, overlap and self-energy matrices to solve
    the Dyson equation and compare with available G.
    """

    gtau_dyson = solve_dyson(
        mbo.fock, mbo.S, mbo.sigma, mbo.mu, mbo.ir
    )

    print(np.max(np.abs(gtau_dyson - mbo.gtau)))
    assert np.max(np.abs(gtau_dyson - mbo.gtau)) < 1e-6
    # NOTE: threshold for testing was 1e-10 previously. We are changing this because in GREEN code,
    #       the structure of output file is such that Gtau is stored, and Sigma obtained from that Gtau is stored.
    #       So, a new Green's function from dyson (H0, Sigma1, Sigma_iw) -> G_new is not the same as G_tau from sim.h5
