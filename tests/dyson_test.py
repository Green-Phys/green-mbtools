from mbanalysis.dyson import solve_dyson
from numpy.linalg import norm

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

    print(norm(gtau_dyson - mbo.gtau))
    assert norm(gtau_dyson - mbo.gtau) <= 1e-10
