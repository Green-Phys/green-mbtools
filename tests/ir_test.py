from numpy.linalg import norm

#
# Test function for IR-Transform
#


def test_ir_transform(mbo):
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
