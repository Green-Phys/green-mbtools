import numpy as np


def solve_dyson(fock, S=None, sigma=None, mu=0, ir=None):
    """Solve Dyson equation and construct Green's function. If overlap is None, it is set to identity.

    Parameters
    ----------
    fock : numpy.ndarray
        Fock matrix of shape (ns, nk, nao, nao)
    S : numpy.ndarray, optional
        Overlap matrix of shape (ns, nk, nao, nao), by default None
    sigma : numpy.ndarray, optional
        Imaginary-time self-energy of shape (ntau, ns, nk, nao, nao), by default None (zero)
    mu : float, optional
        chemical potential, by default 0
    ir : green_mbtools.pesto.ir.IR_factory, optional
        Instance of IR factory for transforming between imaginary time and Matsubara frequencies, by default None

    Returns
    -------
    numpy.ndarray
        Green's function on imaginary time axis of shape (ntau, ns, nk, nao, nao)

    Raises
    ------
    ValueError
        if `ir` is not provided
    """
    if ir is None:
        raise ValueError("ir parameter is necessary to perform Matsubara Fourier transforms")
    nts, nw = ir.nts, ir.nw
    ns, nk, nao = fock.shape[0], fock.shape[1], fock.shape[2]
    if S is None:
        S = np.array([[np.eye(nao)]*nk]*ns)

    gtau = np.zeros((nts, ns, nk, nao, nao), dtype=complex)
    Gw = np.zeros((nw, nao, nao), dtype=complex)
    for s in range(ns):
        for k in range(nk):
            if sigma is not None:
                sigma_w = ir.tau_to_w(sigma[:, s, k])
            for n in range(nw):
                if sigma is None:
                    tmp = (1j * ir.wsample[n] + mu) * S[s, k] - fock[s, k]
                else:
                    tmp = (1j * ir.wsample[n] + mu) * S[s, k] \
                        - fock[s, k] - sigma_w[n]
                Gw[n] = np.linalg.inv(tmp)
            gtau[:, s, k] = ir.w_to_tau(Gw)

    return gtau
