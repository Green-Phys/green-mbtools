import numpy as np

"""
Fourier transform between real and reciprocal space
"""


def construct_rmesh(nkx, nky, nkz):
    """Generat a real-space rmesh.
    rmesh is constructed in the units of lattice translation vectors.
    e.g.:   if nk = 6
            L = 2
            left = 1
            rx = -2, -1, 0, 1, 2, 3
    """
    Lx, Ly, Lz = (nkx - 1) // 2, (nky - 1) // 2, (nkz - 1) // 2
    leftx, lefty, leftz = (nkx - 1) % 2, (nky - 1) % 2, (nkz - 1) % 2

    rx = np.linspace(-Lx, Lx+leftx, nkx, endpoint=True)
    ry = np.linspace(-Ly, Ly+lefty, nky, endpoint=True)
    rz = np.linspace(-Lz, Lz+leftz, nkz, endpoint=True)
    RX, RY, RZ = np.meshgrid(rx, ry, rz)
    rmesh = np.array([RX.flatten(), RY.flatten(), RZ.flatten()]).T

    return rmesh


def construct_symmetric_rmesh(nkx, nky, nkz):
    """Generate a rmesh that is symmetric along the coordinate axes.
    e.g.:   if nk = 6
            L = 2
            left = 1
            rx_symm = -3, -2, -1, 0, 1, 2, 3
    """
    Lx, Ly, Lz = (nkx - 1) // 2, (nky - 1) // 2, (nkz - 1) // 2
    leftx, lefty, leftz = (nkx - 1) % 2, (nky - 1) % 2, (nkz - 1) % 2

    rx = np.linspace(-Lx, Lx + leftx, nkx, endpoint=True)
    ry = np.linspace(-Ly, Ly + lefty, nky, endpoint=True)
    rz = np.linspace(-Lz, Lz + leftz, nkz, endpoint=True)
    RX, RY, RZ = np.meshgrid(rx, ry, rz)
    rmesh = np.array([RX.flatten(), RY.flatten(), RZ.flatten()]).T
    pm_rmesh = np.append(rmesh, -rmesh, axis=0)
    rmesh = np.unique(pm_rmesh, axis=0)

    return rmesh


def compute_fourier_coefficients(kmesh, rmesh):
    """Compute Fourier coefficients for direct and
    inverse Fourier transform

    Input   :   kmesh, rmesh
    Returns :   fkr, frk
                where fkr = coefficients from r to k,
                and frk = coefficients from k to r.
    """
    fkr = np.zeros((kmesh.shape[0], rmesh.shape[0]), dtype=complex)
    frk = np.zeros((rmesh.shape[0], kmesh.shape[0]), dtype=complex)
    for ik, k in enumerate(kmesh):
        for jr, r in enumerate(rmesh):
            dp = 2.j*np.pi*np.dot(k, r)
            # coefficients from r to k
            fkr[ik, jr] = np.exp(-dp)
            # coefficients from k to r
            frk[jr, ik] = np.exp(dp)

    return fkr, frk


def k_to_real(frk, obj_k, weights):
    """
    Perform Fourier transform from reciprocal to real space
    frk     - Fourier transform coefficients from reciprocal to real space
    obj_k   - reciprocal space object
    weights - Fourier coefficient degenerate weights

    return obj_i - inverse Fourier transform of obj_k
    """
    obj_i = np.einsum("k...,k,ki->i...", obj_k, weights, frk.conj().T)
    obj_i /= np.sum(weights)
    return obj_i


def real_to_k(fkr, obj_i):
    """
    Perform Fourier transform from reciprocal to real space
    fkr     - Fourier transform coefficients from real to reciprocal space
    obj_i   - real space object

    return obj_k - Fourier transform of obj_i
    """
    original_shape = obj_i.shape
    obj_k = np.dot(fkr.conj(), obj_i.reshape(original_shape[0], -1))
    obj_k = obj_k.reshape((obj_k.shape[0],) + original_shape[1:])
    return obj_k
