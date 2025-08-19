import numpy as np

"""
Fourier transform between real and reciprocal space
"""


def construct_rmesh(nkx, nky, nkz):
    """Generate real-space mesh, in the units of lattice translation vectors. e.g.,

    .. code-block:: python

        nkx = 6
        L = (nkx - 1) // 2  # = 2
        rx = -2, -1, 0, 1, 2, 3

    Parameters
    ----------
    nkx : int
        number of k-points in the x direction
    nky : int
        number of k-points in the y direction
    nkz : int
        number of k-points in the z direction

    Returns
    -------
    numpy.ndarray
        real-space mesh
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
    """Generate a real-space mesh that is symmetric along coordinate axes. e.g.,

    .. code-block:: python

        nkx = 6
        L = (nkx - 1) // 2  # = 2
        rx = -3, -2, -1, 0, 1, 2, 3

    Parameters
    ----------
    nkx : int
        number of k-points in the x direction
    nky : int
        number of k-points in the y direction
    nkz : int
        number of k-points in the z direction

    Returns
    -------
    numpy.ndarray
        unique / symmetric real-space mesh
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
    """Compute Fourier coefficients for direct and inverse transforms between k and real space

    Parameters
    ----------
    kmesh : numpy.ndarray
        k-mesh of shape (nk, 3)
    rmesh : numpy.ndarray
        real-space mesh

    Returns
    -------
    numpy.ndarray
        coefficients to transform from r to k space
    numpy.ndarray
        coefficients to transform from k to r space
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
    """Perform Fourier transform from reciprocal to real space

    Parameters
    ----------
    frk : numpy.ndarray
        coefficients to transform from k to r space
    obj_k : numpy.ndarray
        Reciprocal space object
    weights : numpy.ndarray
        Fourier coefficient's degeneracy weights

    Returns
    -------
    numpy.ndarray
        Real space (inverse Fourier transformed) object
    """
    obj_i = np.einsum("k...,k,ki->i...", obj_k, weights, frk.conj().T)
    obj_i /= np.sum(weights)
    return obj_i


def real_to_k(fkr, obj_i):
    """Perform Fourier transform from real to reciprocal space

    Parameters
    ----------
    fkr : numpy.ndarray
        coefficients to transform from r to k space
    obj_i : numpy.ndarray
        Real space (inverse Fourier transformed) object

    Returns
    -------
    numpy.ndarray
        Reciprocal space object
    """
    original_shape = obj_i.shape
    obj_k = np.dot(fkr.conj(), obj_i.reshape(original_shape[0], -1))
    obj_k = obj_k.reshape((obj_k.shape[0],) + original_shape[1:])
    return obj_k
