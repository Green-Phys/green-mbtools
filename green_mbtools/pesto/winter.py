import numpy as np

from . import ft
from . import dyson


# Only work for full_bz object
def interpolate(obj_k, kmesh, kpts_inter, dim=3, hermi=False, debug=False, nk_list=None):
    """Perform Wannier interpolation for static quantities

    Parameters
    ----------
    obj_k : numpy.ndarray
        input object on full Brillouin zone k-mesh, of shape (ns, nk, nao, nao)
    kmesh : numpy.ndarray
        original k-mesh (in scaled units)
    kpts_inter : numpy.ndarray
        k-points to interpolate on (in scaled units)
    dim : int, optional
        dimensionality of crystal (2D or 3D), used only when nk_list is not given
    hermi : bool, optional
        set to True if the interpolated matrix Hermitian, by default False
    debug : bool, optional
        set to True if debug information needs to be printed, by default False
    nk_list : array-like of 3 ints, optional
        k-mesh dimensions [nkx, nky, nkz] read from params/nk_list in the HDF5
        input file.  Takes precedence over `dim` when provided.

    Returns
    -------
    numpy.ndarray
        Interpolated tensor

    Raises
    ------
    ValueError
        if `dim` other than 2 or 3 is specified and nk_list is not provided
    """
    ns, Nk, nao = obj_k.shape[:3]
    if nk_list is not None:
        nkx, nky, nkz = nk_list
        rmesh = ft.construct_rmesh(nkx, nky, nkz)
        nk = nkx
    elif dim == 3:
        nk = int(np.cbrt(Nk))
        rmesh = ft.construct_rmesh(nk, nk, nk)
    elif dim == 2:
        nk = int(np.sqrt(Nk))
        rmesh = ft.construct_rmesh(nk, nk, 1)
    else:
        raise ValueError(
            "Wannier interpolation only supports 3D and 2D systems."
        )

    fkr, frk = ft.compute_fourier_coefficients(kmesh, rmesh)
    weights = [1] * kmesh.shape[0]
    obj_i = np.array([ft.k_to_real(frk, obj_k[s], weights) for s in range(ns)])
    if debug:
        center = np.where(np.all(rmesh == (0., 0., 0.), axis=1))[0][0]
        for i in range(nk):
            print("obj_i[", i-nk//2, ", 0, 0] = ")
            print(np.diag(obj_i[0, center - nk//2+i].real))

    fkr_int, frk_int = ft.compute_fourier_coefficients(kpts_inter, rmesh)
    obj_k_int = np.array([ft.real_to_k(fkr_int, obj_i[s]) for s in range(ns)])

    if hermi:
        error = 0.0
        for s in range(ns):
            for ik in range(kpts_inter.shape[0]):
                obj = obj_k_int[s, ik]
                obj_sym = 0.5 * (obj + obj.conj().T)
                error = max(error, np.max(np.abs(obj_sym - obj)))
                obj_k_int[s, ik] = obj_sym
        print("The largest Hermitization error = ", error)

    return obj_k_int


# Only work for full_bz object
# TODO merge interpolate_tk_object and interpolate
def interpolate_tk_object(obj_tk, kmesh, kpts_inter, dim=3, hermi=False, debug=False, nk_list=None):
    """Perform Wannier interpolation for dynamic quantities

    Parameters
    ----------
    obj_k : numpy.ndarray
        input object on full Brillouin zone k-mesh, of shape (ntau, ns, nk, nao, nao)
    kmesh : numpy.ndarray
        original k-mesh (in scaled units)
    kpts_inter : numpy.ndarray
        k-points to interpolate on (in scaled units)
    dim : int, optional
        dimensionality of crystal (2D or 3D), used only when nk_list is not given
    hermi : bool, optional
        set to True if the interpolated matrix Hermitian, by default False
    debug : bool, optional
        set to True if debug information needs to be printed, by default False
    nk_list : array-like of 3 ints, optional
        k-mesh dimensions [nkx, nky, nkz] read from params/nk_list in the HDF5
        input file.  Takes precedence over `dim` when provided.

    Returns
    -------
    numpy.ndarray
        Interpolated tensor

    Raises
    ------
    ValueError
        if `dim` other than 2 or 3 is specified and nk_list is not provided
    """
    nts, ns, Nk, nao = obj_tk.shape[:4]
    if nk_list is not None:
        nkx, nky, nkz = nk_list
        rmesh = ft.construct_rmesh(nkx, nky, nkz)
    elif dim == 3:
        nk = int(np.cbrt(Nk))
        rmesh = ft.construct_rmesh(nk, nk, nk)
    elif dim == 2:
        nk = int(np.sqrt(Nk))
        rmesh = ft.construct_rmesh(nk, nk, 1)
    else:
        raise ValueError(
            "Wannier interpolation only supports 3D and 2D systems."
        )
    fkr, frk = ft.compute_fourier_coefficients(kmesh, rmesh)
    weights = [1]*kmesh.shape[0]
    obj_ti = np.array(
        [
            ft.k_to_real(
                frk, obj_tk[it, s],
                weights
            ) for it in range(nts) for s in range(ns)
        ]
    )

    if debug:
        center = np.where(np.all(rmesh == (0., 0., 0.), axis=1))[0][0]
        for i in range(nk):
            print("obj_i[", i - nk // 2, ", 0, 0] = ")
            print(np.diag(obj_ti[0, center - nk // 2 + i].real))

    fkr_int, frk_int = ft.compute_fourier_coefficients(kpts_inter, rmesh)
    obj_tk_int = np.array(
        [ft.real_to_k(fkr_int, obj_ti[its]) for its in range(nts * ns)]
    )

    if hermi:
        error = 0.0
        for its in range(nts*ns):
            for ik in range(kpts_inter.shape[0]):
                obj = obj_tk_int[its, ik]
                obj_sym = 0.5 * (obj + obj.conj().T)
                error = max(error, np.max(np.abs(obj_sym - obj)))
                obj_tk_int[its, ik] = obj_sym
        print("The largest Hermitization error = ", error)
    obj_tk_int = obj_tk_int.reshape(nts, ns, kpts_inter.shape[0], nao, nao)

    return obj_tk_int


# Only work for full_bz object
def interpolate_G(
    Fk, Sigma_tk, mu, Sk, kmesh, kpts_inter, ir, dim=3,
    hermi=False, debug=False, nk_list=None
):
    """Interpolate Green's function from full BZ k-mesh to specified k-points

    Parameters
    ----------
    Fk : numpy.ndarray
        Fock matrix
    Sigma_tk : numpy.ndarray
        Self-energy on imaginar time axis
    mu : float
        Chemical potential
    Sk : numpy.ndarray
        overlap matrix
    kmesh : numpy.ndarray
        input k-mesh in the Brillouin zone
    kpts_inter : numpy.ndarray
        interpolation k-points
    ir : IR_factory
        handler for Fourier transforms between imaginary time and Matsubara frequencies
    dim : int, optional
        dimensionality of lattice, by default 3; used only when nk_list is not given
    hermi : bool, optional
        Is Green's function expected to be Hermitian, by default False
    debug : bool, optional
        print extra messages for debugging, by default False
    nk_list : array-like of 3 ints, optional
        k-mesh dimensions [nkx, nky, nkz] from params/nk_list in the HDF5 input.
        Takes precedence over `dim` when provided.

    Returns
    -------
    numpy.ndarray
        Green's function interpolated on k-points

    Raises
    ------
    ValueError
        if `dim` other than 2 or 3 is specified and nk_list is not provided
    """
    ns, Nk, nao = Fk.shape[:3]
    if nk_list is None and dim != 3 and dim != 2:
        raise ValueError(
            "Wannier interpolation only supports 3D and 2D systems."
        )
    nts = ir.nts

    if Sigma_tk is not None:
        assert nts == Sigma_tk.shape[0], \
            "Number of imaginary time points mismatches."

    if Sk is not None:
        print("Interpolating overlap...")
        Sk_int = interpolate(Sk, kmesh, kpts_inter, dim, hermi, debug, nk_list=nk_list)
    else:
        Sk_int = None

    print("Interpolating Fock...")
    Fk_int = interpolate(Fk, kmesh, kpts_inter, dim, hermi, debug, nk_list=nk_list)
    # FIXME Too memory demanding and too slow as well.
    if Sigma_tk is not None:
        print("Interpolating self-energy...")
        Sigma_tk_int = interpolate_tk_object(
            Sigma_tk, kmesh, kpts_inter, dim, hermi, debug, nk_list=nk_list
        )
    else:
        Sigma_tk_int = None

    # Optional: Orthogonalization before Dyson

    # Solve Dyson
    Gtk_int = dyson.solve_dyson(Fk_int, Sk_int, Sigma_tk_int, mu, ir)

    return Gtk_int, Sigma_tk_int, ir.tau_mesh, Fk_int, Sk_int
