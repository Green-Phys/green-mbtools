from pyscf.pbc.lib.kpts_helper import unique_with_wrap_around
from pyscf.pbc.lib import kpts as libkpts


def wrap_k(k):
    while k < 0 :
        k = 1 + k
    while (k - 9.9999999999e-1) > 0.0 :
        k = k - 1
    return k


def build_q_struct(mycell, k_mesh, space_symm=False, tr_symm=True):
    """Initialize q-mesh for GDF

    Parameters
    ----------
    mycell : pyscf.pbc.Cell
        unit cell for simulation
    k_mesh : numpy.ndarray
        k-mesh for the Brillouin Zone
    space_symm : bool
        utilize space group symmetry for qmesh reduction
    tr_symm : bool
        utilize time-reversal symmetry for qmesh reduction
    
    Returns
    -------
    pyscf.pbc.lib.kpts.KPoints
        q-mesh struct for the Brillouin Zone
    """
    # Build q = k1 - k2 over the full k-mesh and remove duplicates with
    # the same wrap-around convention as integral generation.
    q_mesh = (k_mesh[None, :, :] - k_mesh[:, None, :]).reshape(-1, 3)
    q_mesh, _, _ = unique_with_wrap_around(mycell, q_mesh)

    # Use the same folding procedure as init_k_mesh.
    for i, _ in enumerate(q_mesh):
        qi = mycell.get_scaled_kpts(q_mesh[i])
        qi = [wrap_k(l) for l in qi]
        q_mesh[i] = mycell.get_abs_kpts(qi)
    for i, _ in enumerate(q_mesh):
        qi = mycell.get_scaled_kpts(q_mesh[i])
        qi = [wrap_k(l) for l in qi]
        q_mesh[i] = mycell.get_abs_kpts(qi)

    qstruct = libkpts.make_kpts(mycell, q_mesh, space_group_symmetry=space_symm, time_reversal_symmetry=tr_symm)

    return qstruct
