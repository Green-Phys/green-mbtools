# -------------Acknowledgements-------------
# The functions "fold_to_unit_cell", "generate_permutation_info", "get_orbital_index" and
# "get_representation" are adopted from Xinyang's implementation:
# https://github.com/CQMP/MBSymmetry
# ------------------------------------------

import numpy as np
import h5py
import warnings


def fold_to_unit_cell(r_cart_scaled):
    """
    Fold a scaled (fractional) coordinate into the primary unit cell.

    Parameters
    ----------
    r_cart_scaled : array_like
        Scaled/fractional coordinate to be folded, shape (3,). This should be in
        the same convention as ``Cell.get_scaled_atom_coords()`` (i.e. expressed
        in units of the lattice vectors, not in Cartesian units). The parameter
        name is historical and does not imply Cartesian coordinates.
    mycell : pyscf.pbc.gto.cell.Cell
        The unit cell object from PySCF.
    Returns
    -------
    frac : ndarray
        The folded scaled/fractional coordinate within the unit cell, shape (3,).
        Each component is wrapped into the interval [-0.5, 0.5) to match the atom
        coordinate convention used elsewhere in this module.

    Fold a Cartesian coordinate into the primary unit cell.

    Parameters:
    -----------
    r_cart_scaled : array_like
        Scaled cartesian coordinate to be folded (3,).
    mycell : pyscf.pbc.gto.cell.Cell
        The unit cell object from PySCF.

    Returns:
    -----------
    r_rel : ndarray
        The folded Cartesian coordinate within the unit cell (3,).
    """
    frac = np.asarray(r_cart_scaled, dtype=float)
    # Wrap to [-0.5, 0.5) to match atom coordinate convention
    frac = np.mod(frac + 0.5, 1.0) - 0.5
    return frac


def generate_permutation_info(mycell, symm_op, tol=1e-8, verbose=False):
    """Generate permutation info for given symmetry operation on the atoms of unit cell.

    Parameters
    ----------
    mycell : pyscf.pbc.gto.cell.Cell
        The unit cell object from PySCF.
    symm_op : pyscf.pbc.symm.space_group.SPGElement
        The symmetry operation element from PySCF.
    tol : float, optional
        Tolerance for numerical comparisons, by default 1e-10
    verbose : bool, optional
        If True, print detailed information, by default False
    
    Returns
    -------
    partner_idx : ndarray of int, shape (n_atom,)
        For each atom, the index of the atom it is mapped onto by the symmetry
        operation.
    pos_diff : ndarray, shape (n_atom, 3)
        For each atom, the lattice translation (in scaled/fractional
        coordinates) carrying its operated position ``R @ r_i + t`` to its
        in-cell partner ``r_j``: ``pos_diff = r_j - (R @ r_i + t)``. Partners are
        matched using folded coordinates; this vector is computed from the
        unfolded (as-input) coordinates that the Bloch phase in
        :func:`get_representation` uses.
    """
    # info about symmetry operation
    rot = symm_op.rot
    trans = symm_op.trans

    # unit cell info
    n_atom = mycell.natm

    # Quantities to be returned
    partner_idx = np.zeros(n_atom, dtype=int)
    pos_diff = np.zeros((n_atom, 3))
    # As-input scaled coordinates: the convention the Bloch integrals
    # (H-k, S-k, ...) use, and the reference for the Bloch phase below.
    coords_raw = mycell.get_scaled_atom_coords().reshape(-1, 3)
    # Folded copy in [-0.5, 0.5), used to match an atom to its image (robust
    # when atoms sit near/across the cell boundary).
    coords_scaled = np.array([fold_to_unit_cell(c) for c in coords_raw])

    for i in range(n_atom):
        i_coord = coords_scaled[i]
        trans_pos = np.dot(rot, i_coord) + trans
        shift_pos = fold_to_unit_cell(trans_pos)

        # Find the corresponding atom partner
        found_partner = False
        min_distance = 1.0
        for j in range(n_atom):
            j_coord = coords_scaled[j]
            distance = np.linalg.norm(shift_pos - j_coord)
            min_distance = min(min_distance, distance)
            if distance < tol:
                if mycell.atom_symbol(i) != mycell.atom_symbol(j):
                    raise RuntimeError("point group maps atoms of different type onto each other")
                # Else
                found_partner = True
                partner_idx[i] = j
                # Lattice translation carrying the operated position R r_i + t
                # to its in-cell partner r_j, in as-input coordinates. This is
                # the vector the Bloch phase exp(i 2*pi k . pos_diff) in
                # get_representation uses.
                pos_diff[i] = coords_raw[j] - (np.dot(rot, coords_raw[i]) + trans)
                if verbose:
                    print(f"Atom {i} ({mycell.atom_symbol(i)}) maps to Atom {j} ({mycell.atom_symbol(j)})"
                          + f" with lattice shift {pos_diff[i]}")
                break

        # Handle error
        if (not found_partner):
            print("atom position: ", coords_scaled[i])
            print("shifted position: ", shift_pos)
            print("symmetry operation: ", symm_op)
            print("rotation: ", rot)
            print("translation vector: ", trans)
            print("transformed position: ", trans_pos)
            print("Min distance: ", min_distance)
            print("Available atom coordinates: ", coords_scaled)
            raise RuntimeError("symmetry analysis could not find partner.");

    return partner_idx, pos_diff


def get_orbital_index(atom_idx, n_, L_, mycell):
    """Get the starting and ending index of orbitals for a given atom and angular momentum.

    Parameters
    ----------
    atom_idx : int
        Index of the atom in the unit cell.
    n_ : int
        Principal quantum number.
    L_ : int
        Angular momentum quantum number.
    mycell : pyscf.pbc.gto.cell.Cell
        The unit cell object from PySCF.

    Returns
    -------
    orb_start : int
        Starting index of the orbitals.
    orb_end : int
        Ending index of the orbitals.
    """
    aoslice = mycell.aoslice_by_atom()
    ao_loc = mycell.ao_loc

    loc_start_idx = aoslice[atom_idx][0]
    loc_end_idx = aoslice[atom_idx][1]

    orb_start = None
    orb_end = None

    for ao_loc_idx in range(loc_start_idx, loc_end_idx):
        L_value = mycell.bas_angular(ao_loc_idx)
        if L_value == L_:
            multiplicity = 2 * L_value + 1
            n_orbs_for_L = ao_loc[ao_loc_idx + 1] - ao_loc[ao_loc_idx]
            n_shells = n_orbs_for_L // multiplicity
            if n_ < n_shells:
                orb_start = ao_loc[ao_loc_idx] + n_ * multiplicity
                orb_end = orb_start + multiplicity
                break

    if orb_start is None or orb_end is None:
        raise ValueError("Specified (n, L) not found for the given atom.")

    return orb_start, orb_end


def get_representation(bz_idx, symm_op_idx, mycell, kstruct, tol=1e-5, verbose=False, tr_phase=True):
    """Get the representation matrix for given symmetry operation on the atoms of unit cell.

    Parameters
    ----------
    bz_idx : int
        Index of the k-point in the Brillouin zone.
    symm_op_idx : int
        Index of the symmetry operation element from PySCF.
    mycell : pyscf.pbc.gto.cell.Cell
        The unit cell object from PySCF.
    kstruct : pyscf.pbc.symm.KPointsSymmetry
        k-point symmetry structure for aux-basis
    tol : float, optional
        Tolerance for atom-position matching in generate_permutation_info.
        Default is 1e-5, matching generate_permutation_info's own default.
        Note: PySCF stores fractional translations with ~6 decimal places
        (e.g. 0.666667 instead of 2/3), introducing ~3e-7 residuals after
        applying the operation. A tighter tol (e.g. 1e-10) would therefore
        fail for any lattice whose space-group translations are not integers.
    verbose : bool, optional
        If True, print detailed information, by default False
    tr_phase : bool, optional
        If True (default), evaluate the Bloch phase at -k for time-reversal-
        paired k-points (``kstruct.time_reversal_symm_bz[bz_idx]`` True). For a
        TR-paired k the spatial operation lands on the image -k and the
        reconstruction ``X(k) = (U X(k_ir) U^dagger)*`` conjugates back to +k,
        so U must be the operator that lands on -k, i.e. its phase must use -k.
        (A no-op when the phase is real, since then +k == -k.) Set False for
        callers that apply their own time-reversal handling -- the x2c==2 spinor
        path -- so the phase is not corrected twice.

    Returns
    -------
    repr_matrix : ndarray
        The representation matrix for the symmetry operation (nao, nao).
    """
    n_atom = mycell.natm
    nao = mycell.nao_nr()
    symm_op = kstruct.ops[symm_op_idx]
    perm_atoms, pos_diff = generate_permutation_info(mycell, symm_op, tol=tol, verbose=verbose)
    repr_matrix = np.zeros((nao, nao), dtype=complex)

    bz_kvec = kstruct.kpts_scaled[bz_idx]
    # Time-reversal-paired k-points: the spatial operation lands on -k (the
    # reconstruction then conjugates to reach +k), so the Bloch phase must be
    # evaluated at -k. No-op for real phases (+k == -k).
    tr_bz = getattr(kstruct, "time_reversal_symm_bz", None)
    if tr_phase and tr_bz is not None and tr_bz[bz_idx]:
        bz_kvec = -bz_kvec
    # (loc_start_idx, loc_end_idx, orb_start, orb_end) for each atom
    aoslice = mycell.aoslice_by_atom()
    # starting index of each AO shell
    ao_loc = mycell.ao_loc
    # get angular momentum info for each shell
    ao_bas = np.zeros(len(ao_loc) - 1, dtype=int)
    for ao_loc_idx in range(len(ao_loc) - 1):
        ao_bas[ao_loc_idx] = mycell.bas_angular(ao_loc_idx)

    for i in range(n_atom):
        # phase
        target_atom = perm_atoms[i]
        phase = np.exp(1j * 2 * np.pi * bz_kvec.dot(pos_diff[i]))
        # starting and ending index for AO shell indices
        loc_start_idx = aoslice[i][0]
        loc_end_idx = aoslice[i][1]
        target_loc_start_idx = aoslice[target_atom][0]
        
        # get matrix representation in orbital basis
        # Match shells by their order within each atom
        for shell_offset, ao_loc_idx in enumerate(range(loc_start_idx, loc_end_idx)):
            # Find corresponding shell in target atom by position
            target_ao_loc_idx = target_loc_start_idx + shell_offset
            
            # angular momentum for the block of AOs
            L_value = ao_bas[ao_loc_idx]
            target_L_value = ao_bas[target_ao_loc_idx]
            
            # Verify angular momentum matches
            if L_value != target_L_value:
                raise RuntimeError(f"Angular momentum mismatch: shell {ao_loc_idx} of atom {i} has L={L_value}, "
                                   f"but shell {target_ao_loc_idx} of atom {target_atom} has L={target_L_value}")
            
            multiplicity = 2 * L_value + 1
            n_orbs_for_L = ao_loc[ao_loc_idx + 1] - ao_loc[ao_loc_idx]
            target_n_orbs = ao_loc[target_ao_loc_idx + 1] - ao_loc[target_ao_loc_idx]
            
            # Verify orbital count matches
            if n_orbs_for_L != target_n_orbs:
                raise RuntimeError(f"Orbital count mismatch: shell {ao_loc_idx} has {n_orbs_for_L} orbitals, "
                                   f"but shell {target_ao_loc_idx} has {target_n_orbs} orbitals")
            
            # number of radial shells in the block
            n_shells = n_orbs_for_L // multiplicity
            
            # Fill representation matrix for each radial shell
            for n_i in range(n_shells):
                i_start = ao_loc[ao_loc_idx] + n_i * multiplicity
                i_end = i_start + multiplicity
                j_start = ao_loc[target_ao_loc_idx] + n_i * multiplicity
                j_end = j_start + multiplicity
                repr_matrix[j_start:j_end, i_start:i_end] = phase * kstruct.Dmats[symm_op_idx][L_value]

    # info about symmetry operation
    return repr_matrix


def rotation_matrix_to_su2(R_cart):
    """Return the SU(2) representative of a proper 3D Cartesian rotation matrix.

    For a rotation by angle :math:`\\varphi` about unit axis :math:`\\hat{n}`:

    .. math::
        D^{1/2}(R) = \\cos(\\varphi/2)\\,I_2
                    + i\\sin(\\varphi/2)\\,(\\hat{n}\\cdot\\boldsymbol{\\sigma})

    Parameters
    ----------
    R_cart : (3, 3) float ndarray
        Proper rotation matrix (``det = +1``) in Cartesian coordinates.

    Returns
    -------
    su2 : (2, 2) complex ndarray
    """
    from scipy.spatial.transform import Rotation
    rotvec = Rotation.from_matrix(R_cart).as_rotvec()
    angle = np.linalg.norm(rotvec)
    if angle < 1e-10:
        return np.eye(2, dtype=np.complex128)
    axis = rotvec / angle
    sx = np.array([[0,  1 ], [1,  0 ]], dtype=np.complex128)
    sy = np.array([[0, -1j], [1j, 0 ]], dtype=np.complex128)
    sz = np.array([[1,  0 ], [0, -1 ]], dtype=np.complex128)
    return (np.cos(angle / 2) * np.eye(2, dtype=np.complex128)
            + 1j * np.sin(angle / 2) * (axis[0]*sx + axis[1]*sy + axis[2]*sz))


def get_spinor_representation(bz_idx, symm_op_idx, mycell, kstruct, tol=1e-5, verbose=False):
    """Double-group spinor AO representation :math:`D^{1/2}(R^{-1}) \\otimes U_\\text{orbital}(R^{-1})`.

    Reads the rotation directly from ``kstruct.ops[symm_op_idx]``, converts it
    to a Cartesian rotation, lifts it to SU(2) via :func:`rotation_matrix_to_su2`,
    and combines it with the orbital representation from :func:`get_representation`.

    PySCF's ``Dmats`` use the passive (inverse) convention :math:`D^L(R^{-1})`, so
    :func:`get_representation` returns :math:`U_\\text{orbital}(R^{-1})`. The matching
    SU(2) factor is therefore :math:`D^{1/2}(R^{-1}) = D^{1/2}(R)^\\dagger`, i.e. the
    conjugate transpose of the direct lift.

    Parameters
    ----------
    bz_idx : int
        Index of the BZ k-point.
    symm_op_idx : int
        Index of the symmetry operation in ``kstruct.ops``.
    mycell : pyscf.pbc.gto.Cell
        PySCF unit cell.
    kstruct : pyscf.pbc.lib.kpts.KPoints
        k-point symmetry structure from ``mycell.make_kpts(...)``.
    tol : float, optional
        Tolerance passed to get_representation for atom-position matching.
        Default 1e-5 matches generate_permutation_info's own default and
        accommodates PySCF's ~6 decimal-place translation precision (~3e-7
        residuals). See get_representation for full discussion.

    Returns
    -------
    u_spinor : (nso, nso) complex ndarray
        Full spinor AO representation, ``nso = 2 * nao``.
    """
    # tr_phase=False here: for a TR point the x2c==2 store keeps
    # (u_spinor @ theta).conj(). That conjugation acts on the whole product, so
    # it also conjugates the orbital block and evaluates its Bloch phase at -k,
    # which is what the reconstruction needs. The orbital phase is therefore
    # built at +k here and the store conjugation supplies the -k.
    u_orbital = get_representation(bz_idx, symm_op_idx, mycell, kstruct, tol=tol, verbose=verbose, tr_phase=False)
    rot_frac = np.array(kstruct.ops[symm_op_idx].rot, dtype=float)
    a = mycell.lattice_vectors()
    rot_cart = a.T @ rot_frac @ np.linalg.inv(a.T)
    if np.linalg.det(rot_cart) < 0:   # improper: inversion is trivial on spinors
        rot_cart = -rot_cart
    # D†= D^{1/2}(R^{-1}): matches PySCF's passive Dmats convention
    su2 = rotation_matrix_to_su2(rot_cart).conj().T
    return np.kron(su2, u_orbital)


def check_kspace_symmetry_breaking(inp_file, datasets):
    """Report symmetry reconstruction residuals for k-resolved matrix quantities.

    Parameters
    ----------
    inp_file : string
        Path to input.h5 file that contains all the output from initialization
    datasets : list
        List of datasets in the input file, for which symmetry checks need to be performed
    """
    finp = h5py.File(inp_file, 'r')
    # get k-symmetry info
    bz2ibz = finp['symmetry/k/bz2ibz'][()]
    tr_conj = finp['symmetry/k/tr_conj'][()]
    nk = finp['symmetry/k/nk'][()]
    k_sym_trans = finp['symmetry/k/k_sym_transform_ao'][()]

    for dset in datasets:
        X = finp[dset][()].view(complex)
        X = X.reshape(X.shape[:-1])
        ns = X.shape[0]

        max_abs = 0.0
        for s in range(ns):
            for k in range(nk):
                k_ir = int(bz2ibz[k])
                Uk = k_sym_trans[k]
                recon = Uk @ X[s, k_ir] @ Uk.conj().T
                if int(tr_conj[k]) != 0:
                    recon = recon.conjugate()
                diff = np.max(np.abs(recon - X[s, k]))
                if diff > max_abs:
                    max_abs = diff

        if max_abs > 1e-3:
            warnings.warn(
                f"Dataset '{dset}' is not symmetric under the stored k-point symmetry operations "
                f"(max residual = {max_abs:.3e}). "
                "The mean-field solution may have broken the assumed space-group symmetry. "
                "Please rerun the initialization with '--grid_only' and '--space_symm=false' "
                "to disable space-group symmetry and obtain a consistent set of k-points.",
                UserWarning,
                stacklevel=2,
            )
    finp.close()
