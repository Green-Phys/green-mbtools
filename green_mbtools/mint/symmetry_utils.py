# -------------Ascknwledgements-------------
# Adopted from Xinyang's implementation:
# https://github.com/CQMP/MBSymmetry
# ------------------------------------------

import numpy as np


def fold_to_unit_cell(r_cart_scaled, mycell):
    """
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
    a = mycell.lattice_vectors()
    cell_idx = np.floor(r_cart_scaled).astype(int)
    r_rel = r_cart_scaled - cell_idx
    return r_rel


def generate_permutation_info(mycell, symm_op, tol=1e-10, verbose=False):
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
    partner_idx : int
        Index of the atom that is the partner under the symmetry operation.
    pos_diff : ndarray
        The positional difference vector due to folding into the unit cell (3,).
    """
    # info about symmetry operation
    rot = symm_op.rot
    trans = symm_op.trans

    # unit cell info
    n_atom = mycell.natm

    # Quantities to be returned
    partner_idx = np.zeros(n_atom, dtype=int)
    pos_diff = np.zeros((n_atom, 3))
    coords_scaled = mycell.get_scaled_atom_coords().reshape(-1,3)

    for i in range(n_atom):
        i_coord = coords_scaled[i]
        trans_pos = np.dot(rot, i_coord) + trans
        shift_pos = fold_to_unit_cell(trans_pos, mycell)
        pos_diff[i] = shift_pos - trans_pos

        # Find the corresponding atom partner
        found_partner = False
        for j in range(n_atom):
            j_coord = coords_scaled[j]
            distance = np.linalg.norm(shift_pos - j_coord)
            if distance < tol:
                if mycell.atom_symbol(i) != mycell.atom_symbol(j):
                    raise RuntimeError("point group maps atoms of different type onto each other")
                # Else
                found_partner = True
                partner_idx[i] = j
                if verbose:
                    print(f"Atom {i} ({mycell.atom_symbol(i)}) maps to Atom {j} ({mycell.atom_symbol(j)})"
                          + f" with shift {shift_pos - trans_pos}")
                break

        # Handle error
        if (not found_partner):
            print("atom position: ", coords_scaled[i])
            print("symmetry operation: ", symm_op)
            print("rotation: ", rot)
            print("translation vector: ", trans)
            print("transformed position: ", trans_pos)
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


def get_representation(bz_idx, symm_op_idx, mycell, kstruct, tol=1e-10, verbose=False):
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
        Tolerance for numerical comparisons, by default 1e-10
    verbose : bool, optional
        If True, print detailed information, by default False

    Returns
    -------
    repr_matrix : ndarray
        The representation matrix for the symmetry operation (n_atm, n_atm).
    """
    n_atom = mycell.natm
    nao = mycell.nao_nr()
    symm_op = kstruct.ops[symm_op_idx]
    perm_atoms, pos_diff = generate_permutation_info(mycell, symm_op, tol=tol, verbose=verbose)
    repr_matrix = np.zeros((nao, nao), dtype=complex)

    bz_kvec = kstruct.kpts_scaled[bz_idx]
    # (loc_start_idx, loc_end_idx, orb_start, orb_end) for each atom
    aoslice = mycell.aoslice_by_atom()
    # starting index of each AO shell
    ao_loc = mycell.ao_loc

    for i in range(n_atom):
        # phase
        target_atom = perm_atoms[i]
        phase = np.exp(1j * 2 * np.pi * bz_kvec.dot(pos_diff[i]))
        # starting and ending index for AO blocks
        loc_start_idx = aoslice[i][0]
        loc_end_idx = aoslice[i][1]
        # get matrix representation in orbital basis
        for ao_loc_idx in range(loc_start_idx, loc_end_idx):
            # angular momentum for the block of AOs
            L_value = mycell.bas_angular(ao_loc_idx)
            multiplicity = 2 * L_value + 1
            n_orbs_for_L = ao_loc[ao_loc_idx + 1] - ao_loc[ao_loc_idx]
            # number of principle quantum number shells in the block
            n_shells = n_orbs_for_L // multiplicity
            for n_i in range(n_shells):
                i_start, i_end = get_orbital_index(i, n_i, L_value, mycell)
                j_start, j_end = get_orbital_index(target_atom, n_i, L_value, mycell)
                repr_matrix[j_start:j_end, i_start:i_end] = phase * kstruct.Dmats[symm_op_idx][L_value]

    # info about symmetry operation
    return repr_matrix
