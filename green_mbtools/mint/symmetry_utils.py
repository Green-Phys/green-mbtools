# -------------Acknowledgements-------------
# The functions "fold_to_unit_cell", "generate_permutation_info", "get_orbital_index" and
# "get_representation" are adopted from Xinyang's implementation:
# https://github.com/CQMP/MBSymmetry
# ------------------------------------------

import logging
import numpy as np
import h5py
import warnings
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SpaceGroupInfo:
    """Container for detected space-group data.

    Parameters
    ----------
    ops : list
        Sorted list of ``SPGElement`` symmetry operation objects as returned by
        ``pyscf.pbc.symm.space_group.SpaceGroup.build()``.
    point_group_symbol : str
        Schoenflies-style point-group symbol (e.g. ``"-3m"`` for D3d), always
        populated regardless of backend.
    international_symbol : str or None
        Hermann-Mauguin space-group symbol; populated only by the ``spglib``
        backend, ``None`` for the ``pyscf`` backend.
    international_number : int or None
        ITA space-group number; populated only by the ``spglib`` backend,
        ``None`` for the ``pyscf`` backend.
    """

    ops: List
    point_group_symbol: str
    international_symbol: Optional[str]
    international_number: Optional[int]


def detect_space_group(cell, backend: str = "pyscf", symprec: float = 1e-5) -> SpaceGroupInfo:
    """Detect the space group of a PySCF periodic cell.

    Parameters
    ----------
    cell : pyscf.pbc.gto.Cell
        Built PySCF periodic cell.
    backend : {"pyscf", "spglib"}, optional
        Symmetry detection backend. ``"pyscf"`` uses the built-in PySCF
        implementation (no external dependencies, but ``international_symbol``
        and ``international_number`` are not populated). ``"spglib"`` delegates
        to the ``spglib`` library and fills all three label fields.
    symprec : float, optional
        Symmetry tolerance passed to the backend. Default ``1e-5``.

    Returns
    -------
    SpaceGroupInfo
        Detected space-group data including the sorted ``SPGElement`` list and
        the three label fields.
    """
    from pyscf.pbc.symm.space_group import SpaceGroup

    sg = SpaceGroup(cell)
    sg.symprec = symprec
    sg.backend = backend
    sg.build()

    gn = sg.groupname
    return SpaceGroupInfo(
        ops=sg.ops,
        point_group_symbol=gn["point_group_symbol"],
        international_symbol=gn.get("international_symbol"),
        international_number=gn.get("international_number"),
    )


def _ops_closed(ops, tol=1e-6):
    """Check whether a list of SPGElement operators is closed under multiplication.

    For every ordered pair (a, b) the product a.dot(b) is computed and checked
    against the kept set.  Rotations are compared exactly (integer matrices) and
    translations are compared modulo 1 (fractional coords) within ``tol``.

    Parameters
    ----------
    ops : list of SPGElement
        Candidate subgroup operators.
    tol : float, optional
        Tolerance for the fractional-translation modular comparison.

    Returns
    -------
    is_closed : bool
        ``True`` iff every pairwise product is already in *ops*.
    missing : list of SPGElement
        Products (modulo lattice translation) that are absent from *ops*.
        Empty when *is_closed* is ``True``.
    """
    missing = []
    for a in ops:
        for b in ops:
            prod = a.dot(b)
            found = False
            for op in ops:
                if np.array_equal(prod.rot, op.rot):
                    trans_diff = np.mod(prod.trans - op.trans + 0.5, 1.0) - 0.5
                    if np.all(np.abs(trans_diff) < tol):
                        found = True
                        break
            if not found:
                # Avoid duplicate entries in the missing list
                already_listed = False
                for m in missing:
                    if np.array_equal(prod.rot, m.rot):
                        td = np.mod(prod.trans - m.trans + 0.5, 1.0) - 0.5
                        if np.all(np.abs(td) < tol):
                            already_listed = True
                            break
                if not already_listed:
                    missing.append(prod)
    return len(missing) == 0, missing


def select_ops(full_ops, mode, spec, cell):
    """Select a subset of space-group operators according to *mode*.

    Parameters
    ----------
    full_ops : list of SPGElement
        The full list of operators returned by :func:`detect_space_group`.
    mode : {"none", "symmorphic", "indices"}
        Selection strategy.

        * ``"none"``         -- return *full_ops* unchanged.
        * ``"symmorphic"``   -- retain only operators with zero fractional
          translation (``op.trans_is_zero``), guaranteeing identity is present.
        * ``"indices"``      -- retain the operators at the given integer indices
          (0-based), prepend identity if absent, and validate closure.

    spec : list of int or None
        Operator indices used by ``"indices"`` mode; ignored otherwise.
    cell : pyscf.pbc.gto.Cell
        The unit cell (currently unused but required for forward compatibility).

    Returns
    -------
    list of SPGElement
        Selected (and possibly augmented) operator list.

    Raises
    ------
    ValueError
        If an index in *spec* is out of the valid range ``0..len(full_ops)-1``
        (message names the range and points to ``--print_symmetry_ops``), or if
        the requested set is not closed under multiplication (message names the
        missing operators and mentions ``closure``).
    """
    if mode == "none":
        return full_ops

    if mode == "symmorphic":
        kept = [op for op in full_ops if op.trans_is_zero]
        # Identity must be in the symmorphic subset (the full set contains it)
        if not any(op.is_eye for op in kept):
            from pyscf.pbc.symm.space_group import SPGElement
            kept = [SPGElement()] + kept
        return kept

    if mode == "indices":
        if not spec:
            raise ValueError(
                "mode='indices' requires a non-empty list of operator indices. "
                "Run with --print_symmetry_ops to list available operators."
            )
        n = len(full_ops)
        # Range-check all indices before selecting
        bad = [idx for idx in spec if not (0 <= idx < n)]
        if bad:
            raise ValueError(
                f"Operator index/indices {bad} are out of range. "
                f"Valid range is 0..{n - 1}. "
                f"Run with --print_symmetry_ops to list available operators."
            )

        kept = [full_ops[i] for i in spec]

        # Prepend identity if not already present
        if not any(op.is_eye for op in kept):
            identity_ops = [op for op in full_ops if op.is_eye]
            if identity_ops:
                kept = [identity_ops[0]] + kept
            else:
                from pyscf.pbc.symm.space_group import SPGElement
                kept = [SPGElement()] + kept

        # Validate closure
        is_closed, missing = _ops_closed(kept)
        if not is_closed:
            rot_strs = [str(m.rot.tolist()) for m in missing]
            raise ValueError(
                f"The requested operator set is not closed under multiplication. "
                f"Missing products (closure failure): {rot_strs}. "
                f"Add the missing operators or choose a valid subgroup."
            )

        return kept

    raise ValueError(f"Unknown mode {mode!r}. Expected one of 'none', 'symmorphic', 'indices'.")


def build_symmetry(cell, ops):
    """Construct a :class:`pyscf.pbc.symm.symmetry.Symmetry` carrying a reduced op set.

    Mirrors the attribute assignments performed by ``Symmetry.build`` but skips
    space-group re-detection: the caller supplies *ops* directly.  The resulting
    object is suitable for injection into ``cell.lattice_symmetry`` so that a
    subsequent ``cell.make_kpts(space_group_symmetry=True)`` uses the reduced set
    without re-running detection.

    The ``.cell`` back-reference is deleted before returning so that
    ``KPoints.build``'s ``__dict__.update(lattice_symmetry.__dict__)`` does not
    overwrite ``KPoints.cell`` with the cell used to build this object.

    Parameters
    ----------
    cell : pyscf.pbc.gto.Cell
        Built PySCF periodic cell used to compute Wigner D-matrices.
    ops : list of SPGElement
        Symmetry operators to inject (e.g. the output of :func:`select_ops`).
        Must include the identity; :func:`select_ops` guarantees this.

    Returns
    -------
    symm : pyscf.pbc.symm.symmetry.Symmetry
        A fully-built ``Symmetry`` object whose ``.ops`` is the supplied list,
        ``.nop`` matches, ``.has_inversion`` reflects the given ops, and
        ``.Dmats`` / ``.l_max`` are computed via ``make_Dmats``.  ``.cell`` is
        absent (deleted).
    """
    from pyscf.pbc.symm.symmetry import Symmetry, make_Dmats
    from pyscf.pbc.symm.space_group import SpaceGroup

    symm = Symmetry(cell)
    symm.ops = list(ops)
    symm.nop = len(symm.ops)
    symm.has_inversion = any(op.rot_is_inversion for op in symm.ops)
    op_rot = [op.a2r(cell).rot for op in symm.ops]
    symm.Dmats, symm.l_max = make_Dmats(cell, op_rot)
    symm._built = True

    # symmorphic: all ops have zero fractional translation (mirrors PySCF semantics).
    symm.symmorphic = all(op.trans_is_zero for op in symm.ops)

    # Build a SpaceGroup for this cell, then override its op list with the injected
    # ops so that symm.spacegroup.ops mirrors symm.ops.  Use the function-argument
    # `cell` (still in scope) rather than symm.cell which we are about to delete.
    sg = SpaceGroup(cell).build(dump_info=False)
    sg.ops = list(symm.ops)
    sg.nop = len(symm.ops)
    # Remove any live cell back-reference from the SpaceGroup.
    if hasattr(sg, "cell"):
        del sg.cell
    symm.spacegroup = sg

    # Remove the .cell back-reference so that KPoints.build's
    # __dict__.update(lattice_symmetry.__dict__) does not clobber KPoints.cell.
    del symm.cell
    return symm


def mode_from_args(args) -> str:
    """Translate CLI namespace flags to a :func:`select_ops` mode string.

    Parameters
    ----------
    args : argparse.Namespace or similar
        Parsed CLI arguments. Relevant attributes (accessed via ``getattr`` with
        safe defaults):

        * ``symmorphic_only`` (bool) -- keep only zero-translation operators.
        * ``symm_ops`` (str or None) -- comma-separated operator indices.

    Returns
    -------
    str
        One of ``"none"``, ``"symmorphic"``, or ``"indices"``.

    Raises
    ------
    ValueError
        If both ``symmorphic_only`` is truthy and ``symm_ops`` is set (not None).
    """
    symmorphic_only = getattr(args, "symmorphic_only", False)
    symm_ops = getattr(args, "symm_ops", None)

    if symmorphic_only and symm_ops is not None:
        raise ValueError(
            "--symmorphic_only and --symm_ops are mutually exclusive; "
            "pass only one of both flags."
        )

    if symmorphic_only:
        return "symmorphic"
    if symm_ops is not None:
        return "indices"
    return "none"


def spec_from_args(args):
    """Parse the ``--symm_ops`` CLI string into a list of integers.

    Parameters
    ----------
    args : argparse.Namespace or similar
        Parsed CLI arguments.  ``symm_ops`` attribute should be a
        comma-separated string of integers (e.g. ``"0,3,4,7"``), or ``None``.

    Returns
    -------
    list of int or None
        Parsed indices, or ``None`` when ``symm_ops`` is not set.
    """
    symm_ops = getattr(args, "symm_ops", None)
    if symm_ops is None:
        return None
    return [int(x.strip()) for x in symm_ops.split(",") if x.strip()]


def apply_symmetry_restriction(cell, args) -> None:
    """Detect the space group, select a subset of operators, and inject them.

    This is the top-level orchestrator called from :func:`pbc_cell` after the
    cell is fully built.  It is a no-op when ``args.space_symm`` is ``False``.

    Parameters
    ----------
    cell : pyscf.pbc.gto.Cell
        Fully built PySCF periodic cell.
    args : argparse.Namespace or similar
        Parsed CLI arguments.  Relevant attributes:

        * ``space_symm`` (bool) -- master switch; if ``False``, returns immediately.
        * ``symmorphic_only`` (bool, default ``False``) -- symmorphic filter.
        * ``symm_ops`` (str or None, default ``None``) -- comma-separated indices.
        * ``symm_backend`` (str, default ``"pyscf"``) -- detection backend.

    Side effects
    ------------
    Sets ``cell.lattice_symmetry`` to a :class:`pyscf.pbc.symm.symmetry.Symmetry`
    carrying the selected operator subset.  Logs before/after op counts and the
    group label.
    """
    space_symm = getattr(args, "space_symm", False)
    symmorphic_only = getattr(args, "symmorphic_only", False)
    symm_ops = getattr(args, "symm_ops", None)

    if not space_symm:
        if symmorphic_only or symm_ops is not None:
            warnings.warn(
                "--symmorphic_only / --symm_ops are ignored because "
                "--space_symm is False.",
                UserWarning,
                stacklevel=2,
            )
        return

    # Call mode_from_args first so mutual-exclusivity is always validated.
    mode = mode_from_args(args)

    if mode == "none":
        return   # no reduction requested; PySCF's own lattice_symmetry is already correct

    backend = getattr(args, "symm_backend", "pyscf")
    full = detect_space_group(cell, backend=backend)
    n_before = len(full.ops)
    logging.info(
        "apply_symmetry_restriction: detected %d ops, group=%s",
        n_before,
        full.point_group_symbol,
    )

    spec = spec_from_args(args)
    ops = select_ops(full.ops, mode, spec, cell)
    n_after = len(ops)

    logging.info(
        "apply_symmetry_restriction: keeping %d/%d ops (mode=%r)",
        n_after,
        n_before,
        mode,
    )

    cell.lattice_symmetry = build_symmetry(cell, ops)

    if mode == "indices":
        warnings.warn(
            "An explicit operator subgroup was selected via --symm_ops. "
            "k-space datasets (HF/Fock-k, symmetry/k/*) will be built "
            "from the reduced group, but auxcell q-space datasets "
            "(symmetry/q/*, e.g. k_sym_transform_j2c / k_sym_transform_p0) "
            "are still derived from the full detected symmetry in this phase. "
            "q-space and aux-derived quantities may be inconsistent for a "
            "point-group reduction. Validate carefully before consuming them.",
            UserWarning,
            stacklevel=2,
        )


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


def format_symmetry_ops(cell, info) -> str:
    """Format a human-readable table of space-group operators.

    Parameters
    ----------
    cell : pyscf.pbc.gto.Cell
        Built PySCF periodic cell (used to compute atom permutations via
        :func:`generate_permutation_info`).
    info : SpaceGroupInfo
        Detected space-group data as returned by :func:`detect_space_group`.

    Returns
    -------
    str
        A multi-line string with:

        * A header line containing the group label(s) (point-group symbol, and
          international symbol / number when available).
        * A column-header line including the word ``perm``.
        * One line per operator showing: index, flattened 3×3 rotation (9
          integers), fractional translation (3 floats), atom permutation, and a
          ``symmorphic`` flag (``True`` when ``op.trans_is_zero``).

        The text satisfies ``text.count("\\n") >= len(info.ops)`` (one line per
        operator plus the header lines).
    """
    parts = [info.point_group_symbol]
    if info.international_symbol is not None:
        parts.append(info.international_symbol)
    if info.international_number is not None:
        parts.append(f"#{info.international_number}")
    header = "Space-group operators: " + "  ".join(parts)

    col_header = (
        f"{'idx':>4}  {'rot (3x3 flattened)':^29}  {'trans (frac)':^28}  "
        f"{'perm':^20}  symmorphic"
    )

    lines = [header, col_header]
    for i, op in enumerate(info.ops):
        rot_str = " ".join(f"{v:2d}" for v in op.rot.flatten())
        trans_str = " ".join(f"{v:8.5f}" for v in op.trans)
        partner_idx, _ = generate_permutation_info(cell, op)
        perm_str = str(partner_idx.tolist())
        symmorphic = op.trans_is_zero
        lines.append(
            f"{i:4d}  [{rot_str}]  [{trans_str}]  {perm_str:^20}  {symmorphic}"
        )

    return "\n".join(lines)


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
        X = finp[dset][()]
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
