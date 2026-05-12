# Double-Group Symmetry for X2C Calculations

## Current State

In `common_utils.py:store_kstruct_ops_info` (line 915–927), `x2c=2` stores only identity/theta
placeholders — no spatial symmetry is used. In `init_k_mesh` (line 627–629),
`space_group_symmetry=False` is forced for `x2c=2`.

## Double-Group Theory

For relativistic spinors, a rotation R must be represented in SU(2) (the double cover of SO(3)):

$$D^{1/2}(R) = \cos(\varphi/2)\,I_2 + i\sin(\varphi/2)\,(\hat{n}\cdot\boldsymbol{\sigma})$$

The full spinor AO representation is the Kronecker product:

$$U_{\text{spinor}}(R) = D^{1/2}(R) \otimes U_{\text{orbital}}(R)$$

where $U_{\text{orbital}}(R)$ comes from the existing `get_representation` in
`symmetry_utils.py` (which handles phases, atom permutations, and spherical harmonics
via `kstruct.Dmats`).

## Package Strategy

| Task | Tool |
|---|---|
| Space group ops (R, t) | spglib (via PySCF's `kstruct.ops`) |
| SU(2) spinor rotation $D^{1/2}(R)$ | **`irrep`** `symop.spinor_rotation` |
| Orbital rep $U_\text{orbital}(R)$ | existing `get_representation` |
| Little co-group for Herring | spglib `get_symmetry_dataset` + filtering |
| Herring classification / band irreps | **`spgrep`** |
| IBZ cross-check | spglib `get_ir_reciprocal_mesh` |

`irrep` (Tsirkin et al.) identifies the space group via spglib internally and attaches the
2×2 SU(2) spinor matrix $D^{1/2}(R)$ to each symmetry operation when initialised with
`spinor=True`. This eliminates any need to implement SU(2) algebra or handle the ±1
phase ambiguity — it is resolved consistently within `irrep`.

### PySCF cell → spglib cell helper

Needed wherever spglib or irrep is called directly:

```python
def cell_to_spglib(mycell):
    """Convert a PySCF Cell to the (lattice, positions, numbers) tuple spglib/irrep expect."""
    lattice   = mycell.lattice_vectors() * 0.529177249  # Bohr -> Angstrom
    positions = mycell.get_scaled_atom_coords().reshape(-1, 3)
    numbers   = np.array(mycell.atom_charges(), dtype=int)
    return (lattice, positions, numbers)
```

## Implementation Plan

### Step 1 — Add to `symmetry_utils.py`

#### 1a. Spinor representation using `irrep`

`irrep` replaces the need for any manual SU(2) derivation. The `SpaceGroup` object
built with `spinor=True` exposes `symop.spinor_rotation` directly:

```python
def get_spinor_representation(bz_idx, symm_op_idx, mycell, kstruct, irrep_sg):
    """Double-group spinor representation: D^{1/2}(R) ⊗ U_orbital(R).

    Parameters
    ----------
    irrep_sg : irrep.spacegroup.SpaceGroup
        Built once per cell with SpaceGroup(cell=cell_to_spglib(mycell), spinor=True).
        Operations in irrep_sg.symmetries must be index-aligned with kstruct.ops.
    """
    u_orbital = get_representation(bz_idx, symm_op_idx, mycell, kstruct)
    D_half    = np.array(irrep_sg.symmetries[symm_op_idx].spinor_rotation, dtype=complex)
    return np.kron(D_half, u_orbital)   # shape (nso, nso)
```

> **Note**: Verify that `irrep_sg.symmetries` and `kstruct.ops` are indexed in the same
> order (same spglib call underneath). A one-time check on a simple cubic system is
> sufficient.

#### 1b. Little co-group via spglib (needed for Herring criterion)

PySCF's `kstruct` exposes only one op per BZ k-point (`stars_ops_bz`), not the full
little co-group. spglib fills that gap:

```python
def little_group_ops(k_scaled, spglib_rotations, tol=1e-6):
    """Indices of spglib rotation ops that fix k modulo a reciprocal lattice vector."""
    little = []
    for i, R in enumerate(spglib_rotations):
        diff = R.T @ k_scaled - k_scaled
        if np.all(np.abs(diff - np.round(diff)) < tol):
            little.append(i)
    return little
```

#### 1c. Herring criterion

```python
def herring_criterion(k_scaled, mycell, kstruct, irrep_sg, spglib_rotations):
    """
    Classify a TR-invariant k-point as Herring type A, B, or C.

    Returns S = sum_{g in G_k} chi_spinor(g^2).
      S > 0  →  type A (no extra degeneracy)
      S = 0  →  type B (Kramers pairs)
      S < 0  →  type C (pairs combine into larger co-reps)
    """
    little_idxs = little_group_ops(k_scaled, spglib_rotations)
    S = 0.0
    for i in little_idxs:
        # chi(g^2) = Tr(D^{1/2}(R^2)) — use irrep's spinor_rotation directly
        su2_sq = np.array(irrep_sg.symmetries[i].spinor_rotation, dtype=complex) ** 2
        S += np.real(np.trace(su2_sq))
    return S
```

### Step 2 — Modify `store_kstruct_ops_info` in `common_utils.py`

Replace the `x2c == 2` block (lines 915–927). Build the `irrep` SpaceGroup once per
cell and pass it in, or build it locally:

```python
else:
    import spglib
    from irrep.spacegroup import SpaceGroup
    nso         = nao * 2
    kspace_orep = np.zeros((nk, nso, nso), dtype=np.complex128)
    theta       = np.kron(np.array([[0, 1], [-1, 0]], dtype=np.complex128), np.eye(nao))
    tr_conj_bz  = kstruct.time_reversal_symm_bz

    # Build irrep SpaceGroup once; spinor=True activates D^{1/2} on each op.
    # get_magnetic_symmetry gives a per-operation TR flag — needed for non-symmorphic
    # groups where gT (not pure T) may be in the little group.
    irrep_sg = SpaceGroup(cell=cell_to_spglib(mycell), spinor=True)
    mag_sym  = spglib.get_magnetic_symmetry(cell_to_spglib(mycell))
    op_is_tr = mag_sym['time_reversals'].astype(bool)

    for ik in range(nk):
        iop      = stars_ops[ik]
        u_spinor = get_spinor_representation(ik, iop, mycell, kstruct, irrep_sg)
        if tr_conj_bz[ik]:
            # Combined spatial+TR: store (U_dg(g)·Θ)* so that the reconstruction
            # X(k) = (Uk @ X_ir @ Uk†)* correctly gives U_dg·Θ·X_ir*·Θ†·U_dg†
            kspace_orep[ik] = (u_spinor @ theta).conj()
        else:
            kspace_orep[ik] = u_spinor
```

### Step 3 — Enable space-group symmetry in `init_k_mesh`

Remove the forced override and add a spglib cross-check:

```python
kstruct = mycell.make_kpts(args.nk, scaled_center=args.center,
                           space_group_symmetry=args.space_symm,
                           time_reversal_symmetry=args.tr_symm)

# Sanity check IBZ count against spglib directly
import spglib
mapping, _ = spglib.get_ir_reciprocal_mesh(
    args.nk, cell_to_spglib(mycell), is_time_reversal=args.tr_symm
)
n_ibz_spglib = len(np.unique(mapping))
assert n_ibz_spglib == kstruct.nkpts_ibz, (
    f"IBZ mismatch: spglib={n_ibz_spglib}, PySCF={kstruct.nkpts_ibz}. "
    "Check space_group_symmetry settings."
)
```

Remove the override print warning (lines 626–629).

## Red-Black / Herring Co-Representation

The Herring classification determines co-representation structure at TR-invariant
k-points ($k = -k + G$). The character sum from `herring_criterion()` (Step 1c)
maps directly to physical consequences for GW:

| Herring type | $S = \sum_{g \in G_k} \chi(g^2)$ | Physical meaning |
|---|---|---|
| **A** | $> 0$ | No extra degeneracy — standard treatment |
| **B** | $= 0$ | Kramers pairs — every level doubly degenerate |
| **C** | $< 0$ | Pairs combine into larger co-reps |

Types B and C impose Kramers pinning that the self-energy must respect.

## Caveats

1. **irrep op ordering**: `irrep_sg.symmetries` and `kstruct.ops` both derive from
   spglib but may not be indexed identically across versions. Validate alignment once
   with a simple system (e.g., FCC Cu with x2c=2) before production use.

2. **Spin-ordering convention**: `irrep` defines $D^{1/2}$ in a specific spin basis.
   Confirm that `np.kron(D_half, U_orbital)` ordering (all-α block then all-β block)
   matches PySCF's X2C spinor layout by checking against the existing
   `theta = kron([[0,1],[-1,0]], eye(nao))` convention.

3. **Non-symmorphic groups**: The translation phase in `get_representation`
   (`symmetry_utils.py:216`) handles the orbital part; the SU(2) from `irrep` is
   unaffected by fractional translations.

4. **Auxiliary cell (`store_auxcell_kstruct_ops_info`)**: The auxiliary basis is scalar
   (no spin), so `kspace_orep_p0` does not need the double-group extension — only the
   main-cell `k_sym_transform_ao` changes.
