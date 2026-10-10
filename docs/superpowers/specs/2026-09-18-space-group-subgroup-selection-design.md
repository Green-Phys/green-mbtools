# Space-group subgroup selection for symmetry-lowered (e.g. Neel) calculations

Date: 2026-09-18
Status: Design, pending review
Module: `green_mbtools/mint`

## 1. Motivation

The mint module (v1.0.0) auto-detects the space group of a periodic system and, when `--space_symm true`, stores only irreducible-BZ (IBZ) quantities plus the symmetry operators needed for the downstream Green code to reconstruct the full BZ. This is efficient but it forbids studying electronic states whose symmetry is lower than the lattice, most importantly collinear antiferromagnetic (Neel) order in a supercell.

In the old workflow the user built a supercell of two primitive cells and let the mean field break the symmetry to produce the AFM solution. In the current code the mean field is still free to break symmetry (see Section 3), but the stored IBZ data plus symmetry operators re-impose the full lattice symmetry during reconstruction, symmetrizing the AFM solution away. `check_kspace_symmetry_breaking` is the tripwire that fires in this case.

This design adds the ability to lower the symmetry group that mint stores and reconstructs against, so that a symmetry-broken (AFM) solution round-trips faithfully while retaining as much symmetry as is compatible with it (for cost). It is deliberately a foundation for future magnetic (Shubnikov) group support.

## 2. Background: how symmetry flows today

1. `common_utils.pbc_cell(args)` builds the PySCF `Cell` with `space_group_symmetry=args.space_symm`. `cell.build()` calls `build_lattice_symmetry()` which sets `cell.lattice_symmetry = Symmetry(cell).build(space_group_symmetry=True, symmorphic=cell.symmorphic)`.
2. `common_utils.init_k_mesh(args, cell)` calls `cell.make_kpts(nk, space_group_symmetry=..., time_reversal_symmetry=...)`, producing the `kstruct` (`KPointsSymmetry`).
3. Everything downstream (IBZ reduction, `stars`, and the AO-space operators written to `k_sym_transform_ao` via `symmetry_utils.get_representation`) is driven by `kstruct.ops`, the list of `SPGElement`s.
4. The q-mesh (`init_q_mesh` -> `kpt_utils.build_q_struct` -> `make_kpts(cell, ...)`) and the auxcell operators (`store_auxcell_kstruct_ops_info` -> `init_q_mesh(auxcell, ...)`) build their own kstructs from the same or a sibling cell.

### The load-bearing PySCF hook (verified)

`KPoints.build` (`pyscf/pbc/lib/kpts.py:977`) does:

```python
if space_group_symmetry:
    _lattice_symm = getattr(self.cell, 'lattice_symmetry', None)
    if isinstance(_lattice_symm, symm.Symmetry):
        self.__dict__.update(_lattice_symm.__dict__)
if not self._built:
    symm.Symmetry.build(self, space_group_symmetry, *args, **kwargs)
```

It copies `cell.lattice_symmetry.__dict__` wholesale, including `ops`, `nop`, `Dmats`, `has_inversion`, and `_built=True`. Because `_built` comes across as `True`, `make_kpts` skips re-detection and uses whatever `ops` we placed on the cell. This was confirmed empirically: restricting a 48-op cubic group to {E, inversion} grew the k-IBZ from 4 to 8 through `make_kpts(space_group_symmetry=True)` with no re-detection.

`Symmetry.ops` is documented (`pyscf/pbc/symm/symmetry.py:145`) as "may be a subset of the operators in the space group", so injecting a subset is a supported use, not a hack.

PySCF is a detector, not a selector: `SpaceGroup.build()` always derives ops from the geometry (native `pyscf` backend or `spglib` backend). There is no API to impose a named group. This is fine, because any group we can validly store must be a subgroup of the detected group anyway (Section 9).

## 3. Where the AFM constraint actually lives (verified)

`common_utils.solve_mean_field` (`common_utils.py:868`) builds `args.mean_field(mycell, mydf.kpts)`, a plain `KUKS`/`KUHF` on the full BZ, not a symmetry-adapted `KsymAdapted*`. So the SCF is free to find the AFM solution. The symmetry constraint that destroys AFM is entirely in storage plus reconstruction: mint writes IBZ-reduced quantities and the `k_sym_transform_ao` operators, and Green rebuilds the full BZ assuming those operators hold. Lowering the stored group is therefore the correct and sufficient lever.

## 4. Goals and non-goals

Goals:

1. Build a custom `Symmetry` object with a reduced operator set and inject it into `cell.lattice_symmetry` at (or right after) cell construction, so all downstream kstructs inherit it.
2. `--symmorphic_only`: keep only operators with zero fractional translation. Covers the simple translational-doubling supercell.
3. Explicit subgroup selection by operator index (the general lever for AFM where sublattices are related by a point operation).
4. `--print_symmetry_ops`: a print-and-exit diagnostic listing every detected operator (index, rotation, fractional translation, atom permutation, symmorphic flag) plus the group label, so the user can see which operator to drop.
5. Optional `spglib` detection backend, giving international symbol/number now and a path to magnetic groups later.
6. Apply the reduced group to the main cell, so both its k-mesh and its q-mesh inherit it. The auxcell is deliberately left auto-detecting in this phase (Section 5.2); whether it needs the same treatment is answered empirically from real simulation data, not assumed.

Non-goals (YAGNI for this phase):

- Magnetic (Shubnikov) groups and antiunitary operators. The seam is shaped for them (Section 12) but they are not implemented here.
- Synthesizing a space group from a number/symbol not implied by the geometry (Section 9 explains why this is never needed for the non-magnetic case).
- Symmetry-adapting the SCF. Not required; the SCF is already free.
- Full subgroup-lattice enumeration. The print tool plus index selection is sufficient.

## 5. Architecture: one symmetry-provider seam

A single new module-level facility owns "produce the reduced operator set and install it on a cell". All symmetry lowering routes through it, so the main cell, q-mesh, and auxcell stay in lockstep.

Proposed home: extend `green_mbtools/mint/symmetry_utils.py` (it already owns `get_representation` and related operator logic).

### 5.1 Core functions

```
detect_space_group(cell, backend="pyscf", symprec=SYMPREC) -> SpaceGroupInfo
    Detect the full group. backend in {"pyscf", "spglib"}.
    Returns ops (sorted, deterministic), point-group symbol, and
    international symbol/number when available (spglib).

select_ops(full_ops, mode, spec, cell) -> list[SPGElement]
    Apply the restriction:
      mode="none"           -> full_ops
      mode="symmorphic"     -> [op for op in full_ops if op.trans_is_zero]
      mode="indices"        -> full_ops[i] for i in spec, then validate closure
    Always guarantees identity is present. Validates that the result is
    closed under composition (a subgroup); raises with a clear message if not.

build_symmetry(cell, ops) -> pyscf.pbc.symm.Symmetry
    Construct a Symmetry carrying `ops` with Dmats computed for THIS cell's
    basis (op_rot = [op.a2r(cell).rot for op in ops]; make_Dmats(cell, op_rot)).
    Sets nop, has_inversion, _built=True. Deletes the back-reference to cell
    before returning (see 5.3).

apply_symmetry_restriction(cell, args) -> None
    Orchestrator. No-op when args.space_symm is False. Otherwise:
      full = detect_space_group(cell, args.symm_backend, ...)
      ops  = select_ops(full.ops, mode_from_args(args), spec_from_args(args), cell)
      cell.lattice_symmetry = build_symmetry(cell, ops)
    Logs the before/after operator counts and the group labels.
```

`mode_from_args`/`spec_from_args` translate the CLI flags (Section 7) into a mode and spec, and enforce mutual exclusivity (Section 10).

### 5.2 Injection points

- Main cell (this phase): call `apply_symmetry_restriction(c, args)` at the end of `common_utils.pbc_cell(args)`, immediately after `c.build()`. This matches the user's stated preference to define and assign the new `Symmetry` object at cell-construction time. Because the k-mesh and the main-cell q-mesh both build kstructs from this same cell, they inherit the reduced group automatically.
- Auxcell (deferred, left as-is): the auxcell is a separate `Cell` (from `addons.make_auxmol`) built inside `store_auxcell_kstruct_ops_info`, and it continues to auto-detect the full group in this phase. We do not modify it preemptively. Rationale and risk boundary: for `--symmorphic_only` the point group is untouched, so the auxcell q-IBZ is unchanged and the only possible difference is which operator `stars_ops` selects; the mismatch, if any, is small and observable. For point-group index reduction (`--symm_ops`) the q-IBZ genuinely differs, so leaving the auxcell at full symmetry can desync the q-space reconstruction from the k-space one. Rather than assume, we generate real simulation data on the auxcell q-mesh and inspect how it behaves against the symmetry of the solution (Section 11, Section 13). If it desyncs, applying `apply_symmetry_restriction(auxcell, args)` before `init_q_mesh` is the one-line follow-up, and `build_symmetry` already handles the aux `Dmats`.

### 5.3 Implementation notes

- After `build_lattice_symmetry`, PySCF deletes `lattice_symmetry.cell` to avoid a circular reference and to prevent `__dict__.update` from overwriting the `KPoints.cell`. Our `build_symmetry` must do the same: delete the `.cell` attribute (and `.spacegroup.cell` if present) before assigning to `cell.lattice_symmetry`.
- `select_ops` for `mode="symmorphic"` reproduces PySCF's native `cell.symmorphic=True` path. We route it through the same helper (rather than setting `cell.symmorphic`) so the main cell and auxcell share one mechanism and so it composes with the spglib backend and index selection uniformly. The equivalence to `cell.symmorphic=True` is asserted in tests.
- Operator ordering must be deterministic for index selection to be meaningful and for cell/auxcell consistency. `SpaceGroup.build` calls `self.ops.sort()`, so ops are already sorted; we rely on that and keep symprec/backend fixed across cell and auxcell.

## 6. Files touched

- `green_mbtools/mint/symmetry_utils.py`: new `detect_space_group`, `select_ops`, `build_symmetry`, `apply_symmetry_restriction`, and `format_symmetry_ops` (the print helper). Reuses existing `generate_permutation_info` to report atom permutations.
- `green_mbtools/mint/common_utils.py`:
  - `add_pbc_params`: new CLI flags (Section 7).
  - `pbc_cell`: call `apply_symmetry_restriction` after build.
  - new `print_symmetry_ops(args)` wrapper (parallels `print_high_symmetry_points`) that builds the cell, calls `format_symmetry_ops`, prints, and returns.
  - `store_auxcell_kstruct_ops_info`: unchanged in this phase (auxcell keeps auto-detection). Flagged for a possible one-line follow-up pending empirical results.
- `green_mbtools/mint/pyscf_init.py`: wire `--print_symmetry_ops` into the early print-and-exit path (near `evaluate_high_symmetry_path` / the `print_high_symmetry_points` handling).
- Tests under `tests/` (Section 11).

## 7. CLI design

Added to `add_pbc_params`:

- `--symmorphic_only` (bool, default false): keep only zero-translation operators. Ignored with a warning if `--space_symm false`.
- `--symm_ops` (string, default None): comma-separated operator indices to keep, e.g. `--symm_ops 0,1,6,7`. Indices refer to the list printed by `--print_symmetry_ops`. The kept set is validated for closure; identity is added if missing.
- `--symm_backend` (choice {pyscf, spglib}, default pyscf): detection backend. Preserves current behavior by default.
- `--print_symmetry_ops` (flag, print-and-exit): list operators and group labels, then exit, mirroring `--print_high_symmetry_points`.

Mutual exclusivity: `--symmorphic_only` and `--symm_ops` are two ways of computing the same reduced set; specifying both is an error (Section 10). A convenience `--point_group SYMBOL` selector is explicitly deferred (Section 13) because a symbol does not disambiguate orientation, which is exactly the choice that matters for magnetic order.

### Example: Neel supercell workflow

```
# 1. See the operators and identify which one swaps the two sublattices
green_mbtools ... --space_symm true --print_symmetry_ops

# 2a. Simple translational doubling: drop all inter-cell (non-symmorphic) ops
green_mbtools ... --space_symm true --symmorphic_only

# 2b. General AFM: keep only the sublattice-preserving subgroup by index
green_mbtools ... --space_symm true --symm_ops 0,3,4,7
```

## 8. Data flow and consistency invariants

- Single source of truth (main cell): the reduced ops installed on `cell.lattice_symmetry`. The k-mesh and the main-cell q-mesh inherit it via `make_kpts` reading the cell.
- Auxcell is intentionally outside the seam this phase. It auto-detects the full group, so the k-space datasets reconstruct against the reduced group while the aux/q datasets reconstruct against the full group. This is a known, deliberate inconsistency we are measuring rather than assuming away (Section 5.2). It is expected to be benign for `--symmorphic_only` and is the primary thing the empirical run inspects for `--symm_ops`.
- Stored datasets follow automatically from whatever group each cell carries: `bz2ibz`, `ibz2bz`, `stars`, `k_sym_transform_ao` come from the reduced main-cell kstructs; `k_sym_transform_j2c`, `k_sym_transform_p0` come from the (currently full-symmetry) auxcell q-struct. No HDF5 format change; only sizes and contents change (larger IBZ for point-group reduction; different selected operators for symmorphic reduction).

## 9. Physics and correctness notes

- Detect then restrict, never synthesize. The operations that leave `H_k`/`S_k` invariant are exactly the detected space group. Imposing an operator outside that set breaks the reconstruction identity `X(k) = U X(k_ir) U^dagger`, which is what `check_kspace_symmetry_breaking` guards. Any valid selection is therefore a subgroup of the detected group. The "my coordinates are slightly off ideal" case is a geometry/symprec problem, not a reason to impose extra operators.
- Symmorphic-only is origin-dependent and narrower than it looks. Pure lattice translations map k to k, so they never enlarge k-stars and never change `nkpts_ibz`; they also never appear in `stars_ops`. The operators that actually mix the two magnetic sublattices in a stored star are the non-symmorphic ones ({inversion|t}, {mirror|t}), which map k to -k while permuting sublattices. With a sublattice-preserving origin these are exactly the ops `--symmorphic_only` drops, which is why it works for translational doubling: it forces `stars_ops` to select sublattice-preserving operators, changing `k_sym_transform_ao` even though `nkpts_ibz` is unchanged. With a poorly chosen origin a genuine point operation can acquire a translation and be dropped too. Validation must therefore inspect stored operators and residuals, not `nkpts_ibz`.
- General AFM needs index selection. When sublattices are related by a point operation (inversion/rotation), that operation is symmorphic and survives `--symmorphic_only`; the user must drop it explicitly via `--symm_ops`. The print tool exists to identify it (the atom-permutation column shows which operators swap sublattices).
- The principled object is the magnetic group. The sublattice swap is a symmetry only when combined with time reversal (swap A/B, flip spin back). That is a Shubnikov group with antiunitary elements, deferred to a later phase.

## 10. Error handling and validation

- No-op safety: `apply_symmetry_restriction` returns immediately when `args.space_symm` is False, and warns if `--symmorphic_only` or `--symm_ops` was set in that case.
- Mutual exclusivity: setting both `--symmorphic_only` and `--symm_ops` raises a clear argument error.
- Index bounds: `--symm_ops` indices are range-checked against the detected operator count; out-of-range raises with the valid range and a pointer to `--print_symmetry_ops`.
- Closure: `select_ops` verifies the kept set is closed under composition (products land back in the set, modulo lattice translation) and contains identity. If not closed, it either errors (default) or, behind an explicit choice, reports the closure it would need to add. Default is to error, so the user gets exactly the group they asked for or a clear reason why not.
- Acceptance tripwire: `check_kspace_symmetry_breaking` remains the end-to-end check. After a symmetry-lowered run, its residuals for `HF/H-k`, `HF/S-k`, `HF/Fock-k` must stay below threshold for the symmetry-broken solution, confirming the stored group is consistent with the solution.

## 11. Testing

Unit tests (extend `tests/symmetry_test.py` and friends):

- `select_ops` symmorphic mode equals PySCF native `cell.symmorphic=True` ops for a known supercell.
- `select_ops` index mode returns the requested operators and rejects a non-closed set.
- `build_symmetry` produces a `Symmetry` whose `make_kpts` yields the expected `nkpts_ibz` for a restricted group (regression on the cubic 48 -> {E, i} case: IBZ 4 -> 8).
- `apply_symmetry_restriction` is a no-op when `space_symm=False`.

Empirical validation (the load-bearing test):

- A concrete collinear-AFM supercell (candidate: a doubled cell with two magnetic sites, magmom up/down). Run mint three ways: full symmetry, `--symmorphic_only`, and an explicit `--symm_ops` subgroup. Assert that (a) full symmetry trips `check_kspace_symmetry_breaking` for the AFM solution, (b) the lowered-symmetry runs do not, and (c) the AFM moment survives the IBZ-to-full-BZ round trip. The exact structure and magmom are an open item (Section 13) and should be pinned before implementation.
- Auxcell behavior is a specific output of this run, not an assumption. With the main cell lowered but the auxcell left at full symmetry, inspect the aux/q quantities (`k_sym_transform_j2c`, `k_sym_transform_p0`, and a real polarization/self-energy on the q-mesh) against the symmetry of the solution. Decide from that evidence whether the auxcell needs `apply_symmetry_restriction` too, or whether the traced aux/q quantities are insensitive to it.

## 12. Phasing and future work: magnetic groups

The seam is shaped so that magnetic groups slot in as an additional detection source, not a rewrite:

- `detect_space_group` gains a magnetic backend using `spglib.get_magnetic_symmetry`/`get_magnetic_symmetry_dataset`, driven by `cell.magmom`. It returns operators tagged with a time-reversal (black/white) flag.
- `Symmetry`/operator handling and `get_representation` gain an antiunitary path (complex conjugation combined with the spatial operator), and the stored operator convention grows a per-operator TR tag. The existing x2c==2 TR handling in `store_kstruct_ops_info` (the `(u_spinor @ theta).conj()` branch) is the template.
- The downstream Green C++ consumer must learn to apply antiunitary operators during reconstruction. That is a coordinated cross-repo change and is out of scope here.

Until then, `--symm_ops` on the unitary group plus a sensible initial `--dm0`/magmom guess gives the user working AFM supercell calculations.

## 13. Open questions

1. Empirical validation case: which specific supercell geometry and magmom pattern do we standardize on for the AFM round-trip test?
2. Auxcell treatment: pending the empirical run, does the auxcell q-struct need the same reduced group, or are the traced aux/q quantities insensitive to it? Current design: leave auxcell auto-detecting and measure. The follow-up, if needed, is a single `apply_symmetry_restriction(auxcell, args)` call before `init_q_mesh`.
3. Non-closed selections: default to hard error, or offer an opt-in auto-close that reports what it added? Current design: hard error by default.
4. Do we want the convenience `--point_group SYMBOL` selector in this phase, accepting that it resolves to "first matching subgroup, with the selected operators printed for confirmation", or defer entirely to index selection? Current design: defer.
5. Does the Green C++ consumer make any implicit assumption about a minimum symmetry or a fixed IBZ size that a lowered group could violate? Needs a consumer-side check before rollout.
