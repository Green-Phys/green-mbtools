# GREEN density-fitting space-group format v1

Optional `time_reversal_enabled` and `exchange_enabled` attributes record the
writer's allowed actions; representative stabilizers use only those actions.
Older archives' explicit per-pair actions still define reconstruction.
Python readers check factor shape, float64 packing and the 128 MiB source
buffer limit before loading a chunk. Expansion applies the same capacity
bound and checks input `nk`, `nso`, `NQ` and any existing correlation gauge ID
before creating output. Cache budgets do not replace the source-buffer bound.

This opt-in format is `green.df.space_group.v1`. Legacy mode remains the default.
An SG directory has `meta.h5` and `SGVQ_<representative_start>.h5`; it has no
legacy `chunk_size` dataset, `VQ_*` files, or legacy version alias. Both old
readers must reject it. Input legacy k/q/pair metadata retains its meaning.

Coordinated minimum consumers on `Chenan-Phys` feature branches:
`green-symmetry` at `82b5c67a5c2f6d53dd0fbf80178eb1a234879041`,
`green-mbpt` at `f480f4ec7bb5ed601f678aed9c8a45903afa5701`, and
`green-gpu` at `0110ef3037e4fd92f58a3d2fdba9f876549b38c9` (or its documentation
successor). This feature is opt-in and is not merged upstream. Use all coordinated
consumer revisions before producing SG files for a solver run. Full-rank metrics
alone are insufficient: actual factors must satisfy the preserving group's
covariance at the configured finite-grid tolerances; a failed stabilizer rejects
representative production.

The initial domain is scalar, spherical, three-dimensional ordinary/Ewald
Coulomb, with square orbital basis changes and captured full positive lower
Cholesky factors. Spinors, negative or truncated metric sectors, and rectangular
orbital spaces are rejected. HF and corrected correlation archives have separate
set identities and actual metric factors. Equal dimensions do not identify a gauge.

All tensor data is complex128 packed as interleaved float64 real/imaginary values
along the last axis: `(nvalid,NQ,nao,2*nao)` for factors and `(n,N,2*N)` for
matrix batches. There is no extra last axis of length two. Pair orientation is
the producer's ordered `(ki,kj)` and auxiliary q is `kj-ki`. No normalization,
k weights, or contraction prefactors change.

Required root attributes are `integral_format`, `format_version=1`, `set_kind`
(`hf` or `correlation`), `finite_size_kind`, `orbital_basis`, `producer_revision`,
`input_fingerprint`, `auxiliary_gauge_id`, `complex_storage=complex128_interleaved_f64`,
`transform_direction=representative_to_requested`, and
`operation_order=spatial,time_reversal,exchange`. Unknown/missing attributes or
incompatible input/set identity are errors. Integral set IDs include gauge and
kernel provenance. The initial SG profile supports ordinary and default Ewald
sets. Special `df_ewald.h5`/`AqQ.h5` sidecars, GF2/GW special finite-size flags,
coarse graining and GW q=0 extrapolation are rejected for SG; legacy support
for those configurations is unchanged.

The `sg` group contains:

| Dataset | Shape/meaning |
|---|---|
| dimensions | nk, nao, NQ, nrep, nq, nops |
| mesh, q_mesh; mesh_denominator, q_denominator | Actual rational input coordinates, including wrapping |
| representative_pairs, pair_to_representative | nrep x 2; nk x nk ordered-pair map |
| pair_operation, time_reversal, exchange | nk x nk separate operation-chain fields |
| reciprocal_wraps | nk x nk x 2 x 3, transformed-minus-requested wrapping |
| pair_q | nk x nk actual captured gauge/q IDs |
| rotations, translation_numerator, translation_denominator | Full affine operations; no k-permutation deduplication |
| multiplication, translation_carries, inverses | Full group validation tables |
| orbital, auxiliary | Deduplicated sparse zero-phase angular/permutation blocks and row lattice phases for each operation |
| captured_C | nq x NQ x 2*NQ, actual lower factors used to whiten the source |
| stored_x, stored_x_inverse | Optional nk x nao x 2*nao, producer's square X basis maps |
| chunk_size, chunk_start, chunk_valid_count | Capacity and actual representative coverage, including the final partial chunk |

Each sparse group has `offsets` (nops+1), `rows`, `columns`, `values`
(interleaved f64), and `phase_vectors` (nops x nbasis x 3). Rows are target
and columns are source. Values contain the angular action at zero Bloch phase.
At scaled target momentum k, multiply row r by
`exp(2*pi*i*dot(k,phase_vectors[op,r]))`. Phase vectors are derived from actual
as-input atom coordinates and affine operations, and are lattice vectors within
the validated geometry tolerance.

For an operation on representative pair (i,j), first determine both spatial
target points using the same reciprocal rotation. TR then negates both momenta;
exchange then swaps their order. Let K conjugate when TR XOR exchange is true.
The auxiliary map is `D = solve(C_target, A_spatial^(K) C_source^(K))` and acts
in the actual target gauge. The factor is
`D [left K(V_rep)^(optional transpose) right^dagger]`. Orbital left/right maps
swap for exchange and conjugate for TR. For stored basis `X V X^dagger`, their
maps are `X_target U K_TR(X_source_inverse)` with source legs swapped as needed.

Target auxiliary slicing must still sum over every source auxiliary row. Raw
representative buffers never alias requested output. Dense auxiliary maps and
loaded metric factors are created lazily with an explicit byte-limited cache;
the native consumer default is 64 MiB per reader (Python standalone: 256 MiB).
GF2 has six readers, so their aggregate default cache budget is 384 MiB.
Cache keys include integral-set identity,
operation, source/target q, conjugation, precision and representation version.
The small integer maps and sparse operation blocks are separate descriptor memory.

Offline conversion reads an existing complete reference, validates every omitted
factor in its captured frame, then writes a fresh directory. It saves storage,
not original integral construction time. Expansion writes fresh legacy chunks
using the original legacy pair order and records the final chunk's valid count.
The native readers use the SG representative count, independently of legacy
one-body pair counts. Preload and chunked paths share the same host reconstruction.

The native flags are `--integral_symmetry space_group`,
`--integral_symmetry_cache_bytes N`, and `--integral_symmetry_preload_bytes N`.
The latter limits SG representative host preload per node (default 1 GiB),
with checked size products and MPI byte counts. Individual factor/chunk buffers
are bounded to 128 MiB. Full-rank positive Cholesky gauges and square orbital X
are supported; ED, truncated/negative metric sectors, rectangular X, X2C/spinors,
lower-dimensional and range-separated physical interactions are rejected.

Producer opt-in uses `--integral_symmetry space_group` and a fresh
`--integral_symmetry_work DIRECTORY`, alongside fresh input/HF/correlation paths.
The complete SCF CDERI remains separate. The representative CCGDF producer
captures all needed q metrics but requests only representative three-center pairs.
The default corrected q=0 blocks come from `green_igen.df._make_j3c`, restricted
to diagonal representatives. Its actual Cholesky solve is captured through a
scoped module proxy; class hooks and module ownership are restored on failure.
Nonzero-q representatives come from the ordinary builder. A PySCF builder's
`exx="ewald"` setting is not interchangeable with this legacy GREEN contract.
Reduced-q GW metadata is written in the actual correlation gauge and identified
by `integral_symmetry/correlation_gauge_id`. Existing stars and weights are kept.
Offline `examples/compress_df_integrals.py` streams a complete captured-frame
reference, retaining only bounded factor buffers and a metric LRU; a legacy
chunk set without its actual whitening factors cannot safely be compressed.

The representative-producer integration supplies an automated physical q=0 regression in
`tests/test_integral_default_ewald_physics.py`. It calls the original
`green_igen.df._make_j3c` directly for every reference diagonal, independently
of the SG capture wrapper, and checks all logical factors and all fixed-q
cross-pair ERIs in small full-rank diamond and triclinic identity cases. It also requires a nonzero change
from the ordinary q=0 interaction. This is an opt-in compute test:
`GREEN_RUN_PHYSICAL_SG_TESTS=1 python -m pytest -q
tests/test_integral_default_ewald_physics.py --basetemp /path/to/scratch/fresh-pytest`.
An optional fresh `GREEN_SG_PHYSICS_FIXTURE_DIR` retains native input/reference
fixtures under `diamond/` and `identity/`, each with `hf/` and `correlation/`
subdirectories for the coordinated C++ reader tests. The identity case must
have no extra spatial reduction beyond the legacy TR/exchange pair count.
The complete-reference exports also support frozen solver comparisons.
Run integral builds on the
workstation, not on a laptop. Successful execution, rather than merely adding
the test, is required before treating its physical coverage as closed.
