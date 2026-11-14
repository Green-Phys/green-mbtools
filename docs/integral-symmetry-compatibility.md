# Integral-symmetry compatibility

The opt-in integral-symmetry feature uses PySCF private periodic GDF interfaces.
It requires PySCF **2.14.x**, with **2.14.0** pinned in the feature CI matrix
(Python 3.8, 3.12 and 3.13). This feature boundary does not narrow the existing
legacy paths' package requirements. Other PySCF versions fail explicitly before
constructing physical symmetry geometry. Captured CCGDF builders additionally
check the private method signatures and metric decomposition shapes.

The corrected legacy capture requires **green-igen 0.2.8**. It runs the original
routine's Python code with a per-call globals dictionary, private GDF subclass,
and private Cholesky capture proxy. Shared PySCF/green-igen classes and module
references are never patched. Calls with separate output files can coexist;
the tests exercise overlapping captures and failure exits. External files and
provider instances are owned by the caller, not shared between captures.

`test_integral_physical_pair_map.py` always constructs a small complete PySCF
reference, then checks real spatial phases, all four TR/exchange combinations,
stored bases, slices and cross-pair Coulomb interactions through `PhysicalPairMap`.
Its negative control injects a wrong pair-dependent map before reconstruction.
It requires no private workstation fixtures or opt-in environment variable.

The floating input k coordinates are checked for finite values and consistent
Cartesian/scaled ordering, then rationalized within a stated tolerance and
denominator bound. Subsequent momentum actions use exact integers. Affine
translations retain arbitrary coordinate origins; group-wide consistency is
checked before use. Orbit descriptors are immutable arrays.

Diagnostic output reports the enabled time reversal and exchange flags.
`matched_baseline_representatives` applies those same flags without spatial
reduction. The stored `legacy_representatives` count is historical input data;
its flags must not be inferred from its count alone.
