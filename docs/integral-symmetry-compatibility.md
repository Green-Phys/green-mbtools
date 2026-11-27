# Integral-symmetry compatibility

GREEN MBTools requires PySCF 2.14.x. The application CI runs the integral suite
with PySCF 2.14.0 on Python 3.9 and 3.13. Captured CCGDF builders check the
private method signatures and metric decomposition shapes.

Corrected legacy capture supports `green-igen>=0.2.8,<0.3` and checks the
Python routine's signature and captured Cholesky factor. Each call uses a
private GDF subclass and SciPy proxy. The returned provider reports the FFT
mesh used in the Ewald calculation. Omitting `mesh` uses green-igen's default.

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
