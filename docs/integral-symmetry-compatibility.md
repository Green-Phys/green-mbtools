# Integral-symmetry compatibility

The opt-in integral-symmetry feature uses PySCF private periodic GDF interfaces.
It requires PySCF **2.14.x**, with **2.14.0** pinned in the feature CI matrix
(Python 3.8 and 3.12; the existing general CI retains its own Python selection). This feature boundary does not narrow the existing
legacy paths' package requirements. Other PySCF versions fail explicitly before
constructing physical symmetry geometry. Captured CCGDF builders additionally
check the private method signatures and metric decomposition shapes.

The floating input k coordinates are checked for finite values and consistent
Cartesian/scaled ordering, then rationalized within a stated tolerance and
denominator bound. Subsequent momentum actions use exact integers. Affine
translations retain arbitrary coordinate origins; group-wide consistency is
checked before use. Orbit descriptors are immutable arrays.

Diagnostic output reports the enabled time reversal and exchange flags.
`matched_baseline_representatives` applies those same flags without spatial
reduction. The stored `legacy_representatives` count is historical input data;
its flags must not be inferred from its count alone.
