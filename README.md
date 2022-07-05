mbanalysis
===================

Python package for post processing of Green's-function-based many-body calculations. 
* The main classe is the mb_post class for both spin-unpolarized and spin-polarized calculations. 
* Check `examples` for examples to start. 
  
Installation
----------
This package requires two levels of installation:
* The python package can be installed using `python setup.py install`.
* The Nevanlinna analytic continuation is set up as a C++ extension module. In the new version of this package, no additional installation is required for the Nevanlinna module.

Features
----------
* Compute molecular/natural orbitals.
* Mulliken analyasis based on symmetrical atomic orbitals.
* Fourier transformation between real and reciprocal space.
* Fourier transformation between imaginary-time and Matsuabara frequency domain.
* Wannier interpolation
* Maxent analytical continuation (External C++ library required: https://github.com/CQMP/Maxent) 
* Nevanlinna analytical continuation (External C++ library in `Nevanlinna`, authored by Jiani Fei [Phys. Rev. Lett. 126, 056402 (2021)].

To-Do-List
-----------
* There are two separate functions for Nevanlinna analytical continuation: `analyt_cont/nevan_run` and `analyt_cont/nevan_run_selfenergy`. Only difference now is the value of the parameter `spectral` (1 for Green's function continuation, and 0 for self-energy) and the final HDF5 output file. Remove this redundancy and merge into one code.

* Hardy optimization: Read the `coeff` file and optimize `theta[M+1] (z)` within the python package itself. I believe that we shouldn't need more than `np.complex128` precision to implement this.

* Fix `examples/winter*.py`: The original example for Silicon has been modified to Hydrogen (for which test data is available). The high-symmetry points in the interpolation need to be readjusted accordingly.