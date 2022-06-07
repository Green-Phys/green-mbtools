[![Python application](https://github.com/gauravharsha/mbanalysis/actions/workflows/python-app.yml/badge.svg)](https://github.com/gauravharsha/mbanalysis/actions/workflows/python-app.yml)

mbanalysis
===================

Python package for post processing of Green's-function-based many-body calculations. 
* The main classe is the mb_post class for both spin-unpolarized and spin-polarized calculations. 
* Check `examples` for examples to start. 
  
Installation
----------
This package requires two levels of installation:
* The python package can be installed using `python setup.py install`.
* The Nevanlinna analytic continuation is set up as a C++ module. To build this module, run `cd Nevanlinna && make`.

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
* Compile Nevanlinna C++ module during the setup.py installation. Right now this module sits wherever the `mbanalysis` repo is cloned.
* Add test functions
* Problems with `example/quasiparticle.py`: First, it needs `nevan_sigma`, which needs to be added to the repository. Second, there is 
* Fix `examples/winter*.py`: The original example for Silicon has been modified to Hydrogen (for which test data is available). The high-symmetry points in the interpolation need to be readjusted accordingly.
