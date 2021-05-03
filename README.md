mb_analysis
===================

2021-05-03

mb_analysis provides functions for post processing of Green's-function-based many-body calculations. 
* The main classe is the mb_post class for both spin-unpolarized and spin-polarized calculations. 
* Check `examples` for examples to start. 
  
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