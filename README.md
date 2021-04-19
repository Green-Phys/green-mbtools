mb_analysis
===================

2021-04-18

mb_analysis provides functions for post processing of Green's-function-based many-body calculations. 
* The main classe is the mb class for both spin-unpolarized and spin-polarized calculations. 
It takes Fock matrix and self-energy as inputs.
  
Features
----------
* Compute molecular/natural orbitals.
* Mulliken analyasis based on symmetrical atomic orbitals.
* Fourier transformation between real and reciprocal space.
* Fourier transformation between imaginary-time and Matsuabara frequency domain.

To-Do-List
-----------
* Wannier interpolation
* Analytical Continuation that use external library such as Maxent and Nevanlinna.