[![Python application](https://github.com/Green-Phys/MB_analysis/actions/workflows/python-app.yml/badge.svg)](https://github.com/Green-Phys/MB_analysis/actions/workflows/python-app.yml)

mbanalysis
===================

Python package for post-processing of Green's-function-based many-body calculations. 
  
Installation
----------
* Install the dependencies (see below).
* Modify the `include_dirs` flag in the `setup.py` file. Two example setup files are already provided.
*  To install the `mbanalysis` package simply execute:
  ```bash
  python setup.py install
  ```
  This will automatically build the Nevanlinna and Caratheodory extensions, which are written in C++.

**Dependencies:**
1. Python packages: `numpy`, `scipy`, `h5py`, `irbasis`, `cvxpy`, and `baryrat`.
2. C++ libraries: `Eigen`, `GMP`, `MPFR`, and `MPFR-C++` ([GitHub link for MPFR-C++](https://github.com/advanpix/mpreal))
3. `pytest` for testing
4. [Maxent external C++ librar](https://github.com/CQMP/Maxent) needs to be built separately to run analytic continuation tasks using the maxent method.

Features
----------
* Compute molecular/natural orbitals.
* Mulliken analyasis based on symmetrical atomic orbitals.
* Fourier transformation between real and reciprocal space.
* Fourier transformation between imaginary-time and Matsuabara frequency domain on IR grids.
* Wannier interpolation
* Maxent analytical continuation (External C++ library required: https://github.com/CQMP/Maxent) 
* Nevanlinna analytical continuation, based on Fei et.al. [Phys. Rev. Lett. 126, 056402 (2021)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.126.056402).
* PES analytical continuation, based on [Phys. Rev. B 107, 075151 (2023)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.107.075151).
* Caratheodory analytical continuation for complex valued matrices (External C++ library in `Caratheodory`, authored by Fei et.al. [Phys. Rev. B 104, 165111 (2021)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.104.165111).

To-Do-List
-----------
* Migrate to newer IR grid format
* Re-code Nevanlinna and Caratheodory for GNU Licence compliance before making the code public
* Improved README and documentation on where to find examples (perhaps a sub-README in the examples folder),
* Replace multiple setup files with one
* (for current implementation of Nevanlinna) re-integrate Hardy optimization
* Fix `examples/winter*.py`: The original example for Silicon has been modified to Hydrogen (for which test data is available). The high-symmetry points in the interpolation need to be readjusted accordingly.


Contirbutions to the code
-----------
* Most of the source code was written by @cnyeh
* Nevanlinna analytic continuation was implemented by @jianiOnly
* Clean up and packaging of the code and extension modules, including Nevanlinna and Caratheodory, was performed by by @gauravharsha
* ES analytic continuation implemented by @gauravharsha
* Example for spaghetti diagrams for occupation numbers added by @pavel-po
