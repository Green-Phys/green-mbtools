[![Python application](https://github.com/Green-Phys/MB_analysis/actions/workflows/python-app.yml/badge.svg)](https://github.com/Green-Phys/MB_analysis/actions/workflows/python-app.yml)
[![GitHub license](https://img.shields.io/github/license/Green-Phys/green-mbpt?cacheSeconds=3600&color=informational&label=License)](./LICENSE)

```

 █▀▀█ █▀▀█ █▀▀ █▀▀ █▀▀▄
 █ ▄▄ █▄▄▀ █▀▀ █▀▀ █  █
 █▄▄█ ▀ ▀▀ ▀▀▀ ▀▀▀ ▀  ▀

 █▀▄▀█ █▀▀█ █▀▀▄ █  █     █▀▀█ █▀▀█ █▀▀▄ █  █ 　 ▀▀█▀▀ █▀▀█ █▀▀█ █   █▀▀
 █ █ █ █▄▄█ █  █ █▄▄█ ▀▀  █▀▀▄ █  █ █  █ █▄▄█ 　   █   █  █ █  █ █   ▀▀█
 █   █ ▀  ▀ ▀  ▀ ▄▄▄█     █▄▄█ ▀▀▀▀ ▀▀▀  ▄▄▄█ 　   █   ▀▀▀▀ ▀▀▀▀ ▀▀▀ ▀▀▀

```

===================

Python Tool Suite for Green's-function-based many-body calculations using Green Software Package.

Package contains two main modules:

   - mint (Mean-feald INput generation Toolkit): generates input files for Green/WeakCoupling using PySCF, ASE and SPGLib 
   - pesto (Post-processing Evaluation Software TOols): provides access to various post-processing tools such as analytical continuation, and Mulliken analysis.
  
Installation
----------
*  To install the `green-mbtools` binary package simply execute:
  ```bash
  pip install green-mbtools
  ```

*  To build from sources download the source code either from github or from [PyPI](https://pypi.org/project/green-mbtools/#files) and run:
  ```bash
  python -m build
  ```

  This will automatically build the Nevanlinna and Caratheodory extensions, which are written in C++.

**Dependencies:**
1. Python packages: `numpy`, `scipy`, `h5py`, `green-ac`, `pyscf`, `ase`, `spglib`, `irbasis`, `cvxpy`, and `baryrat`.
2. C++ libraries: `Eigen`, `GMP`, `MPFR`
3. `pytest` for testing
4. [Maxent external C++ librar](https://github.com/CQMP/Maxent) needs to be built separately to run analytic continuation tasks using the maxent method.

Features
----------
* Generate input for Green/WeakCoupling.
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
* Improved README and documentation on where to find examples (perhaps a sub-README in the examples folder),
* (for current implementation of Nevanlinna) re-integrate Hardy optimization
* Fix `examples/winter*.py`: The original example for Silicon has been modified to Hydrogen (for which test data is available). The high-symmetry points in the interpolation need to be readjusted accordingly.


Contirbutions to the code
-----------
* Most of the `pesto` source code was written by @cnyeh
* Nevanlinna analytic continuation was implemented by @jianiOnly
* Clean up and packaging of the code and extension modules, including Nevanlinna and Caratheodory, was performed by by @gauravharsha
* ES analytic continuation implemented by @gauravharsha
* Example for spaghetti diagrams for occupation numbers added by @pavel-po
* C++ wrapper to GMP and MPFR float numbers and packaging @iskakoff
* Green/WeakCoupling input generation @cnyeh and @iskakoff
