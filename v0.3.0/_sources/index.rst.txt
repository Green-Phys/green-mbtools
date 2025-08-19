.. green-mbtools documentation master file, created by
   sphinx-quickstart on Mon Jul 21 18:42:26 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Overview
========

The `green-mbtools` is a python tool suite for Green's-function-based many-body calculations implemented in the
`Green Software Package <https://www.Green-Phys.org>`_.

This package contains two main modules:

- `mint` (Mean-field INput generation Toolkit):
   generates input files for
   `Green/WeakCoupling <https://github.com/Green-Phys/green-mbpt>`_ using PySCF, ASE and SPGLib 
- `pesto` (Post-processing Evaluation Software TOols):
   provides access to various post-processing tools such as analytical continuation, wannier interpolation
   and Mulliken analysis.


Quickstart
----------

To install the `green-mbtools` binary package simply execute

.. code-block:: bash

   pip install green-mbtools

To get started with the post-processing of `green-mbpt` output, you can initialize the ``MB_post`` class using the
input, output and Matsubara grid (aka IR grid) files:

.. code-block:: python

   from green_mbtools.pesto import mb
   import matplotlib.pyplot as plt
   import numpy as np

   if __name__ == "__main__":
       # path to files
       input_path = "path/to/input.h5"
       output_path = "path/to/sim.h5"
       ir_path = "path/to/ir/grid.h5"

       # initialize the MB_post class
       my_mb = mb.initialize_MB_post(output_path, input_path, ir_path)

       # use my_mb to do other tasks
       # e.g., Nevanlinna analytic continuation to get spectral function on real-axis
       # with frequencies between -5 and 5 a.u. with 1001 grid points
       # and broadening parameter of 0.01
       freqs, Aw = my_mb.AC_nevanlinna(n_real=1001, w_min=-5.0, w_max=5.0, eta=0.01)

       # trace over spin, k-points and orbitals
       Aw_traced = np.einsum('wska -> w', Aw)

       # plot the density of states
       freqs_ev = freqs * 27.211  # convert frequency from a.u. to eV
       plt.plot(freqs_ev, Aw_traced)
       plt.show()


Features
--------

The `mbtools` package comes with two sub-packages: `mint` and `pesto`. The `mint` sub-package contains utilities for the following tasks:

* Generate input for Green/WeakCoupling.
* Compute molecular/natural orbitals.
* Mulliken analyasis based on symmetrical atomic orbitals.

On the other hand, the `pesto` sub-package, formerly known as the `mbanalysis` package, caters to:

* Fourier transformation between real and reciprocal space
* Fourier transformation between imaginary-time and Matsuabara frequency domain on intermediate representation (IR) grids (see `Phys. Rev. B 101, 035144 (2020) <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.101.035144>`_ for further details)
* Wannier interpolation
* Maxent analytical continuation using the `external C++ library <https://github.com/CQMP/Maxent>`_
* Nevanlinna analytical continuation, based on Fei et.al. `Phys. Rev. Lett. 126, 056402 (2021) <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.126.056402>`_
* PES analytical continuation, based on `Phys. Rev. B 107, 075151 (2023) <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.107.075151>`_
* Caratheodory analytical continuation for complex valued matrices, implemented as a C++ library based on Fei et.al. `Phys. Rev. B 104, 165111 (2021) <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.104.165111>`_


Authors
-------
The package is currently maintained by `Green-Phys <https://github.com/Green-Phys>`_, including `@egull <https://github.com/egull>`_
and `@gauravharsha <https://github.com/gauravharsha>`_.

Complete list of contributors is available on the `Github page <https://github.com/Green-Phys/green-mbtools/graphs/contributors>`_.


Documentation for older versions
--------------------------------

* `0.3.0 (latest) <https://Green-Phys.github.io/green-mbtools>`_

.. toctree::
   :maxdepth: 3
   :caption: Contents:
   :glob:
   :hidden:

   install
   examples
   source_code_doc
   Github <https://github.com/Green-Phys/green-mbtools>
