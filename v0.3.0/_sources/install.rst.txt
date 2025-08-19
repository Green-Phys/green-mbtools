Installation
============

To install the `green-mbtools` binary package simply execute

.. code-block:: bash

   pip install green-mbtools

To build from sources download the source code either from github or from
`PyPI <https://pypi.org/project/green-mbtools/#files>`_ and run

.. code-block:: bash
   
   git clone https://github.com/Green-Phys/green-mbtools.git
   python -m build

This will automatically build the Nevanlinna and Caratheodory extensions, which are written in C++.

Dependencies
------------

1. Python packages: `numpy`, `scipy`, `h5py`, `green-ac`, `pyscf`, `ase`, `spglib`, `irbasis`, `cvxpy`, and `baryrat`.
2. C++ libraries: `Eigen`, `GMP`, `MPFR`
3. `pytest` for testing
4. The `Maxent external C++ library <https://github.com/CQMP/Maxent>`_ needs to be built separately to run analytic continuation tasks using the maxent method.


.. Contents
   ========

.. Add your content using ``reStructuredText`` syntax. See the
   `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
   documentation for details.