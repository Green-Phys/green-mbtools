Examples
========

Several `example scripts <https://github.com/Green-Phys/green-mbtools/tree/master/examples>`_ are provided in the
Github repo of the green-mbtools package, particularly for the post-processing of Green output.
Here, we will go over some of the basic cases.


Initialization
--------------

For analyzing the output of Green's function simulation using, e.g. GW, GF2, or self-energy embedding theory,
we use the ``MB_post`` class from ``green_mbtools.pesto.mb``.
The script `examples/MB_post_example.py <https://github.com/Green-Phys/green-mbtools/blob/master/examples/MB_post_example.py>`_
demonstrates the key steps in initializing the post-processing class:

1. Read the input and output files from a Green simulation, and initialize the many-body post processing class:

   .. code-block:: python
   
      from green_mbtools.pesto import mb
   
      if __name__ == "__main__":
          fname_inp = '/path/to/input.h5'
          fname_sim = '/path/to/sim.h5'
          ir_file = '/path/to/ir.h5'
   
          mb_obj = mb.initialize_MB_post(sim_path=fname_sim, input_path=fname_inp, ir_file=ir_file)

2. Access Green's function, self-energy, and other information

   .. code-block:: python
  
      if __name__ == "__main__":

          # ... initialization

          mu = mb_obj.mu()  # chemical potential
          gtau = mb_obj.gtau()  # Green's function
          sigma_tau = mb_obj.sigma  # self-energy on imaginary-time axis
          dm = mb_obj.dm()  # get density matrix from Green's function
  

Matsubara Transformation
------------------------

We are often interested in visualizing the Green's function and self-energy on the Matsubara frequency or :math:`i\omega` axis
rather than the imaginary-time or :math:`\tau` axis.
As the data in Green software package is stored on sparse grids also known as the intermediate representation (IR) grids,
we use the information from the IR grid file to execute the :math:`\tau \leftrightarrow i\omega` transformation.

Once an ``MB_post`` object is initialized, the transformation can be performed simply as:

.. code-block:: python

   if __name__ == "__main__":

       # ... initialization

       # tau to omega transform
       sigma_iw = mb_obj.ir.tau_to_w(sigma_tau)
       g_iw = mb_obj.ir.tau_to_w(gtau)
       # back transform
       sigma_tau_back = mb_obj.ir.w_to_tau(sigma_iw)
       gtau_back = mb_obj.ir.w_to_tau(g_iw)

The `examples/IR_transform.py <https://github.com/Green-Phys/green-mbtools/blob/master/examples/IR_transform.py>`_ script
also shows how such a transformation can be performed without initializing the ``MB_post`` object for one of the testing
data files.
The IR transform is supported by the class ``IR_factory``:

.. code-block:: python

   if __name__ == "__main__":

       # ... initialization

       # Data files
       sim_file = '../tests/test_data/H2_GW/sim.h5'
       f = h5py.File(sim_file, 'r')
       it = f["iter"][()]
       G_tau = f["iter" + str(it) + "/G_tau/data"][()].view(complex)
       tau_mesh = f["iter" + str(it) + "/G_tau/mesh"][()]
       f.close()

       # Here the H2 GW simulation uses lambda = 1e4
       ir_file = '../tests/test_data/ir_grid/1e4.h5'
       beta = tau_mesh[-1]  # inverse temperature in simulation
       nts = tau_mesh.shape[0]
       my_ir = ir.IR_factory(beta, ir_file)

       # Fourier transform from G(tau) to  G(iw_n)
       G_iw = my_ir.tau_to_w(G_tau)
       # Fourier transform from G(iw_n) to G(tau)
       G_tau_2 = my_ir.w_to_tau(G_iw)


Analytic continuation
---------------------

One of the primary steps in the post-processing of Green's function data is analytic continuation (AC) of the results from
imaginary time and frequency axis to the real frequency axis.
The `green-mbtools.pesto` module supports several implementations for AC based on theory of:

* Nevanlinna functions, see `examples/nevanlinna.py <https://github.com/Green-Phys/tree/master/examples/nevanlinna.py>`_,
* Caratheodory functions, and
* Pole estimation and semi-definite relaxation, see `examples/es_nevanlinna.py <https://github.com/Green-Phys/tree/master/examples/es_nevanlinna.py>`_.

In addition, an interface to the `Maxent <https://github.com/CQMP/Maxent>`_ AC library is also provided
(see `examples/maxent.py <https://github.com/Green-Phys/green-mbtools/tree/master/examples/maxent.py>`_.

Nevanlinna AC is one of the more robust approaches in our opinion and a direct interface is provided in the ``MB_post`` class.
Continuing with the above examples, we can directly obtain the spectral function on the real axis using:

.. code-block:: python

   if __name__ == "__main__":

       # ... initialization

       # w_min and w_max: min and max real frequency bracket (in a.u.)
       # n_real: number of frequency points to consider -- more poitns = smoother plots
       # eta: Broadening parameter
       n_real = 10001
       w_min = -5.0
       w_max = 5.0
       eta = 0.01
       freqs, Aw = mb_obj.AC_nevanlinna(n_real=n_real, w_min=w_min, w_max=w_max, eta=eta)

.. note:: Most AC implementations use parallelization using multiprocessing utilities in python.
   As a result, analytic continuation should always be placed inside the
   :code:`if __name__ == "__main__":` block.


Wannier interpolation
---------------------

For solid-state systems, simulations are generally performed on evenly sampled grid-points in the first Brillouin zone
in the reciprocal space.
However, the spectral function and band structure are generally desirable along a special, high-symmetry path in the same
Brillouin zone.
This is facilitated by `k`-point or Wannier interpolation, which is provided by ``green_mbtools.pesto.ft`` library.

Examples for interpolating the mean-field (e.g., DFT, Hartree-Fock) and the correlated calculation (e.g., GW, GF2, SEET)
results are provided in
`examples/winter_mean_field.py <https://github.com/Green-Phys/green-mbtools/tree/master/examples/winter_mean_field.py>`_
and
`examples/winter_correlated.py <https://github.com/Green-Phys/green-mbtools/tree/master/examples/winter_correlated.py>`_,
respectively.

Following the above exmple, once the ``MB_post`` object is initialized, the interpolation can be performed directly using
the function ``wannier_interpolation``.

.. code-block:: python

   from ase.spacegroup import crystal
   from green_mbtools.pest import mb

   if __name__ == "__main__":

       # initialization

       # obtain high-symmetry path for fictitious H2 solid example using ase
       cc = crystal(
           symbols=['H', 'H'],
           basis=[(-0.25, -0.25, -0.25), (0.25, 0.25, 0.25)],
           spacegroup=group,
           cellpar=[a, b, c, alpha, beta, gamma], primitive_cell=True
       )
       path = cc.cell.bandpath('GXMGR', npoints=100)
       kpts_inter = path.kpts

       # Obtain interpolated Green's function, self-energy, tau-mesh, Fock and overlap matrix
       G_tk_int, Sigma_tk_int, tau_mesh, Fk_int, Sk_int = MB.wannier_interpolation(
           kpts_inter, hermi=True, debug=debug
       )

.. note:: The accuracy of interpolation depends on the density of the original Brillouin zone grid.
   To reduce the errors, we recommend using PySCF to directly calculate the one-body Hamiltonain and the
   overlap matrix along the high-symmetry path rather than interpolation. See, e.g.,
   `examples/useful_scripts/nvnl_winter_analysis.py <https://github.com/Green-Phys/green-mbtools/tree/master/examples/useful_scripts/nvnl_winter_analysis.py>`_.


Other useful scripts
--------------------

For practical application, a combination of initialization, interpolation and analytic continuation is required.
We provide several python scripts to automate these tasks for solid-state and molecular systems.
These are available in the
`examples/useful_scripts <https://github.com/Green-Phys/green-mbtools/tree/master/examples/useful_scripts>`_ directory.
