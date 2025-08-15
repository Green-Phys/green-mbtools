import numpy as np
import time
import h5py
from os.path import abspath

from green_mbtools.pesto import mb
from green_mbtools.pesto import orth
from green_mbtools.pesto.analyt_cont import es_nevan_run, g_iw_projection


#
# Example
# Perform analytical continuation using pole estimation
# and semidefinite optimization (ES).
#

if __name__ == "__main__":
    
    #
    # Input parameters
    #
    
    # Inverse temperature
    T_inv = 1000
    
    # Input files
    data_path = abspath('../tests/test_data')
    input_path = data_path + '/H2_GW/input.h5'
    sim_path = data_path + '/H2_GW/sim.h5'
    ir_file = data_path + '/ir_grid/1e4.h5'
    
    #
    # Read input data
    #
    
    # Read converged imaginary time calculation
    print("Reading simulation data.")
    f = h5py.File(sim_path, 'r')
    it = f["iter"][()]
    Sigma1r = f["iter" + str(it) + "/Sigma1"][()].view(complex)
    Sigmar = f["iter" + str(it) + "/Selfenergy/data"][()].view(complex)
    Gr = f["iter" + str(it) + "/G_tau/data"][()].view(complex)
    mu = f["iter" + str(it) + "/mu"][()]
    f.close()
    print("Completed reading simulation data.")
    
    # Read data about grids
    print("Reading mean-field data")
    f = h5py.File(input_path, 'r')
    Hk = f['HF/H-k'][()].view(complex)
    Hk = Hk.reshape(Hk.shape[:-1])
    Sk = f['HF/S-k'][()].view(complex)
    Sk = Sk.reshape(Sk.shape[:-1])
    mo_coeff = f["/HF/mo_coeff"][()]
    ir_list = f["/grid/ir_list"][()]
    index = f["/grid/index"][()]
    conj_list = f["grid/conj_list"][()]
    f.close()
    print("Completed reading mean-field data.")
    
    # All k-dependent matrices should lie on a full Monkhorst-Pack grid.
    # Transform from reduced BZ to full BZ
    print("Transforming data from reduced BZ to full BZ.")
    Sigma1 = mb.to_full_bz(Sigma1r, conj_list, ir_list, index, 1)
    Sigmak = mb.to_full_bz(Sigmar, conj_list, ir_list, index, 2)
    Gk = mb.to_full_bz(Gr, conj_list, ir_list, index, 2)
    Fk = Sigma1 + Hk
    print('Pre analysis complete')
    
    #
    # Use ES analytical continuation to get spectral function
    #
    
    # 0. Construct mbanalysis object
    print("Setting up mbanalysis post processing object.")
    t1 = time.time()
    MB = mb.MB_post(fock=Fk, sigma=Sigmak, gtau=Gk, mu=mu, S=Sk, beta=T_inv, ir_file=ir_file)
    t2 = time.time()
    print("Time required to set up post processing: ", t2 - t1)
    
    # 1. construct G in orthogonal basis
    t3 = time.time()
    gtau_sao = orth.sao_orth(Gk, Sk, type='g')
    
    # 2. fourier transform to g_iw
    giw_sao = MB.ir.tau_to_w(gtau_sao)
    iw_vals = MB.ir.wsample
    n_iw = len(iw_vals)
    
    # 3. Extract green's function in upper complex plane
    giw_sao_pos = giw_sao[n_iw//2:]
    iw_pos = iw_vals[n_iw//2:]
    
    # 4a. (Optional) Perform projection
    do_projection = False
    if do_projection:
        g_iw_proj = g_iw_projection(
            giw_sao_pos, iw_pos, diag=False
        )
    
    # 4. perform analytic continuation.
    # The input parameters for the function have the following meaning:
    #   1.  G_iw        :   imaginary frequency green's function
    #   2.  wsample     :   (real part) of the imaginary frequencies
    #   3.  n_real      :   number of grid-points to use on the real freuency axis
    #   4.  w_min       :   lowest real frequency to get results for
    #   4.  w_max       :   highest real frequency to get results for
    #   5.  eta         :   broadening for real-axis.
    #   6.  diag        :   value of True will perform analytic continuation only
    #                       for the diagonal values of Green's function
    #                       (precisely what we need for spectral function)
    #                       (NOTE: we can diagonalize the G_iw first then input
    #                       with diag=False as well)
    #   7. parallel     :   How to parallelize (over openMP) the analytic cont.
    #                       'sk' meanse we parallelize over spin and k-ponts.
    #                       'ska' would parallelize over spin, k and orbitals
    #                       Typically, 'sk' is most optimal, because the
    #                       analyic continuation is vectorized over orbtal indices
    #   8. solver       :   ES uses semi-definite programming to perform
    #                       analytical continuation. Different solvers can be
    #                       employed, e.g., 'SCS', 'MOSEK', 'CLARABEL'.
    #   9. solver_opts  :   A dictionary of options, e.g., tolerance, can be
    #                       passed to this function as well
    freqs, gw_sao_diag, es_error = es_nevan_run(
        G_iw=giw_sao_pos, wsample=iw_pos, n_real=1000, w_min=-5, w_max=5, eta=0.01,
        diag=True, parallel='sk'
    )
    
    # 5. Step 4 results in a real-frequeny Green's function of shape
    #    (n_real, ns, nk, nao). We still need to obtain the spectral function.
    A_w = -(1 / np.pi) * np.imag(np.einsum('wska -> wk', gw_sao_diag))
    
    t4 = time.time()
    print("Time required for Nevanlinna AC: ", t4 - t3)
    
    f1 = h5py.File('es_dos_sao.h5', 'w')
    f1['freqs'] = freqs
    f1['A_w'] = A_w
    f1['A_w_opt'] = A_w
    f1.close()
    