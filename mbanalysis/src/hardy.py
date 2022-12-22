import os
import numpy as np
from multiprocessing import cpu_count, Process
import mbanalysis.sobolev as sobolev_exe
from scipy.optimize import minimize


# Find the number of cpus to use for parallelization of analtyic continuation
slurm_env_var = 'SLURM_JOB_CPUS_PER_NODE'
phys_ncpu = cpu_count()

if slurm_env_var in os.environ:
    _ncpu = int(
        os.environ['SLURM_JOB_CPUS_PER_NODE']
    )
else:
    print(
        "Variable SLURM_JOB_CPUS_PER_NODE not found. "
        + "Using half of the physical number of threads available."
    )
    _ncpu = int(phys_ncpu // 2)


def sobolev_wrapper(
    hardy_params, coeff_file='coeff.txt', n_real=10001, w_min=-10., w_max=10.,
    eta=0.01, param_file='params.txt', spectral_file='A_new.txt', lagr=1e-5
):
    """Python wrapper for target function sobolev.
    """
    n_params = len(hardy_params) * 2
    # Write the hardy parameters to param_file
    with open(param_file, 'w') as fp:
        for i in range(len(hardy_params)):
            fp.write(str(hardy_params[i].real) + '\n')
            fp.write(str(hardy_params[i].imag) + '\n')

    # run the sobolev
    norm = sobolev_exe.sobolev(
        coeff_file, n_params, n_real, w_min, w_max, eta,
        param_file, spectral_file, lagr
    )

    return norm


def optimize(
    params_in, tol=1e-8, max_iter=2000, coeff_file='coeff.txt',
    spectral_file='A_new.txt', n_real=100001, w_min=-10., w_max=10.,
    eta=0.01, lagr=1e-5
):
    """Nevanlinna analytic continuation is performed separately for each
    k-point, spin-index and generally also each AO (or SAO).
    This function performs Hardy optimization for one k, spin and AO index.
    NOTE: here, we assume we are already in the working directory.
    """
    # params_out = fmin_cg(
    #     lambda x: sobolev_wrapper(
    #         hardy_params=(x[0::2] + 1j * x[1::2]), coeff_file=coeff_file,
    #         n_real=n_real, w_min=w_min, w_max=w_max, eta=eta,
    #         spectral_file=spectral_file, lagr=lagr
    #     ), params_in, gtol=tol, maxiter=max_iter
    # )
    res = minimize(
        lambda x: sobolev_wrapper(
            hardy_params=(x[0::2] + 1j * x[1::2]), coeff_file=coeff_file,
            n_real=n_real, w_min=w_min, w_max=w_max, eta=eta,
            spectral_file=spectral_file, lagr=lagr
        ), params_in, method='Nelder-Mead', tol=tol,
        options={'maxiter': max_iter, 'disp': True}
    )
    print("Optimization result: ", res.success)

    return res.x


def hardy_optimization(
    n_basis=25, tol=1e-8, max_iter=2000, nevanlinna_dir='Nevanlinna',
    coeff_file='coeff.txt', spectral_file='A_new.txt',
    n_real=10001, w_min=-10., w_max=10., eta=0.01, lagr=1e-5
):
    """Perform Hardy optimization for Nevanlinna analytic continued data.
    The input parameters are:
        n_basis         :   number of Hardy basis funcs (default: 25)
        tol             :   tolerance for convergence (default: 1e-8)
        max_iter        :   max number of iterations (default: 2000)
        nevanlinna_dir  :   parent directory where Nevanlinna data is stored
        coeff_file      :   Nevanlinna-Schur coefficient filename
        spectral_file   :   filename for storing new spectral function
        n_real          :   number of real frequency points
                            (should be same as original Nevanlinna calculation)
        w_min, w_max    :   interval for real frequencies
        eta             :   broadening
        lagr            :   lagrange multiplier value in sobolev function
    """
    # Change working directory
    wkdir = os.path.abspath(os.getcwd())
    os.chdir(nevanlinna_dir)

    # info about dimensions
    dims = np.loadtxt('dimensions.txt').astype(int)
    num_points = int(np.prod(dims))
    params = np.zeros((4 * n_basis), dtype=np.float64)
    A_w = np.zeros((n_real, num_points))

    # loop over each folder
    processes = []
    p_num = 0
    for i in range(num_points):
        # change working directory
        os.chdir(str(i))
        # define process
        p = Process(
            target=optimize,
            args=(params,),
            kwargs={
                'tol': tol, 'max_iter': max_iter, 'coeff_file': coeff_file,
                'spectral_file': spectral_file, 'n_real': n_real,
                'w_min': w_min, 'w_max': w_max, 'eta': eta, 'lagr': lagr
            }
        )
        # start process
        p.start()
        processes.append(p)
        p_num += 1
        # if number of processes = number of cpus,
        # then wait for all the processes to finish before starting next ones
        if p_num % _ncpu == 0:
            for p in processes:
                p.join()
            processes = []
        os.chdir('..')

    # wait for rest of the processes to finish
    for p in processes:
        p.join()

    for i in range(num_points):
        os.chdir(str(i))
        # load the optimized spectral data
        # -- this should correspond to the latest spectral file
        X_wsk = np.loadtxt(spectral_file)
        A_w[:, i] = X_wsk[:, 1]
        os.chdir('..')

    # real frequency data
    freqs = X_wsk[:, 0]
    # reshape A_w
    if np.size(dims) == 1:
        output_shape = (len(freqs), dims)
    else:
        output_shape = (len(freqs), ) + dims
    A_w = A_w.reshape(output_shape)

    # change back to working directory
    os.chdir(wkdir)

    return freqs, A_w
