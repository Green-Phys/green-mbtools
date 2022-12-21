import os
import subprocess
import shutil
import numpy as np
import h5py
from multiprocessing import cpu_count, Process
import mbanalysis.nevanlinna as nevan_exe
import mbanalysis.sobolev as sobolev_exe


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
    hardy_params, coeff_file='coeff', n_real=10001, w_min=-10., w_max=10.,
    eta=0.01, param_file='params.txt', spectral_file='A.txt', lagr=1e-5
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


# def hardy_optimization(
#     n_basis=25, tol=1e-8, max_iter=2000,
#     nevanlinna_dir='Nevanlinna', coeff_file='coeff'
# ):
#     return None
