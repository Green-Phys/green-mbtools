import os
import subprocess
import shutil
import numpy as np
import h5py
from multiprocessing import cpu_count, Process
from .pes_utils import run_es, cvx_matrix_projection, cvx_diag_projection
from .ac_utils import dump_input_caratheodory_data, load_caratheodory_data
import green_mbtools.pesto.caratheodory as carath_exe
import green_ac

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


def maxent_run(
    gtau, tau_mesh, error=5e-3, params="green.param", exe_path='maxent',
    outdir="Maxent"
):
    """Maxent analytic continuation for G(iw) or diagonal of self-energy

    Input parameters
    ----------------
        gtau        :   contains matrix valued data on imaginary time axis
                        shape of gtau should be:  (nt, :)
        tau_mesh    :   tau grid points
        error       :   error threshold in optimization
        params      :   parameter file name to pass into maxent continuation
        exe_path    :   Path to maxent program provided by CQMP:
                        https://github.com/CQMP/Maxent
        outdir      :   output directory in which data will be stored

    Returns
    ----------------
        The maxent spectral function is stored in '<outdir>/DOS.h5'
    """

    wkdir = os.path.abspath(os.getcwd())
    # outdir = os.path.abspath(wkdir+'/'+outdir)
    print("Maxent output:", os.path.abspath(wkdir+'/'+outdir))

    beta = tau_mesh[-1]
    nts = tau_mesh.shape[0]
    g_shape = gtau.shape
    assert nts == gtau.shape[0], "Number of imaginary time points mismatches."
    gtau = gtau.reshape(nts, -1)
    dim1 = gtau.shape[1]

    if not os.path.exists(outdir):
        os.mkdir(outdir)
    os.chdir(outdir)

    # Save dim1 for better understanding of output
    np.savetxt("dimensions.txt", np.asarray(g_shape[1:], dtype=int))
    # FIXME Can we have multiple layers of folders so that one can separate
    # the whole job into chunks?
    for d1 in range(dim1):
        if not os.path.exists(str(d1)):
            os.mkdir(str(d1))
        np.savetxt(
            "{}/G_tau.txt".format(d1),
            np.column_stack(
                (tau_mesh, gtau[:, d1].real, np.array([error] * nts))
            )
        )

    # Start analytical continuation
    processes = []
    pp = 0
    for d1 in range(dim1):
        os.chdir(str(d1))
        try:
            shutil.copy(str(params), "./green.param")
        except Exception:
            shutil.copy(os.path.abspath(wkdir+'/'+params), "./green.param")

        with open("log.txt", "w") as log:
            p = subprocess.Popen(
                [
                    exe_path, "./green.param", "--DATA=G_tau.txt",
                    "--BETA=" + str(beta), "--NDAT=" + str(nts)
                ],
                stdout=log, stderr=log
            )
            processes.append(p)
        pp += 1
        if pp % _ncpu == 0:
            for proc in processes:
                proc.communicate()
            processes = []
        os.chdir("..")

    for proc in processes:
        proc.communicate()

    # Combine output
    dump_A = False
    for d1 in range(dim1):
        try:
            freqs = np.loadtxt("{}/green.out.maxspec.dat".format(d1))[:, 0]
        except IOError:
            pass
        else:
            dump_A = True
            break
    if dump_A:
        Aw = np.zeros((freqs.shape[0], dim1), dtype=float)
        for d1 in range(dim1):
            try:
                Aw[:, d1] = np.loadtxt(
                    "{}/green.out.maxspec.dat".format(d1)
                )[:, 1]
            except IOError:
                print(
                    "green.out.maxspec.dat is missing in {} folder. Possibly \
                    analytical continuation fails at that point.".format(d1)
                )
        Aw = Aw.reshape((freqs.shape[0],) + g_shape[1:])
        gtau = gtau.reshape(g_shape)
        f = h5py.File("DOS.h5", 'w')
        f["freqs"] = freqs
        f["DOS"] = Aw
        f["taumesh"] = tau_mesh
        f["Gtau"] = gtau
        f.close()
    else:
        print("All AC fails. Will not dump to DOS.h5")
        gtau = gtau.reshape(g_shape)

    os.chdir("..")


def nevan_run(
    X_iw, wsample, n_real=10000, w_min=-10, w_max=10,
    eta=0.01, spectral=True, prec=128
):
    """Nevanlinna analytic continuation for G(iw) or diagonal of self-energy
    Input parameters
    ----------------
        X_iw        :   contains matrix valued data on imaginary freq. points
                        shape of X_iw should be:  (nw, ns, nk, nao, nao)
        wsample     :   value of imaginary frequencies
        n_real      :   number of real frequency points
                        (used if custom_freqs = None)
        w_min       :   minimum value of real frequency range
                        (used if custom_freqs = None)
        w_max       :   maximum value of real frequency range
                        (used if custom_freqs = None)
        eta         :   broadening parameter
        spectral    :   returns real valued spectral function if True,
                        full complex valued function otherwise
        prec        :   Precision to use in Nevanlinna

    Returns:
    ----------------
        freqs       :   Real frequency grid on which AC data is obtained
        X_w         :   Real valued spectral function (if spectral=True)
                        or complex valued AC quantity (if spectral=False)
    """
    # print acknowledgments
    print("----------------------------------------------")
    print("Performing Nevanlinna analytic continuation.")
    print("Reference:")
    print("Fei et al, Phys. Rev. Lett. 126, 056402 (2021)")
    print("----------------------------------------------")

    nw = wsample.shape[0]
    assert nw == X_iw.shape[0], "Number of imaginary frequency points \
        mismatches between \"input_parser\" and Gw."
    X_iw_shape = X_iw.shape
    X_iw = X_iw.reshape(nw, -1)
    dim1 = X_iw.shape[1]

    freqs = np.linspace(w_min, w_max, n_real)
    X_w = green_ac.solve("Nevanlinna", wsample, freqs, X_iw, prec=prec, eta=eta)
    X_w = X_w.reshape((freqs.shape[0],) + X_iw_shape[1:])
    if spectral:
        X_w = -X_w.imag/np.pi

    return freqs, X_w


def caratheodory_run(
    X_iw, wsample, outdir='Caratheodory', custom_freqs=None,
    n_real=2001, w_min=-10, w_max=10, eta=0.01,
):
    """Caratheodory analytic continuation for G(iw) or self-energy

    Input parameters
    ----------------
        X_iw            : contains matrix valued data on imaginary freq. points
                        shape of X_iw should be: (nw, ns, nk, nao, nao)
        wsample         : value of imaginary frequencies
        outdir          : output directory in which data will be stored
        custom_freqs    : custom freq points on which to perform carath. AC
        n_real          : number of real frequency points
                          (used if custom_freqs = None)
        w_min           : minimum value of real frequency range
                          (used if custom_freqs = None)
        w_max           : maximum value of real frequency range
                          (used if custom_freqs = None)
        eta             : broadening parameter

    Returns:
    ----------------
        freqs       :   Real frequency grid on which AC data is obtained
        Xc_w        :   analytically continued matrix data
        XA_w        :   spectral function for AC data
    """

    # print acknowledgments
    print("----------------------------------------------")
    print("Performing Caratheodory analytic continuation.")
    print("Reference:")
    print("Fei et al, Phys. Rev. B 104, 165111 (2021)")
    print("----------------------------------------------")

    # set defaults
    ifile = 'X_iw.txt'
    matrix_ofile = 'X_c.txt'
    spectral_ofile = 'X_A.txt'

    # create input data
    wkdir = os.path.abspath(os.getcwd())
    print("Caratheodory output:", os.path.abspath(wkdir + '/' + outdir))
    print("Dumping input to: ", ifile)
    print("Dumping Caratheodory matrix output to: ", matrix_ofile)
    print("Dumping Caratheodory spectral output to: ", spectral_ofile)

    nw = wsample.shape[0]
    assert nw == X_iw.shape[0], "Number of imaginary frequency points \
        mismatches between \"input_parser\" and Gw."

    # whether or not to use custom grid for continuation
    use_custom_real_grid = 0  # False
    grid_file = 'grid_file.txt'
    if custom_freqs is not None:
        use_custom_real_grid = 1  # True
        n_real = custom_freqs.shape[0]

    # assuming that the input
    X_iw_shape = X_iw.shape
    _, ns, nk, nao = X_iw_shape[:4]

    # Build and move to working directory
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    os.chdir(outdir)

    # Dump input data to files for caratheodory
    dump_input_caratheodory_data(wsample, X_iw, ifile)

    # Start analytical continuation
    processes = []
    pp = 0
    dim1 = ns * nk
    for d1 in range(dim1):
        os.chdir(str(d1))
        if custom_freqs is not None:
            np.savetxt(grid_file, custom_freqs)
        p = Process(
            target=carath_exe.caratheodory,
            args=(
                ifile, nw, nao, matrix_ofile, spectral_ofile,
                use_custom_real_grid, grid_file, n_real, w_min, w_max, eta
            )
        )
        p.start()
        processes.append(p)
        pp += 1
        if pp % _ncpu == 0:
            for proc in processes:
                proc.join()
            processes = []
        os.chdir("..")

    for proc in processes:
        proc.join()

    # Combine output
    freqs, Xc_w, XA_w = load_caratheodory_data(
        matrix_ofile, spectral_ofile, X_iw_shape
    )

    # Get out of working directory
    os.chdir(wkdir)

    return freqs, Xc_w, XA_w


def es_nevan_run(
    G_iw, wsample, n_real=10000, w_min=-10, w_max=10, eta=0.01, diag=True,
    eps_pol=1.0, parallel='sk', outdir='PESNevan', solver='SCS', **solver_opts
):
    """ES analytic continuation for G(iw) or diagonal of self-energy.

    Input parameters
    ----------------
        G_iw        :   imaginary frequency green's function
        wsample     :   (real part) of the imaginary frequencies
        n_real      :   number of grid-points to use on the real freuency axis
        w_min       :   lowest real frequency to get results for
        w_max       :   highest real frequency to get results for
        eta         :   broadening for real-axis.
        diag        :   value of True will perform analytic continuation only
                        for the diagonal values of Green's function
                        (precisely what we need for spectral function)
                        (NOTE: we can diagonalize the G_iw first then input
                        with diag=False as well)
        eps_pol     :   Threshold on the imaginary part of poles
        parallel    :   How to parallelize (over openMP) the analytic cont.
                        'sk' meanse we parallelize over spin and k-ponts.
                        'ska' would parallelize over spin, k and orbitals
                        Typically, 'sk' is most optimal, because the
                        analyic continuation is vectorized over orbtal indices
        solver      :   ES uses semi-definite programming to perform
                        analytical continuation. Different solvers can be
                        employed, e.g., 'SCS', 'MOSEK', 'CLARABEL'.
        solver_opts :   A dictionary of options, e.g., tolerance, can be
                        passed to this function as well

    Returns
    ----------------
        w_vals      :   real frequency grid
        G_w         :   analytically continued Green's function on real grid
    """

    # print acknowledgments
    print("----------------------------------------------")
    print("Performing ES analytic continuation.")
    print("Reference:")
    print("Huang et al, Phys. Rev. B 107, 075151 (2023)")
    print("----------------------------------------------")

    # set defaults
    ofile = 'Aw.txt'

    # Handle directories for saving output files
    wkdir = os.path.abspath(os.getcwd())
    print("PES Nevanlinna output:", os.path.abspath(wkdir + '/' + outdir))
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    base_dir = wkdir + '/' + outdir

    # Bring G_iw in the shape: (iw, s, k, a, b)
    orig_shape = G_iw.shape
    if len(orig_shape) == 1:
        # this is simply G(iw)
        G_iw = G_iw.reshape((orig_shape[0], 1, 1, 1))
        print("Expecting dimensions to represent (w)")
    elif len(orig_shape) == 2:
        # assuming this is G(iw, a)
        nw, nao = orig_shape
        G_iw = G_iw.reshape((nw, 1, 1, nao))
        print("Expecting dimensions to represent (w, a)")
    elif len(orig_shape) == 4:
        # for diagonal approx only
        print("Expecting dimensions to represent (w, s, k, a)")
    elif len(orig_shape) == 5:
        # this is already the form we want
        print("Expecting dimensions to represent (w, s, k, a, b)")
    else:
        raise ValueError(
            'Incorrect shape G_iw. Expecting either 1-, 2-, 4- or 5-d array'
        )

    # Diagonal elements or not
    if diag and len(G_iw.shape) == 5:
        G_iw = np.einsum('wskaa -> wska', G_iw, optimize=True)
        # update the shape
        orig_shape = G_iw.shape

    # Check consistency in number of frequencies and input data
    nw, ns, nk = G_iw.shape[0:3]
    assert nw == wsample.shape[0], "Number of imaginary frequency points \
        mismatches between \"input_parser\" and Gw."

    # Pre-processing - divide the data based on k-, spin-, and AO-indices
    # for parallelized analytic continuation
    if parallel == 'ska':
        if not diag:
            raise ValueError(
                "Cannot use the 'ska' parallel scheme for full matrix"
            )
        dim1 = np.prod(G_iw.shape[1:])
        G_iw = G_iw.reshape(nw, dim1, 1)
    elif parallel == 'sk':
        dim1 = ns * nk
        G_iw = G_iw.reshape((nw, ns * nk) + G_iw.shape[3:])
    else:
        raise ValueError(
            "Invalid value for the argument 'parralel': {}.".format(parallel)
            + "Expecting 'ska' or 'sk'"
        )

    # Save dimensions for better understanding of output
    np.savetxt("dimensions.txt", np.asarray(G_iw.shape[1:], dtype=int))

    # define the frequency mesh
    w_vals = np.linspace(w_min, w_max, n_real)

    # perform analytical continuation
    processes = []
    pp = 0
    for d1 in range(dim1):
        if not os.path.exists(base_dir + '/' + str(d1)):
            os.mkdir(base_dir + '/' + str(d1))
        out_file = base_dir + '/{}/{}'.format(str(d1), ofile)
        p = Process(
            target=run_es,
            args=(
                wsample, G_iw[:, d1, :], w_vals
            ),
            kwargs={
                'diag': diag,
                'eta': eta,
                'eps_pol': eps_pol,
                'ofile': out_file,
                'solver': solver,
                **solver_opts
            }
        )
        p.start()
        processes.append(p)
        pp += 1
        if pp % _ncpu == 0:
            for proc in processes:
                proc.join()
            processes = []

    for proc in processes:
        proc.join()

    # read the computed spectrum and quantitie
    G_w = np.zeros((n_real, ) + G_iw.shape[1:], dtype=complex)
    error_es = np.zeros(dim1)
    pp = 0
    for d1 in range(dim1):
        # G_w output
        out_file = base_dir + '/{}/{}'.format(str(d1), ofile)
        Gw_here = np.loadtxt(out_file, dtype=complex)
        G_w[:, d1, :] = Gw_here.reshape((n_real, ) + G_iw.shape[2:])
        # G_w optimization error
        err_file = out_file.replace('.txt', '_error.txt')
        err_here = np.loadtxt(err_file)
        error_es[d1] = err_here

    # reshape and return
    G_w = G_w.reshape((n_real, ) + orig_shape[1:])
    error_es = error_es.reshape(orig_shape[1:])

    return w_vals, G_w, error_es


def g_iw_projection(
    G_iw, wsample, diag=True, solver='SCS', wcut=5, n_real=101, **solver_opts
):
    """Projection of G(iw) to a simple pole structure.

    Input parameters
    ----------------
        G_iw        :   imaginary frequency green's function
        wsample     :   (real part) of the imaginary frequencies
        diag        :   value of True will perform analytic continuation only
                        for the diagonal values of Green's function
                        (precisely what we need for spectral function)
                        (NOTE: we can diagonalize the G_iw first then input
                        with diag=False as well)
        solver      :   Projection uses semi-definite programming to perform
                        analytical continuation. Different solvers can be
                        employed, e.g., 'SCS', 'MOSEK', 'CLARABEL'.
        solver_opts :   A dictionary of options, e.g., tolerance, can be
                        passed to this function as well
        wcut        :   real frequency cutoff to employ in projection
        n_real      :   number of points to use in the range (-wcut, wcut)

    Returns
    ----------------
        G_iw        :   projected Matsubara Green's function

    Practical info
    ----------------
    For different practical applications, different CVXPy solvers may be
    required to achieve optimal results. Some options to consider:
    *   SCS (default)
    *   MOSEK
    *   CLARABEL
    For individual use, all three options are available free of use.
    """

    # print acknowledgments
    print("----------------------------------------------")
    print("Performing Projection of Matsubara Green's function.")
    print("Reference:")
    print("Huang et al, Phys. Rev. B 107, 075151 (2023)")
    print("----------------------------------------------")

    # Handle directories for saving output files
    outdir = 'Projection'
    ofile = 'Proj_Giw.txt'
    wkdir = os.path.abspath(os.getcwd())
    print("PES Nevanlinna output:", os.path.abspath(wkdir + '/' + outdir))
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    base_dir = wkdir + '/' + outdir

    # Bring G_iw in the shape: (iw, s, k, a, b)
    orig_shape = G_iw.shape
    if len(orig_shape) == 1:
        # this is simply G(iw)
        G_iw = G_iw.reshape((orig_shape[0], 1, 1, 1))
        print("Expecting dimensions to represent (w)")
    elif len(orig_shape) == 2:
        # assuming this is G(iw, a)
        nw, nao = orig_shape
        G_iw = G_iw.reshape((nw, 1, 1, nao))
        print("Expecting dimensions to represent (w, a)")
    elif len(orig_shape) == 4:
        # for diagonal approx only
        print("Expecting dimensions to represent (w, s, k, a)")
    elif len(orig_shape) == 5:
        # this is already the form we want
        print("Expecting dimensions to represent (w, s, k, a, b)")
    else:
        raise ValueError(
            'Incorrect shape G_iw. Expecting either 1-, 2-, 4- or 5-d array'
        )

    # Diagonal elements or not
    projection_function = cvx_matrix_projection
    if diag:
        projection_function = cvx_diag_projection
    if diag and len(G_iw.shape) == 5:
        G_iw = np.einsum('wskaa -> wska', G_iw, optimize=True)
        # update the shape
        orig_shape = G_iw.shape

    # Check consistency in number of frequencies and input data
    nw, ns, nk, nao = G_iw.shape[0:4]
    assert nw == wsample.shape[0], "Number of imaginary frequency points \
        mismatches between \"input_parser\" and Gw."

    # Pre-processing - divide the data based on k-, spin--indices
    # for parallelized projection of data
    dim1 = ns * nk
    G_iw = G_iw.reshape((nw, ns * nk) + G_iw.shape[3:])

    # Save dimensions for better understanding of output
    np.savetxt("dimensions.txt", np.asarray(G_iw.shape[1:], dtype=int))

    # perform analytical continuation
    processes = []
    pp = 0
    for d1 in range(dim1):
        if not os.path.exists(base_dir + '/' + str(d1)):
            os.mkdir(base_dir + '/' + str(d1))
        out_file = base_dir + '/{}/{}'.format(str(d1), ofile)
        p = Process(
            target=projection_function,
            args=(
                1j * wsample, G_iw[:, d1, :]
            ),
            kwargs={
                'w_cut': wcut,
                'n_real': n_real,
                'ofile': out_file,
                'solver': solver,
                **solver_opts
            }
        )
        p.start()
        processes.append(p)
        pp += 1
        if pp % _ncpu == 0:
            for proc in processes:
                proc.join()
            processes = []

    for proc in processes:
        proc.join()

    # read the computed spectrum and quantitie
    if diag:
        G_iw_proj = np.zeros((nw, dim1, nao), dtype=complex)
    else:
        G_iw_proj = np.zeros((nw, dim1, nao * nao), dtype=complex)
    pp = 0
    for d1 in range(dim1):
        out_file = base_dir + '/{}/{}'.format(str(d1), ofile)
        Gw_here = np.loadtxt(out_file, dtype=complex)
        G_iw_proj[:, d1, :] = Gw_here * 1.0

    # reshape and return
    G_iw_proj = G_iw_proj.reshape((nw, ) + orig_shape[1:])

    return G_iw_proj
