import os
import subprocess
import shutil
import numpy as np
import h5py
from multiprocessing import cpu_count, Process
from .pes_utils import run_es
import mbanalysis.nevanlinna as nevan_exe
import mbanalysis.caratheodory as carath_exe


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
    """Run dim0 times maxent continuation for gtau
    :param gtau: (nts, dim1), dim1 = (ns, nk, nao), (ns, nk), (ns) ... etc
    :param tau_mesh:
    :param exe_path: Maxent executable path
    :param outdir: output directory w.r.t. the current working directory
    :return:
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
            for p in processes:
                p.communicate()
            processes = []
        os.chdir("..")

    for p in processes:
        p.communicate()

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
    X_iw, wsample, outdir='Nevanlinna', ifile='X_iw.txt', ofile='X_w.txt',
    coeff_file='coeff.txt', n_real=10000, w_min=-10, w_max=10, eta=0.01,
    green=True, prec=128
):
    """Function to perform Nevanlinna analytic continuation for any quantity.
    TODO: Provide a description about the input
    """

    wkdir = os.path.abspath(os.getcwd())
    print("Nevanlinna output:", os.path.abspath(wkdir + '/' + outdir))
    print("Will dump input to {} and output to {}".format(ifile, ofile))

    nw = wsample.shape[0]
    assert nw == X_iw.shape[0], "Number of imaginary frequency points \
        mismatches between \"input_parser\" and Gw."
    X_iw_shape = X_iw.shape
    X_iw = X_iw.reshape(nw, -1)
    dim1 = X_iw.shape[1]

    # For self-energy, we want both the real and imag data,
    # not just the spectral function
    spectral = green

    if not os.path.exists(outdir):
        os.mkdir(outdir)
    os.chdir(outdir)

    # Save dim1 for better understanding of output
    np.savetxt("dimensions.txt", np.asarray(X_iw_shape[1:], dtype=int))

    for d1 in range(dim1):
        if not os.path.exists(str(d1)):
            os.mkdir(str(d1))
        np.savetxt(
            "{}/{}".format(d1, ifile),
            np.column_stack(
                (wsample, X_iw[:, d1].real, X_iw[:, d1].imag)
            )
        )

    # Start analytical continuation
    processes = []
    pp = 0
    for d1 in range(dim1):
        os.chdir(str(d1))
        # arg_ls = (ifile, nw, ofile, coeff_file, spectral,
        #   prec, n_real, w_min, w_max, eta)
        p = Process(
            target=nevan_exe.nevanlinna,
            args=(
                ifile, nw, ofile, coeff_file, prec, spectral,
                n_real, w_min, w_max, eta
            )
        )
        p.start()
        processes.append(p)
        pp += 1
        if pp % _ncpu == 0:
            for p in processes:
                p.join()
            processes = []
        os.chdir("..")

    for p in processes:
        p.join()

    # Combine output
    dump_A = False
    try:
        X_w = np.loadtxt("0/{}".format(ofile))
        dtype = complex if X_w.shape[1] == 3 else float
        freqs = X_w[:, 0]
        X_wk = np.loadtxt("0/{}".format(ofile))
        dtype = complex if X_wk.shape[1] == 3 else float
        freqs = X_wk[:, 0]
    except IOError:
        pass
    else:
        dump_A = True
    if dump_A:
        X_w = np.zeros((freqs.shape[0], dim1), dtype=dtype)
        for d1 in range(dim1):
            # Read X_w data
            try:
                X_wk = np.loadtxt("{}/{}".format(d1, ofile))
                if dtype == complex:
                    X_w[:, d1].real = X_wk[:, 1]
                    X_w[:, d1].imag = X_wk[:, 2]
                else:
                    X_w[:, d1] = X_wk[:, 1]
            except IOError:
                print(
                    "{} is missing in {} folder. Possibly analytical \
                    continuation fails at that point.".format(ofile, d1)
                )
        X_w = X_w.reshape((freqs.shape[0],) + X_iw_shape[1:])
        X_iw = X_iw.reshape(X_iw_shape)
    else:
        print("All AC fails. Will not dump to DOS.h5")
        X_iw = X_iw.reshape(X_iw_shape)

    os.chdir(wkdir)

    return freqs, X_w


def caratheodory_run(
    X_iw, wsample, outdir='Caratheodory', ifile='X_iw.txt',
    matrix_ofile='X_c.txt', spectral_ofile='X_A.txt', custom_freqs=None,
    n_real=2001, w_min=-10, w_max=10, eta=0.01,
):
    """Function to perform Caratheodory analytic continuation for any quantity.
    Input parameters:
        X_iw            - contains matrix valued data on imaginary freq. points
                        shape of X_iw should be: (nw, ns, nk, nao, nao)
        wsample         - value of imaginary frequencies
        outdir          - output directory in which data will be stored
        ifile           - intermediate input file name
        matrix_ofile    - output file name for matrix valued data on real w
        spectral_ofile  - output file name for spectral function data on real w
        custom_freqs    - custom freq points on which to perform carath. AC
        n_real          - number of real frequency points
                        (used if custom_freqs = None)
        w_min           - minimum value of real frequency range
                        (used if custom_freqs = None)
        w_max           - maximum value of real frequency range
                        (used if custom_freqs = None)
        eta             - broadening parameter
    """

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
            for p in processes:
                p.join()
            processes = []
        os.chdir("..")

    for p in processes:
        p.join()

    # Combine output
    freqs, Xc_w, XA_w = load_caratheodory_data(
        matrix_ofile, spectral_ofile, X_iw_shape
    )

    # Get out of working directory
    os.chdir(wkdir)

    return freqs, Xc_w, XA_w


def dump_input_caratheodory_data(wsample, X_iw, ifile):
    """Function to dump input data to files for caratheodory
    analytic continuation.

    Caratheodory requires input data file to be in the following format:
        iw_1 iw_2 iw_3 ... iw_n

        Real X(iw_1) Matrix

        Imag X(iw_1) Matrix

        Real X(iw_2) Matrix

        Imag X(iw_2) Matrix

        ... and so on.
    """

    nw, ns, nk, nao = X_iw.shape[:4]

    # Save dim1 for better understanding of output
    np.savetxt("dimensions.txt", np.asarray(X_iw.shape[1:3], dtype=int))

    # Dump input data to file
    dim = 0
    for js in range(ns):
        for jk in range(nk):
            # make the directory for given (js, jk) point
            if not os.path.exists(str(dim)):
                os.mkdir(str(dim))
            # Write the input data to file
            output_fname = str(dim) + '/' + ifile
            with open(output_fname, 'w') as fs:
                # write frequencies in the first line
                for w_j in wsample:
                    fs.write(str(w_j) + '\t')
                fs.write('\n\n')
                for jw in range(nw):
                    # Real part for jw-th frequency
                    for p in range(nao):
                        for q in range(nao):
                            fs.write(str(X_iw[jw, js, jk, p, q].real) + '\t')
                        fs.write('\n')
                    fs.write('\n')
                    # Imag part for jw-th frequency
                    for p in range(nao):
                        for q in range(nao):
                            fs.write(str(X_iw[jw, js, jk, p, q].imag) + '\t')
                        fs.write('\n')
                    fs.write('\n')
            dim += 1

    return None


def load_caratheodory_data(matrix_file, spectral_file, X_dims):
    """Load output data from caratheodory analytic continuation.
    The spectral function output file has the format:
        w1 XA(w1 + i eta)
        w2 XA(w2 + i eta)
        ... and so on.
    And the complex matrix output file has the format:
        w1 Re.Xc[11](w1 + i eta) Im.Xc[11](w1 + i eta) Re.Xc[12](w1 + i eta)
        w2 Re.Xc[11](w2 + i eta) Im.Xc[11](w2 + i eta) Re.Xc[12](w2 + i eta)
        ... and so on.
    """

    # Dimensions
    _, ns, nk, nao = X_dims[:4]
    dim1 = ns * nk

    # Load the spectral function data
    dump_A = False
    try:
        XA_w = np.loadtxt("0/{}".format(spectral_file))
        freqs = XA_w[:, 0]
    except IOError:
        pass
    else:
        dump_A = True

    if dump_A:
        XA_w = np.zeros((freqs.shape[0], dim1))
        for d1 in range(dim1):
            # Read X_w data
            try:
                X_wsk = np.loadtxt("{}/{}".format(d1, spectral_file))
                XA_w[:, d1] = X_wsk[:, 1]
            except IOError:
                print(
                    "{} is missing in {} folder. Analytical continuation \
                    may have failed at that point.".format(spectral_file, d1)
                )
        # reshape the spectral data
        XA_w = XA_w.reshape((freqs.shape[0],) + (ns, nk))
    else:
        print("All AC fails. Will not dump to DOS.h5")

    # Load the complex matrix data
    dump_c = False
    try:
        Xc_w = np.loadtxt("0/{}".format(matrix_file))
    except IOError:
        pass
    else:
        dump_c = True

    if dump_c:
        Xc_w = np.zeros((freqs.shape[0], dim1, nao, nao), dtype=complex)
        for d1 in range(dim1):
            # Read X_c data
            try:
                X_wsk = np.loadtxt("{}/{}".format(d1, matrix_file))
                for jw in range(len(freqs)):
                    # need to convert the real + imag data into complex type
                    Xc_imtd = X_wsk[jw, 1:].view(complex)
                    Xc_w[jw, d1, :, :] = Xc_imtd.reshape((nao, nao))
            except IOError:
                print(
                    "{} is missing in {} folder. Analytical continuation \
                    may have failed at that point.".format(spectral_file, d1)
                )
        # reshape the complex matrix data
        Xc_w = Xc_w.reshape((freqs.shape[0],) + (ns, nk, nao, nao))

    return freqs, Xc_w, XA_w


def es_nevan_run(
    G_iw, wsample, n_real=10000, w_min=-10, w_max=10, eta=0.01, diag=True,
    eps_pol=1.0, parallel='ska', outdir='PESNevan', ofile='Aw.txt'
):
    """Perform ES Nevanlinna analytic continuation for G(iw) or Sigma(iw)
    TODO: Provide a description about the input
    """
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
    elif len(orig_shape) == 2:
        # assuming this is G(iw, a)
        nw, nao = orig_shape
        G_iw = G_iw.reshape((nw, 1, 1, nao))
    elif len(orig_shape) == 5:
        # this is already the form we want
        pass
    else:
        raise ValueError(
            'Incorrect shape G_iw. Expecting either 1-, 2- or 5-d array'
        )

    # Diagonal elements or not
    if diag and len(G_iw.shape) == 5:
        G_iw = np.einsum('wskaa -> wska', G_iw)
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
        # arg_ls = (ifile, nw, ofile, coeff_file, spectral,
        #   prec, n_real, w_min, w_max, eta)
        p = Process(
            target=run_es,
            args=(
                wsample, G_iw[:, d1, :], w_vals
            ),
            kwargs={
                'diag': diag,
                'eta': eta,
                'eps_pol': eps_pol,
                'ofile': out_file
            }
        )
        p.start()
        processes.append(p)
        pp += 1
        if pp % _ncpu == 0:
            for p in processes:
                p.join()
            processes = []

    for p in processes:
        p.join()

    # read the computed spectrum and quantitie
    G_w = np.zeros((n_real, ) + G_iw.shape[1:], dtype=complex)
    pp = 0
    for d1 in range(dim1):
        out_file = base_dir + '/{}/{}'.format(str(d1), ofile)
        Gw_here = np.loadtxt(out_file, dtype=complex)
        G_w[:, d1, :] = Gw_here.reshape((n_real, ) + G_iw.shape[2:])

    # reshape and return
    G_w = G_w.reshape((n_real, ) + orig_shape[1:])

    return w_vals, G_w
