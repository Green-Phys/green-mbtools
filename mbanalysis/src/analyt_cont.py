import os
import subprocess
import shutil
import numpy as np
import h5py
from multiprocessing import cpu_count, Process
import mbanalysis.nevanlinna as nevan_exe


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
    coefile='coeff.txt', n_real=10000, w_min=-10, w_max=10, eta=0.01,
    green=True
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

    # Template function to run nevanlinna
    def template_nevan_func(input):
        return nevan_exe.nevanlinna(
            input, nw, ofile, coefile, spectral, n_real,
            w_min, w_max, eta
        )

    # Start analytical continuation
    processes = []
    pp = 0
    for d1 in range(dim1):
        os.chdir(str(d1))
        p = Process(target=template_nevan_func, args=(ifile,))
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
        coeff_w = np.zeros((freqs.shape[0], dim1, 4), dtype=np.complex128)
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
            # Read coeff_w data
            try:
                coeff_wk = np.loadtxt("{}/{}".format(d1, coefile))
                coeff_w[:, d1, 0].real = coeff_wk[:, 1]
                coeff_w[:, d1, 0].imag = coeff_wk[:, 2]
                coeff_w[:, d1, 1].real = coeff_wk[:, 3]
                coeff_w[:, d1, 1].imag = coeff_wk[:, 4]
                coeff_w[:, d1, 2].real = coeff_wk[:, 5]
                coeff_w[:, d1, 2].imag = coeff_wk[:, 6]
                coeff_w[:, d1, 3].real = coeff_wk[:, 7]
                coeff_w[:, d1, 3].imag = coeff_wk[:, 8]
            except IOError:
                print(
                    "{} is missing in {} folder. Possibly analytical \
                    continuation fails at that point.".format(ofile, d1)
                )

        X_w = X_w.reshape((freqs.shape[0],) + X_iw_shape[1:])
        X_iw = X_iw.reshape(X_iw_shape)

        coeff_w = coeff_w.reshape((freqs.shape[0],) + X_iw.shape[1:] + (4, ))

        if green:
            fname = 'dos.h5'
            dname = 'dos'
        else:
            fname = 'sigma_w.h5'
            dname = 'sigma'
        f = h5py.File(fname, 'w')
        f["freqs"] = freqs
        f["iwsample"] = wsample
        f[dname + "_w"] = X_w
        f[dname + "_iw"] = X_iw
        f['coeff'] = coeff_w
        f.close()
    else:
        print("All AC fails. Will not dump to DOS.h5")
        X_iw = X_iw.reshape(X_iw_shape)

    os.chdir("..")

    return
