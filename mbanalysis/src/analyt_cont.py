import os
import subprocess
import shutil
import numpy as np
import h5py
from multiprocessing import cpu_count, Process
import mbanalysis.nevanlinna as nevan_exe
# from mbanalysis.nevanlinna import nevanlinna


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
                X_w = np.loadtxt("{}/{}".format(d1, ofile))
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


# #
# # NOTE: Hardy optimization will require multi precision arithmetic.
# #

# def hardy_optimization(
#     xfile='dos_w.h5', eta=0.01, Nh=25, green=False, lam=1e-3
# ):
#     """Perform Hardy optimization on Nevanlinna AC data.
#     """

#     # Get freqs and coeff data
#     xf = h5py.File(xfile, 'r')
#     freqs = xf['freqs'][()] + 1j * eta
#     coeff = xf['coeff'][()]
#     dim = np.prod(coeff.shape[1:-1])
#     nw = len(freqs)
#     coeff2 = coeff.reshape((nw, dim, 4))

#     pq_params = np.zeros((dim, 2 * Nh), dtype=complex)

#     # TODO:
#     #   1.  Implement a function that computes the double derivative, and
#     #       returns the integral of square of d( d(xf) / dw) / dw
#     #   2.  Use this function as smoothness norm to optimize w.r.t.
#     #       Nkh number of parameters

#     if not green:
#         # Hardy optimization for Self-energy
#         for d in range(dim):
#             res = minimize(
#                 lambda pq: _func_to_hardy_optimize(pq, freqs, coeff2[:, d]),
#                 x0=pq_params
#             )
#             pq_params[d, :] = res.x
#     else:
#         # Hardy optimization for Green's or Spectral function
#         pass

#     final_data = np.zeros((nw, dim), dtype=complex)
#     for j, w in enumerate(freqs):
#         for d in range(dim):
#             pk = pq_params[d, :Nh]
#             qk = pq_params[d, Nh:]
#             theta_mp1_z = _build_theta_m_plus_1(w, pk, qk)
#             final_data[j, d] = (
#                 coeff[j, d, 0] * theta_mp1_z + coeff[j, d, 1]
#             ) / (
#                 coeff[j, d, 2] * theta_mp1_z + coeff[j, d, 3]
#             )

#     return final_data


# def _func_to_hardy_optimize(pq_coeffs, freqs, coeff):
#     """For a given set of {pk, qk}, build the analytically continued values and
#     get the smoothness norm.
#     freqs   ::  w + i eta
#     """
#     Nh = len(pq_coeffs) // 2
#     pk = pq_coeffs[:Nh]
#     qk = pq_coeffs[Nh:]

#     Nw = len(freqs)
#     theta_z = np.zeros(Nw, dtype=complex)
#     for w in range(Nw):
#         theta_mp1_z = _build_theta_m_plus_1(freqs[w], pk, qk)
#         theta_z[w] = (
#             coeff[w, 0] * theta_mp1_z + coeff[w, 1]
#         ) / (
#             coeff[w, 2] * theta_mp1_z + coeff[w, 3]
#         )

#     dw = np.real(freqs[1] - freqs[0])
#     norm = _get_smoothness_norm(dw, theta_z.real)

#     return norm


# def _build_theta_m_plus_1(z, pk, qk):
#     """Build theta_{M+1} (z) = sum_k ak fk(z) + bk fk*(z).
#     """
#     Nh = pk.shape[0]
#     for i in range(Nh):
#         theta_mp1 = pk[i] * _hardy_fk(z, i) + qk[i] * _hardy_fk(z, i).conj()

#     return theta_mp1


# def _hardy_fk(z, k):
#     fkz = 1 / (np.sqrt(np.pi) * (z + 1j))
#     fkz *= ((z - 1j) / (z + 1j))**k

#     return fkz


# def _get_smoothness_norm(dx, y):
#     """Compute the smoothness norm for analytically continued GW data.
#     """

#     # Finite difference derivative
#     dy = np.diff(y) / dx
#     d2y = np.diff(dy) / dx

#     # Mod square
#     d2y2 = np.abs(d2y)**2

#     # Integral (TODO: use trapezoidal)
#     res = np.sum(d2y2) * dx

#     return res
