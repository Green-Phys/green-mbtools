import numpy as np
import os


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
