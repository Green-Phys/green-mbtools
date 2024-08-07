from functools import reduce
import numpy as np
import irbasis
import h5py
import os


class IR_factory(object):
    """
    Fourier transform between imaginary time and Matsubara frequency using
    the intermediate representation (IR) grids.
    """

    def __init__(self, beta, ir_file=None, legacy_ir=False):
        """Initialize IR basis class for post processing of Green data.

        Parameters
        ----------
        beta : float
            inverse temperature of the calculation
        ir_file : string, required
            IR-grid file, by default None
        legacy : bool, optional
            toggle legacy format for IR grid file, by default False

        Raises
        ------
        ValueError
            when a valid IR-grid file is not provided
        """

        if ir_file is None:
            raise ValueError(
                "{} is not an acceptable IR-grid file.".format(ir_file)
                + " Provide a valid hdf5 IR-grid file."
            )

        self.beta = beta
        self.ir_file = ir_file
        if legacy_ir:
            self.read_IR_matrices = legacy_read_IR_matrices
        else:
            self.read_IR_matrices = new_read_IR_matrices
        self.tau_mesh, self.wsample, self.Ttc, self.Tcn, \
            self.Tnc, self.Tct = self.read_IR_matrices(
                os.path.abspath(ir_file), self.beta
            )
        self.nts = self.tau_mesh.shape[0]
        self.nw = self.wsample.shape[0]

    def update(self, beta=None, ir_file=None):
        """Update IR grid information in run-time.
        """
        if ir_file is not None:
            self.ir_file = ir_file
        if beta is not None:
            self.beta = beta
        self.tau_mesh, self.wsample, self.Ttc, self.Tcn, \
            self.Tnc, self.Tct = self.read_IR_matrices(self.ir_file, self.beta)
        self.nts = self.tau_mesh.shape[0]
        self.nw = self.wsample.shape[0]

    def tau_to_w(self, X_t):
        """Transform `X_t` from tau to imaginary frequency representation.
        """
        X_w = np.zeros((self.nw,) + X_t.shape[1:], dtype=complex)
        original_shape = X_w.shape

        X_w, X_t = X_w.reshape(self.nw, -1), X_t.reshape(self.nts, -1)
        X_w = reduce(np.dot, (self.Tnc, self.Tct, X_t[1:-1]))
        X_w = X_w.reshape(original_shape)
        return X_w

    def w_to_tau(self, X_w, debug=False):
        """Transform `X_w` from imaginary frequency to tau representation.
        """
        X_t = np.zeros((self.nts,) + X_w.shape[1:], dtype=complex)
        original_shape = X_t.shape

        X_w, X_t = X_w.reshape(self.nw, -1), X_t.reshape(self.nts, -1)
        X_t = reduce(np.dot, (self.Ttc, self.Tcn, X_w))
        X_t = X_t.reshape(original_shape)
        if debug:
            # Check the imaginary parts
            print(
                "The largest imaginary parts in X_t is {}. Please double \
                check whether this is consistent to your expectation!".format(
                    np.max(np.abs(X_t.imag))
                )
            )
        return X_t

    def tauf_to_wb(self, X_t):
        """Transform quantity from fermionic tau-grid to bosonic
        frequency grid.
        E.g., P0(tau) -> P0(i Omega).
        """

        _, wsample_bose, Ttc_b, _, Tnc_b, Tct_b = self.read_IR_matrices(
                os.path.abspath(self.ir_file), self.beta, ptype='bose'
            )
        fir = h5py.File(self.ir_file, 'r')
        Ttc_other_tmp = fir['fermi/other_uxl'][()]
        nx_b = fir['bose/nx'][()]
        fir.close()
        Ttc_other = Ttc_b @ Tct_b @ Ttc_other_tmp
        Ttc_other *= np.sqrt(2.0 / self.beta)
        Tnt_bf = Tnc_b @ Tct_b @ Ttc_other[1:1 + nx_b, :] @ self.Tct

        nw = wsample_bose.shape[0]
        X_wb = np.zeros((nw,)+X_t.shape[1:], dtype=complex)
        original_shape = X_wb.shape

        X_wb, X_t = X_wb.reshape(nw, -1), X_t.reshape(self.nts, -1)
        X_wb = reduce(np.dot, (Tnt_bf, X_t[1:-1]))
        X_wb = X_wb.reshape(original_shape)

        return X_wb

    def wb_to_tauf(self, X_wb):
        """Transform quantity from bosonic frequency grid to fermionic
        tau-grid, e.g., P0(i Omega) -> P0(tau)
        """
        _, wsample_bose, _, Tcn_b, _, _ = self.read_IR_matrices(
                os.path.abspath(self.ir_file), self.beta, ptype='bose'
            )
        nw_b = len(wsample_bose)
        fir = h5py.File(self.ir_file, 'r')
        Ttc_b_other_tmp = fir['bose/other_uxl'][()]
        fir.close()
        Ttc_b_other = self.Ttc @ self.Tct @ Ttc_b_other_tmp
        Ttc_b_other *= np.sqrt(2.0 / self.beta)
        Ttn_fb = Ttc_b_other @ Tcn_b

        X_t = np.zeros((self.nts,) + X_wb.shape[1:], dtype=complex)
        original_shape = X_t.shape

        X_w, X_t = X_wb.reshape(nw_b, -1), X_t.reshape(self.nts, -1)
        X_t = reduce(np.dot, (Ttn_fb, X_w))
        X_t = X_t.reshape(original_shape)

        return X_t

    # TODO Specify the version of irbasis.
    def tau_to_w_other(self, X_t, wsample):
        """Use IR basis python package to intrinsically transform other type of
        quantities from tau to imaginary frequency basis.
        XXX: What are these "other" quantities?
        """
        nw = wsample.shape[0]
        X_w = np.zeros((nw,)+X_t.shape[1:], dtype=complex)
        original_shape = X_w.shape

        ir_factory = irbasis.load("F", float(self.ir_file))
        tnc = ir_factory.compute_unl(wsample)
        tnc *= np.sqrt(self.beta)

        X_w, X_t = X_w.reshape(nw, -1), X_t.reshape(self.nts, -1)
        X_w = reduce(np.dot, (tnc, self.Tct, X_t[1:-1]))
        X_w = X_w.reshape(original_shape)
        return X_w


def new_read_IR_matrices(ir_path, beta, ptype='fermi'):
    ir = h5py.File(ir_path, 'r')
    wsample = ir[ptype + "/ngrid"][()]
    xsample = ir[ptype + "/xgrid"][()]

    Ttc_minus1 = ir[ptype + "/u1l_neg"][()]
    Ttc_tmp = ir[ptype + "/uxl"][()]
    Ttc_1 = ir[ptype + "/u1l_pos"][()]
    Ttc = np.zeros((Ttc_tmp.shape[0]+2, Ttc_tmp.shape[1]))
    Ttc[0], Ttc[1:-1], Ttc[-1] = Ttc_minus1, Ttc_tmp, Ttc_1
    Tnc = ir[ptype + "/uwl"][()]
    ir.close()

    if ptype == 'fermi':
        zeta = 1
    else:
        zeta = 0
    wsample = (2*wsample + zeta) * np.pi / beta
    tau_mesh = np.zeros(xsample.shape[0]+2)
    tau_mesh[0], tau_mesh[1:-1], tau_mesh[-1] = 0, (xsample+1)*beta/2.0, beta

    Ttc *= np.sqrt(2.0/beta)
    Tnc *= np.sqrt(beta)
    Tct = np.linalg.inv(Ttc[1:-1])
    Tcn = np.linalg.inv(Tnc)

    return tau_mesh, wsample, Ttc, Tcn, Tnc, Tct


def legacy_read_IR_matrices(ir_path, beta, ptype='fermi'):
    ir = h5py.File(ir_path, 'r')
    wsample = ir[ptype + "/wsample"][()]
    xsample = ir[ptype + "/xsample"][()]

    Ttc_minus1 = ir[ptype + "/ux1l_minus"][()]
    Ttc_tmp = ir[ptype + "/uxl"][()]
    Ttc_1 = ir[ptype + "/ux1l"][()]
    Ttc = np.zeros((Ttc_tmp.shape[0]+2, Ttc_tmp.shape[1]))
    Ttc[0], Ttc[1:-1], Ttc[-1] = Ttc_minus1, Ttc_tmp, Ttc_1
    Tnc_re = ir[ptype + "/uwl_re"][()]
    Tnc_im = ir[ptype + "/uwl_im"][()]
    Tnc = Tnc_re + 1j*Tnc_im
    ir.close()

    if ptype == 'fermi':
        zeta = 1
    else:
        zeta = 0
    wsample = (2*wsample + zeta) * np.pi / beta
    tau_mesh = np.zeros(xsample.shape[0]+2)
    tau_mesh[0], tau_mesh[1:-1], tau_mesh[-1] = 0, (xsample+1)*beta/2.0, beta

    Ttc *= np.sqrt(2.0/beta)
    Tnc *= np.sqrt(beta)
    Tct = np.linalg.inv(Ttc[1:-1])
    Tcn = np.linalg.inv(Tnc)

    return tau_mesh, wsample, Ttc, Tcn, Tnc, Tct
