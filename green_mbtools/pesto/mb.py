from functools import reduce
import numpy as np
import scipy.linalg as LA

from . import spectral as spec
from . import orth as orth
from .ir import IR_factory
from . import dyson as dyson
from . import winter as winter
from . import analyt_cont as AC


class MB_post(object):
    """Many-body analysis class"""
    def __init__(
        self, fock, sigma=None, mu=None, gtau=None, S=None, kmesh=None,
        beta=None, ir_file=None
    ):
        """ Initialization """
        # Public instance variables
        self.sigma = None
        self.fock = None
        self.S = None
        self.kmesh = None

        # Private instance variables
        self._gtau = None
        self._S_inv_12 = None
        self._nts = None
        self._ns = None
        self._ink = None
        self._nao = None
        self._ir_list = None
        self._weight = None
        self.ir = None
        self._ir_file = None
        self._beta = None
        self._mu = None

        """Setup"""
        if fock.ndim == 4:
            # For UHF data
            self._ns = fock.shape[0]
        elif fock.ndim == 3:
            # For RHF data, reshape matrices are originally in the shape:
            #   self_energy / gtau  :   (ntau, nk, nao, nao)
            #   fock                :   (nk, nao, nao)
            # To make RHF/UHF analysis easier, we transform the dimensions to:
            #   self_energy / gtau  :   (ntau, ns=1, nk, nao, nao)
            #   fock                :   (ns=1, nk, nao, nao)
            self._ns = 1
            fock = fock.reshape((1,) + fock.shape)
            if gtau is not None:
                gtau = gtau.reshape((gtau.shape[0], 1) + gtau.shape[1:])
            if sigma is not None:
                sigma = sigma.reshape((sigma.shape[0], 1) + sigma.shape[1:])
            if S is not None:
                S = S.reshape((1,) + S.shape)
        else:
            raise ValueError(
                'Incorrect dimensions of self-energy or Fock. Acceptable \
                shapes are (nts, ns, nk, nao, nao) or (nts, nk, nao, nao) \
                for self energy and (ns, nk, nao, nao) or (nk, nao, nao) \
                for Fock matrix.'
            )

        if mu is None:
            print("Warning: Default chemical potential, mu = 0.0, is used.")
            self._mu = 0.0
        else:
            self._mu = mu

        if beta is None:
            print("Warning: Inverse temperature is set to the default value\
                1000 a.u.^{-1}.")
            self._beta = 1000
        else:
            self._beta = beta

        if ir_file is None:
            raise ValueError(
                "{} is not an acceptable IR-grid file.".format(ir_file)
                + " Provide a valid hdf5 IR-grid file."
            )
        else:
            self.ir_file = ir_file

        self._ink = fock.shape[1]
        self._nao = fock.shape[2]
        self._ir_list = np.arange(self._ink)
        self._weight = np.array([1 for i in range(self._ink)])

        self.fock = fock.copy()
        if sigma is not None:
            self.sigma = sigma.copy()
        if S is not None:
            self.S = S.copy()
        if kmesh is not None:
            self.kmesh = kmesh.copy()
        if gtau is not None:
            self.gtau = gtau.copy()

        print(self)

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        """
        Changing beta will automatically update self.ir for consistency
        """
        print("Updated beta = {}".format(value))
        self._beta = value
        if self.ir is None:
            self.ir = IR_factory(self.beta, self.ir_file)
        else:
            self.ir.update(self.beta, self.ir_file)

    @property
    def ir_file(self):
        return self._ir_file

    @ir_file.setter
    def ir_file(self, value):
        """Changing ir_file will automatically update both self._nts
        and self.ir for consistency.
        :param value: Path to the IR grid HDF5 file.
        :return:
        """
        print("Setting up IR grid for {}".format(value))
        self._ir_file = value
        if self.ir is None:
            self.ir = IR_factory(self.beta, self.ir_file)
        else:
            self.ir.update(self.beta, self.ir_file)
        self._nts = self.ir.nts

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, value):
        """Updating chemical potential implicitly implies updating the
        Green's function and density matrix.
        :param value:
        :return:
        """
        print("Updated mu = {}".format(value))
        self._mu = value
        if self._gtau is not None:  # or self.dm is not None:
            self.solve_dyson()

    @property
    def gtau(self):
        if self._gtau is None:
            self.solve_dyson()
        return self._gtau

    @gtau.setter
    def gtau(self, G):
        """
        Updating gtau will implicitly update density matrix (dm).
        :param G:
        :return:
        """
        self._gtau = G

    @property
    def dm(self):
        return -1.0 * self.gtau[-1]

    def __str__(self):
        return "######### MBPT analysis class #########\n" \
            "nts  = {} \n" \
            "ns   = {}\n" \
            "nk   = {}\n" \
            "nao  = {}\n" \
            "mu   = {}\n" \
            "beta = {}\n" \
            "#######################################".format(
                self._nts, self._ns, self._ink, self._nao, self.mu, self.beta
            )

    def solve_dyson(self):
        """
        Compute Green's function through Dyson's equation and update self.gtau
        and self.dm.
        :return:
        """
        self.gtau = dyson.solve_dyson(
            self.fock, self.S, self.sigma, self.mu, self.ir
        )

    def eigh(self, F, S, thr=1e-7):
        return LA.eigh(F, S)
    # FIXME c here is differ with the c from eigh() by a phase factor.
    # Fix it or leave it like this?

    def eigh_canonical(self, F, S, thr=1e-7):
        # S: m*m, x =: m*n, xFx: n*n, c: n*n, e: n, xc: m*n
        x = orth.canonical_matrices(S, thr, 'f')
        xFx = reduce(np.dot, (x.T.conj(), F, x))
        e, c = LA.eigh(xFx)
        c = np.dot(x, c)

        return e, c

    def get_mo(self, canonical=False, thr=1e-7):
        """
        Compute molecular orbital energy by solving FC=SCE
        :return:
        """
        if not canonical:
            eigh = self.eigh
        else:
            eigh = self.eigh_canonical
        mo_energy, mo_coeff = spec.compute_mo(self.fock, self.S, eigh, thr)
        return mo_energy, mo_coeff

    def get_no(self):
        """
        Compute natural orbitals by diagonalizing density matrix
        :return:
        """
        occ, no_coeff = spec.compute_no(self.dm, self.S)
        return occ, no_coeff

    def mulliken_analysis(self, orbitals=None):
        if orbitals is None:
            orbitals = np.arange(self._nao)
        occupations = np.zeros((self._ns, orbitals.shape[0]), dtype=complex)
        if self.S is not None:
            occupations = np.einsum(
                'k,skij,skji->si', self._weight, self.dm, self.S
            )
        else:
            occupations = np.einsum('k,skii->si', self._weight, self.dm)
        num_k = len(self._weight)
        occupations /= num_k

        # Check imaginary part
        imag = np.max(np.abs(occupations.imag))
        print("The maximum of imaginary part is ", imag)

        return occupations.real

    def wannier_interpolation(self, kpts_inter, hermi=False, debug=False):
        """
        Wannier interpolation
        :param kpts_int: Scaled k-points for the target k grid
        :return:
        """
        if self.kmesh is None:
            raise ValueError(
                "kmesh of input data is unknown. Please provide it."
            )
        Gtk_int, Sigma_tk_int, tau_mesh, Fk_int, Sk_int = winter.interpolate_G(
            self.fock, self.sigma, self.mu, self.S,
            self.kmesh, kpts_inter, self.ir, hermi=hermi, debug=debug
        )
        return Gtk_int, Sigma_tk_int, tau_mesh, Fk_int, Sk_int

    def AC_maxent(
        self, error=5e-3, maxent_exe='maxent', params='green.param',
        outdir='Maxent', gtau_orth=None
    ):
        """
        Analytical continuation using Maxent
        :param error:
        :param maxent_exe:
        :param params:
        :param outdir:
        :param gtau:
        :return:
        """
        if gtau_orth is None:
            gtau_orth = orth.sao_orth(
                self.gtau, self.S, type='g'
            ) if self.S is not None else self.gtau
            gtau_inp = np.einsum("...ii->...i", gtau_orth)
        else:
            gtau_inp = gtau_orth
        tau_mesh = self.ir.tau_mesh

        AC.maxent_run(gtau_inp, tau_mesh, error, params, maxent_exe, outdir)

    def AC_nevanlinna(
        self, gtau_orth=None, n_real=10001, w_min=-10., w_max=10.,
        eta=0.01, outdir="Nevanlinna"
    ):
        """
        Analytical continuation using Nevanlinna interpolation
        :param gtau_orth: imaginary time Green's function in the orthogonal basis, will be obtained from curren self.gtau if None
        :param n_real: number of real frequency points
        :param w_min: smallest value on real frequency grid
        :param w_max: largest value on real frequency grid
        :param eta: broadening parameter
        :param outdir: [DEPRECATED]
        :return: real frequency grid along with spectral function for a given Green's function
        """
        if gtau_orth is None:
            gtau_orth = orth.sao_orth(
                self.gtau, self.S, type='g'
            ) if self.S is not None else self.gtau
            gtau_orth = np.einsum("...ii->...i", gtau_orth)
        nw = self.ir.wsample.shape[0]
        Gw_inp = self.ir.tau_to_w(gtau_orth)[nw//2:]

        wsample = self.ir.wsample[nw//2:]
        freqs, A_w = AC.nevan_run(
            Gw_inp, wsample, n_real=n_real,
            w_min=w_min, w_max=w_max, eta=eta, spectral=True
        )

        return freqs, A_w


def minus_k_to_k_TRsym(X):
    nso = X.shape[-1]
    nao = nso // 2
    Y = np.zeros(X.shape, dtype=X.dtype)
    Y[:nao, :nao] = X[nao:, nao:].conj()
    Y[nao:, nao:] = X[:nao, :nao].conj()
    Y[:nao, nao:] = -1.0 * X[nao:, :nao].conj()
    Y[nao:, :nao] = Y[:nao, nao:].conj().transpose()
    return Y


def to_full_bz_TRsym(X, conj_list, ir_list, bz_index, k_ind):
    index_list = np.zeros(bz_index.shape, dtype=int)
    for i, irn in enumerate(ir_list):
        index_list[irn] = i
    old_shape = X.shape
    new_shape = np.copy(old_shape)
    new_shape[k_ind] = conj_list.shape[0]
    Y = np.zeros(new_shape, dtype=X.dtype)
    for ik, kk in enumerate(bz_index):
        k = index_list[kk]
        Y = Y.reshape((-1,) + Y.shape[k_ind:])
        X = X.reshape((-1,) + X.shape[k_ind:])
        for i in range(Y.shape[0]):
            Y[i, ik] = minus_k_to_k_TRsym(
                X[i, k]
            ) if conj_list[ik] else X[i, k]
        Y = Y.reshape(new_shape)
        X = X.reshape(old_shape)

    return Y


def to_full_bz(X, conj_list, ir_list, bz_index, k_ind):
    """Transform input quantity from irreducible number of k-points
    to the entire Brillouin Zone.
    """
    index_list = np.zeros(bz_index.shape, dtype=int)
    for i, irn in enumerate(ir_list):
        index_list[irn] = i
    old_shape = X.shape
    new_shape = np.copy(old_shape)
    new_shape[k_ind] = conj_list.shape[0]
    Y = np.zeros(new_shape, dtype=X.dtype)
    for ik, kk in enumerate(bz_index):
        k = index_list[kk]
        if k_ind == 0:
            Y[ik, ::] = X[k, ::].conj() if conj_list[ik] else X[k, ::]
        elif k_ind == 1:
            Y[:, ik, ::] = X[:, k, ::].conj() \
                if conj_list[ik] else X[:, k, ::]
        elif k_ind == 2:
            Y[:, :, ik, ::] = X[:, :, k, ::].conj() \
                if conj_list[ik] else X[:, :, k, ::]
    return Y


def initialize_MB_post(sim_path=None, input_path=None, ir_file=None):
    import h5py
    f = h5py.File(sim_path, 'r')
    it = f["iter"][()]
    Sr = f["S-k"][()].view(complex)
    Sr = Sr.reshape(Sr.shape[:-1])
    Fr = f["iter"+str(it)+"/Fock-k"][()].view(complex)
    Fr = Fr.reshape(Fr.shape[:-1])
    Sigmar = f["iter"+str(it)+"/Selfenergy/data"][()].view(complex)
    Sigmar = Sigmar.reshape(Sigmar.shape[:-1])
    Gr = f["iter"+str(it)+"/G_tau/data"][()].view(complex)
    Gr = Gr.reshape(Gr.shape[:-1])
    tau_mesh = f["iter"+str(it)+"/G_tau/mesh"][()]
    beta = tau_mesh[-1]
    mu = f["iter"+str(it)+"/mu"][()]
    f.close()

    f = h5py.File(input_path, 'r')
    ir_list = f["/grid/ir_list"][()]
    index = f["/grid/index"][()]
    conj_list = f["grid/conj_list"][()]
    f.close()

    """ All k-dependent matrices should lie on a full Monkhorst-Pack grid. """
    F = to_full_bz(Fr, conj_list, ir_list, index, 1)
    S = to_full_bz(Sr, conj_list, ir_list, index, 1)
    Sigma = to_full_bz(Sigmar, conj_list, ir_list, index, 2)
    G = to_full_bz(Gr, conj_list, ir_list, index, 2)

    """ Results from correlated methods """

    # Standard way to initialize
    return MB_post(
        fock=F, sigma=Sigma, mu=mu, gtau=G, S=S, beta=beta, ir_file=ir_file
    )
