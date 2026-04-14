import numpy as np
import h5py
from green_grids.repn import ir as green_grids_ir
import warnings


class TransformIR:
    def __init__(self, ir_file, beta, statistics='fermi'):
        self.beta = beta
        self.ir_file = ir_file
        self.statistics = statistics

        with h5py.File(ir_file, 'r') as f:
            g = f[statistics]
            self.lmbda = int(g['metadata/lambda'][()])
            self.grid_size = int(g['metadata/ncoeff'][()])

            self._uwl = g['uwl'][:]   
            self._ulx = g['ulx'][:]   
            self._uxl = g['uxl'][:]   
            self._ulw = g['ulw'][:]
            self.x_sample = g['xgrid'][:]
            self.w_sample = g['ngrid'][:]

        # Apply beta-dependent scaling
        self.Uwl = np.sqrt(self.beta) * self._uwl
        self.Ulw = np.linalg.pinv(self.Uwl)
        self.Utl = np.sqrt(2.0 / self.beta) * self._uxl
        self.Ult = np.linalg.pinv(self.Utl)
        self.Utw = np.einsum("ij,jk->ik", self.Utl, self.Ulw)
        self.Uwt = np.einsum("ij,jk->ik", self.Uwl, self.Ult)

        self.tau = (self.x_sample + 1.0) * self.beta / 2.0
        self.omega = (2 * self.w_sample + 1) * np.pi / self.beta

        self.check_grid_consistency()
        self.basis = green_grids_ir.Basis(self.lmbda, self.grid_size, self.statistics, trim=True)

    def check_grid_consistency(self):
        basis = green_grids_ir.Basis(self.lmbda, self.grid_size, self.statistics, trim=True)
        x_sample_check = basis.sampling_points_x(self.grid_size)
        w_sample_check = basis.sampling_points_matsubara(self.grid_size)
        if not np.allclose(self.x_sample, x_sample_check):
            warnings.warn("x_sample from HDF5 does not match green_grids_ir sampling points.")
        if not np.allclose(self.w_sample, w_sample_check):
            warnings.warn("w_sample from HDF5 does not match green_grids_ir sampling points.")


    def build_transform_new_tau(self, tau_new):
        assert np.all((tau_new >= 0) & (tau_new <= self.beta)), "tau_new must be within [0, beta]"
        x_sample_new = 2.0 * tau_new / self.beta - 1.0
        self.Ut_new_l = np.sqrt(2.0 / self.beta) * np.array([self.basis.uxl(l=range(self.grid_size), x=x) for x in x_sample_new])
        self.Ut_new_t = np.einsum("ij,jk->ik", self.Ut_new_l, self.Ult)
        self.Ut_new_w = np.einsum("ij,jk->ik", self.Ut_new_l, self.Ulw)
        self.tau_new = tau_new

    def build_transform_new_omega(self, omega_new):
        w_sample_new = (omega_new * self.beta / np.pi - 1) / 2.0
        assert np.all(np.isclose(w_sample_new, np.round(w_sample_new))), "omega_new must correspond to integer Matsubara frequencies"
        w_sample_new = np.round(w_sample_new).astype(int)
        self.Uw_new_l = np.sqrt(self.beta) * self.basis.compute_unl(w_sample_new)[:, :self.grid_size]
        self.Uw_new_t = np.einsum("ij,jk->ik", self.Uw_new_l, self.Ult)
        self.Uw_new_w = np.einsum("ij,jk->ik", self.Uw_new_l, self.Ulw)
        self.omega_new = omega_new

    def transform_tau_to_new_tau(self, G_tau_old, tau_new):
        self.build_transform_new_tau(tau_new)
        return np.einsum("nt,t...->n...", self.Ut_new_t, G_tau_old)
    
    def transform_tau_to_new_omega(self, G_tau_old, omega_new):
        self.build_transform_new_omega(omega_new)
        return np.einsum("nt,t...->n...", self.Uw_new_t, G_tau_old)
    
    def transform_omega_to_new_omega(self, G_omega_old, omega_new):
        self.build_transform_new_omega(omega_new)
        return np.einsum("nw,w...->n...", self.Uw_new_w, G_omega_old)
    
    def transform_omega_to_new_tau(self, G_omega_old, tau_new):
        self.build_transform_new_tau(tau_new)
        return np.einsum("nw,w...->n...", self.Ut_new_w, G_omega_old)

