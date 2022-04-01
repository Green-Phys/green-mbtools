import h5py
import numpy as np
from MB_analysis.src import ir

f = h5py.File("../data/H2_GW/sim.h5", 'r')
it = f["iter"][()]
G_tau = f["iter"+str(it)+"/G_tau/data"][()].view(complex)
tau_mesh = f["iter"+str(it)+"/G_tau/mesh"][()]
f.close()

# Lambda is determined by the tau_mesh you use. Here the H2 GW simulation uses lambda = 1e4
lamb = '1e4'
beta = tau_mesh[-1]
nts = tau_mesh.shape[0]
my_ir = ir.IR_factory(beta, lamb)

# Fourier transform from G(tau) to  G(iw_n)
G_iw = my_ir.tau_to_w(G_tau)
# Fourier transform from G(iw_n) to G(tau)
G_tau_2 = my_ir.w_to_tau(G_iw)

diff = G_tau - G_tau_2
print(np.max(np.abs(diff.real)))
print(np.max(np.abs(diff.imag)))
