import h5py
import numpy as np
from green_mbtools.pesto import ir

#
# Example
# Perform IR transform: navigating between imaginary-time
# and Matsubara frequency axes
#

# Data files
sim_file = '../tests/test_data/H2_GW/sim.h5'
f = h5py.File(sim_file, 'r')
it = f["iter"][()]
G_tau = f["iter" + str(it) + "/G_tau/data"][()].view(complex)
tau_mesh = f["iter" + str(it) + "/G_tau/mesh"][()]
f.close()

# Here the H2 GW simulation uses lambda = 1e4
ir_file = '../tests/test_data/ir_grid/1e4.h5'
beta = tau_mesh[-1]
nts = tau_mesh.shape[0]
my_ir = ir.IR_factory(beta, ir_file)

# Fourier transform from G(tau) to  G(iw_n)
G_iw = my_ir.tau_to_w(G_tau)
# Fourier transform from G(iw_n) to G(tau)
G_tau_2 = my_ir.w_to_tau(G_iw)
# List w values
print("Matsubara frequencies: ", my_ir.wsample)

diff = G_tau - G_tau_2
print(np.max(np.abs(diff.real)))
print(np.max(np.abs(diff.imag)))
