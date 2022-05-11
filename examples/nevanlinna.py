import numpy as np
import time
import h5py
from os.path import abspath

from mbanalysis import mb
from mbanalysis.src import orth

##################
#
# Input parameters
#
##################
# Inverse temperature
T_inv = 1000
debug = True

# Input files
data_path = abspath('../data')
input_path = data_path + '/H2_GW/input.h5'
sim_path = data_path + '/H2_GW/sim.h5'
lamb = '1e4'

# Compiled Nevanlinna file
nevan_exe = abspath('../Nevanlinna/nevanlinna')

##################
#
# Read input data
#
##################

# Read converged imaginary time calculation
print("Reading simulation data.")
f = h5py.File(sim_path, 'r')
Sr = f["S-k"][()].view(complex)
Sr = Sr.reshape(Sr.shape[:-1])
Fr = f["iter14/Fock-k"][()].view(complex)
Fr = Fr.reshape(Fr.shape[:-1])
Sigmar = f["iter14/Selfenergy/data"][()].view(complex)
Sigmar = Sigmar.reshape(Sigmar.shape[:-1])
Gr = f["iter14/G_tau/data"][()].view(complex)
Gr = Gr.reshape(Gr.shape[:-1])
mu = f["iter14/mu"][()]
f.close()
print("Completed reading simulation data.")

# Read data about grids
print("Reading mean-field data")
f = h5py.File(input_path, 'r')
mo_coeff = f["/HF/mo_coeff"][()]
ir_list = f["/grid/ir_list"][()]
weight = f["/grid/weight"][()]
index = f["/grid/index"][()]
conj_list = f["grid/conj_list"][()]
f.close()
print("Completed reading mean-field data.")

# All k-dependent matrices should lie on a full Monkhorst-Pack grid.
# Transform from reduced BZ to full BZ
print("Transforming data from reduced BZ to full BZ.")
Fk = mb.to_full_bz(Fr, conj_list, ir_list, index, 1)
Sk = mb.to_full_bz(Sr, conj_list, ir_list, index, 1)
Sigmak = mb.to_full_bz(Sigmar, conj_list, ir_list, index, 2)
Gk = mb.to_full_bz(Gr, conj_list, ir_list, index, 2)
print('Pre analysis complete')

##################
#
# Nevanlinna analytical continuation
#
##################

# Construct MB_post class
print("Setting up mbanalysis post processing object.")
t1 = time.time()
MB = mb.MB_post(
    fock=Fk, sigma=Sigmak, gtau=Gk, mu=mu, S=Sk, beta=T_inv, lamb=lamb
)
t2 = time.time()
print("Time required to set up post processing: ", t2 - t1)

# By default, running Nevanlinna for all diagonal elements of MB.gtau
# in SAO basis

t3 = time.time()
MB.AC_nevanlinna(
    nevan_exe=nevan_exe, outdir='Nevanlinna'
)
t4 = time.time()
print("TIme required for Nevanlinna AC: ", t4 - t3)

# Running Nevanlinna for given G(t) in whatever orthogonal basis
t5 = time.time()
Gt_canonical = orth.canonical_orth(MB.gtau, MB.S, type='g')
Gt_sao = orth.sao_orth(MB.gtau, MB.S, type='g')
Gt_orbsum = np.einsum("tskii->tsk", Gt_sao)
MB.AC_nevanlinna(
    nevan_exe=nevan_exe, outdir='Nevanlinna_orbsum', gtau_orth=Gt_orbsum
)
t6 = time.time()
print("Time required for Nevanlinna AC in orthogonal basis: ", t6 - t5)
