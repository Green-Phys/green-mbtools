import numpy as np
import MB_analysis.umb as manybody
import MB_analysis.util as util
import h5py
import os
import sys
import argparse

def parse_basis(basis_list):
  if len(basis_list) % 2 == 0:
    b = {}
    for atom_i in range(0, len(basis_list), 2):
      bas_i = basis_list[atom_i + 1]
      if os.path.exists(bas_i) :
        with open(bas_i) as bfile:
          bas = mgto.parse(bfile.read())
      # if basis specified as a standard basis
      else:
        bas = bas_i
      b[basis_list[atom_i]] = bas
    return b
  else:
    return basis_list[0]

def parse_geometry(g):
  res = ""
  if os.path.exists(g) :
    with open(g) as gf:
      res = gf.read()
  else:
    res = g
  return res

parser = argparse.ArgumentParser(description="MB post processing script")
parser.add_argument("--input", type=str, default='input.h5', help="input file")  
parser.add_argument("--sim", type=str, default='sim.h5', help="sim file")
parser.add_argument("--iter", type=int, default=None, help="iteration")
parser.add_argument("--element_ao_end", type=int, nargs="*", default=[None], help="Last ao index for each atom")
parser.add_argument("--element_Z", type=int, nargs="*", default=[None], help="Nuclear charges for each atom")

args = parser.parse_args()

sim = args.sim
iter = args.iter
element_ao = args.element_ao_end
element_Z  = args.element_Z

print("Nuclear charges of each element = ", element_Z)
print("Last AO index for each atom = ", element_ao)

# Read input data
sim_file = h5py.File(sim)
if iter is None:
    last_iter = sim_file["/iter"][()]
else: 
    last_iter = iter
S = sim_file["S-k"][()].view(np.complex)
F = sim_file["/iter"+str(last_iter)+"/Fock-k"][()].view(np.complex)
gtau = sim_file["/iter"+str(last_iter)+"/G_tau/data"][()].view(np.complex)
sigma = sim_file["/iter"+str(last_iter)+"/Selfenergy/data"][()].view(np.complex)
sim_file.close()

with h5py.File(args.input, "r") as inp_data:
  if "grid/weight" in inp_data:
    weight = inp_data["grid/weight"][()]
    conj_list = inp_data["grid/conj_list"][()]
    ir_list = inp_data["grid/ir_list"][()]
    bz_index = inp_data["grid/index"][()]
  else:
    weight = [1] * kmesh.shape[0]
    conj_list = [0] * kmesh.shape[0]
    ir_list = range(kmesh.shape[0])
    bz_index = range(kmesh.shape[0])


nts = gtau.shape[0]
ns = gtau.shape[1]
ink = gtau.shape[2]
nao = gtau.shape[3]
S = S.reshape(ns, ink, nao, nao)
F = F.reshape(ns, ink, nao, nao)
gtau = gtau.reshape(nts, ns, ink, nao, nao)
sigma = sigma.reshape(nts, ns, ink, nao, nao)

mb = manybody.umb(gtau, sigma, F, S, ir_list, weight)

# Mulliken anallysis
natom = len(element_ao)
ao_start = 0
for i in range(natom):
  aos = np.arange(ao_start, element_ao[i])
  Z = element_Z[i]
  net_charge = mb.mulliken_charge(aos, Z)
  magnetic_moment = mb.mulliken_mu(aos)
  print("element "+str(i)+":")
  print("Magnetic moment = ", magnetic_moment)
  print("Charges = ", net_charge)

  ao_start = element_ao[i]