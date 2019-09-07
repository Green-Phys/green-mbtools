import numpy as np
import h5py
import MB_analysis.rmb as manybody
import MB_analysis.umb as umanybody
import MB_analysis.trans as trans
from scipy.interpolate import interp1d
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


parser = argparse.ArgumentParser(description="beta extrapolation script")
parser.add_argument("--inp", type=str, nargs="*", default=[None], help="data to be extrapolated")
parser.add_argument("--out", type=str, nargs="*", default=[None], help="output extrapolated data")
parser.add_argument("--beta_out", type=float, default=None, help="beta of output data")
parser.add_argument("--mu", type=float, default=0.0, help="chemical potetntial for input data")
parser.add_argument("--ir_path", type=str, nargs="*", default=[None], help="ir file")

args = parser.parse_args()
inp = parse_basis(args.inp)
out = parse_basis(args.out)
beta_out = args.beta_out
mu = args.mu
ir_path = parse_basis(args.ir_path)


# Read input data
inp_file = h5py.File(inp)
last_iter = inp_file["/iter"][()]
S = inp_file["/S-k"][()].view(np.complex)
F = inp_file["/iter"+str(last_iter)+"/Fock-k"][()].view(np.complex)
gtau = inp_file["/iter"+str(last_iter)+"/G_tau/data"][()].view(np.complex)
sigma = inp_file["/iter"+str(last_iter)+"/Selfenergy/data"][()].view(np.complex)
tau_mesh = inp_file["/iter"+str(last_iter)+"/G_tau/mesh"][()]
inp_file.close()

nts = gtau.shape[0]
ns = gtau.shape[1]
nk = gtau.shape[2]
nao = gtau.shape[3]
beta = tau_mesh[nts-1]
S = S.reshape(ns, nk, nao, nao)
F = F.reshape(ns, nk, nao, nao)
gtau = gtau.reshape(nts, ns, nk, nao, nao)
sigma = sigma.reshape(nts, ns, nk, nao, nao)

# Read scaled tau and Matsubara frequency mesh
ir_file = h5py.File(ir_path)
iw_list = ir_file["/fermi/wsample"][()]
nw = len(iw_list)
iw_mesh = trans.iwmesh(iw_list, beta)
x_list = ir_file["fermi/xsample"][()]
ir_file.close()

mb = umanybody.umb(gtau, sigma, F, S)
mb._beta = beta
mb._iw_list = iw_list

# Compute sigma_w at input beta
sigma_w = mb.sigma_w_ir(ir_path)

S = S.reshape(nk*ns, nao, nao)
F = F.reshape(nk*ns, nao, nao)
gtau = gtau.reshape(nts, ns*nk, nao, nao)
sigma = sigma.reshape(nts, ns*nk, nao, nao)
sigma_w = sigma_w.reshape(nw, ns*nk, nao, nao)

# Extrapolate to new iw_mesh by cubic splines
iw_mesh_new = trans.iwmesh(iw_list, beta_out)
sigma_w_new = np.zeros(sigma_w.shape, dtype=np.complex)
for iks in range(nk*ns):
    for i in range(nao):
        for j in range(nao):
            f_real = interp1d(iw_mesh.imag, sigma_w[:,iks,i,j].real, kind='cubic', fill_value="extrapolate")
            f_imag = interp1d(iw_mesh.imag, sigma_w[:,iks,i,j].imag, kind='cubic', fill_value="extrapolate")
            sigma_w_new[:,iks,i,j] = f_real(iw_mesh_new.imag) + 1j*f_imag(iw_mesh_new.imag)

# Dyson's equation
g_w_new = np.zeros((nw, ns*nk, nao, nao), dtype=np.complex)
for n in range(nw):
    for iks in range(nk*ns):
        g_w_inv = (iw_mesh_new[n] + mu)*S[iks] - F[iks] - sigma_w_new[n, iks]
        g_w_new[n, iks] = np.linalg.inv(g_w_inv)
# Transform back to tau axis
gtau_new = trans.w_to_tau_ir(g_w_new, ir_path, beta_out)
sigma_new = trans.w_to_tau_ir(sigma_w_new, ir_path, beta_out)

gtau_new = gtau_new.reshape(nts, ns, nk, nao, nao)
sigma_new = sigma_new.reshape(nts, ns, nk, nao, nao)

inp_file = h5py.File(inp)
out_file = h5py.File(out, 'w')
out_file["S-k"] = inp_file["S-k"][()]
last_it = inp_file["iter"][()]
out_file["iter"] = 1
out_file["iter1/Energy/2nd"] = inp_file["iter"+str(last_it)+"/Energy/2nd"][()]
out_file["iter1/Energy/hf"] = inp_file["iter"+str(last_it)+"/Energy/hf"][()]
out_file["iter1/Fock-k"] = inp_file["iter"+str(last_it)+"/Fock-k"][()]
out_file["iter1/G_tau/data"] = gtau_new.view(np.float).reshape(gtau_new.shape[0],gtau_new.shape[1],gtau_new.shape[2],gtau_new.shape[3],gtau_new.shape[4],2)
out_file["iter1/G_tau/mesh"] = tau_mesh * beta_out/beta
out_file["iter1/Selfenergy/data"] = sigma_new.view(np.float).reshape(sigma_new.shape[0],sigma_new.shape[1],sigma_new.shape[2],sigma_new.shape[3],sigma_new.shape[4],2)
out_file["iter1/Selfenergy/mesh"] = tau_mesh * beta_out/beta
inp_file.close()
out_file.close()

