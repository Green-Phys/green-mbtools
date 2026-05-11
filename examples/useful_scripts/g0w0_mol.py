"""
G0W0 quasi-particle calculation for molecules.

Usage
-----
    python g0w0_mol.py --input input.h5 --sim sim.h5 --ir_file 1e4_104.h5 \\
                       --xc pbe0 [--dm dm.h5] [--out g0w0_out.h5]

Arguments
---------
  --input    input.h5 produced by green_mbtools mean-field initialisation
  --sim      sim.h5 produced by the GREEN GW/GF2 solver
  --ir_file  IR-grid file (e.g. 1e4_104.h5)
  --xc       DFT functional used in the mean-field (e.g. pbe0, b3lyp).
             Omit or pass "hf" for a Hartree-Fock reference (correction vanishes).
  --dm       Path to dm.h5 (written next to input.h5 by green_mbtools).
             If omitted, the density matrix is reconstructed from HF/mo_coeff.
  --iter     GW iteration to use (default: last iteration stored in sim.h5).
  --out      Output HDF5 file name (default: g0w0_out.h5).

Physics
-------
The quasiparticle equation solved here is:

    eps_QP = eps_DFT + Re[Sigma_c(eps_QP)] + Vhf_pp - Vxc_pp

where eps_DFT are the mean-field eigenvalues stored in HF/mo_energy,
Sigma_c is the correlation self-energy from Selfenergy/data (sim.h5),
and Vhf / Vxc are the HF and DFT effective potentials recomputed via PySCF
from the converged density matrix.  The self-energy is analytically continued
from imaginary frequency to the real axis using Pade-Thiele approximants.

See: Zhu and Chan, J. Chem. Theory Comput. 2021, 17, 2, 727-741.
"""

import h5py
import time
import argparse
import numpy as np
from pyscf.pbc import gto as pbc_gto
from pyscf import gto as mgto, scf as mscf, dft as mdft
from green_mbtools.pesto import mb
from scipy.optimize import newton
from pyscf.gw.gw_ac import AC_pade_thiele_diag, pade_thiele


np.set_printoptions(suppress=True, precision=5, linewidth=400)


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--debug", type=bool, default=False, help="Debug mode")
parser.add_argument("--input", type=str, default="input.h5", help="Input file used in GW calculation")
parser.add_argument("--sim", type=str, default="sim.h5", help="GW output file (sim.h5)")
parser.add_argument("--iter", type=int, default=-1, help="GW iteration to use (default: last)")
parser.add_argument("--ir_file", type=str, default='1e4_104.h5', help="IR-grid HDF5 file")
parser.add_argument("--out", type=str, default='g0w0_out.h5', help="Output file name")
parser.add_argument(
    "--xc", type=str, default=None,
    help="DFT functional of the mean-field (e.g. pbe0). Omit or use 'hf' for HF reference."
)
parser.add_argument(
    "--dm", type=str, default=None,
    help="Path to dm.h5 containing HF/dm-k. If omitted, DM is reconstructed from HF/mo_coeff."
)
args = parser.parse_args()

input_path = args.input
sim_path = args.sim
it = args.iter
ir_file = args.ir_file
output = args.out


# ---------------------------------------------------------------------------
# Read input.h5
# ---------------------------------------------------------------------------
print("Reading input file")
f = h5py.File(input_path, 'r')
kmesh_scaled = f["/symmetry/k/mesh_scaled"][()]
ibz2bz = f["/symmetry/k/ibz2bz"][()]
bz2ibz = f["/symmetry/k/bz2ibz"][()]
tr_conj = f["/symmetry/k/tr_conj"][()]
k_sym_trans = f["/symmetry/k/k_sym_transform_ao"][()]

mf_H0 = f["HF/H-k"][()].view(complex)
mf_H0 = mf_H0.reshape(mf_H0.shape[:-1])
mf_Sk = f["HF/S-k"][()].view(complex)
mf_Sk = mf_Sk.reshape(mf_Sk.shape[:-1])

params_nao = int(f["/params/nao"][()])
params_nso = int(f["/params/nso"][()])
nocc = int(f["/params/nel_cell"][()])

cell_str = f["Cell"][()]
mo_coeff_raw = f["HF/mo_coeff"][()]
mo_energy_raw = f["HF/mo_energy"][()]

nao = mf_H0.shape[-1]   # = params_nso; orbital space size
ns = mf_H0.shape[0]
nk = bz2ibz.shape[0]
ink = ibz2bz.shape[0]
is_x2c = (params_nso == 2 * params_nao)
f.close()

# Normalize mo_coeff to (ns, nk, nao, nao).
# Molecular RHF/GHF stores (nao, nao); UHF stores (ns, nao, nao).
mc = mo_coeff_raw
if mc.ndim == 2:
    mo_coeff = mc[np.newaxis, np.newaxis]
elif mc.ndim == 3 and mc.shape[0] == ns:
    mo_coeff = mc[:, np.newaxis]
elif mc.ndim == 3:
    mo_coeff = mc[np.newaxis]
else:
    mo_coeff = mc

# Normalize mo_energy to (ns, nk, nao).
me = mo_energy_raw
if me.ndim == 1:
    fock_eigs = me[np.newaxis, np.newaxis]
elif me.ndim == 2 and me.shape[0] == ns:
    fock_eigs = me[:, np.newaxis]
elif me.ndim == 2:
    fock_eigs = me[np.newaxis]
else:
    fock_eigs = me


# ---------------------------------------------------------------------------
# Read sim.h5
# ---------------------------------------------------------------------------
print("Reading sim file")
f = h5py.File(sim_path, 'r')
if it == -1:
    it = f["iter"][()]
rSigma1 = f["iter" + str(it) + "/Sigma1"][()].view(complex)
rGk = f["iter" + str(it) + "/G_tau/data"][()].view(complex)
rSigmak = f["iter" + str(it) + "/Selfenergy/data"][()].view(complex)
tau_mesh = f["iter" + str(it) + "/Selfenergy/mesh"][()]
T_inv = tau_mesh[-1]                        # beta = last point of the tau mesh
mu = f["iter" + str(it) + "/mu"][()]
nts = rSigmak.shape[0]
f.close()

print("Green shape:", rGk.shape)
print("H0 shape:  ", mf_H0.shape)
print("Overlap shape:", mf_Sk.shape)
print("Sigma shape:", rSigmak.shape)
print("beta =", T_inv, ", mu =", mu)

Sk = 1.0 * mf_Sk
Sigma_tk = 1.0 * rSigmak
G_tk = 1.0 * rGk
del mf_H0, mf_Sk, rSigma1, rSigmak, rGk


# ---------------------------------------------------------------------------
# Reconstruct pyscf Mole from the Cell stored in input.h5
# ---------------------------------------------------------------------------
cell = pbc_gto.loads(cell_str)
mol = mgto.Mole()
mol.atom = cell.atom
mol.basis = cell.basis
mol.charge = cell.charge
mol.spin = cell.spin
if cell.ecp:
    mol.ecp = cell.ecp
mol.verbose = 0
mol.build()


# ---------------------------------------------------------------------------
# Load or reconstruct density matrix (shape: ns, nk, nao, nao)
# ---------------------------------------------------------------------------
if args.dm is not None:
    print("Loading density matrix from", args.dm)
    f_dm = h5py.File(args.dm, 'r')
    dm = f_dm["HF/dm-k"][()].view(complex)
    dm = dm.reshape(dm.shape[:-1])
    f_dm.close()
else:
    print("Reconstructing density matrix from HF/mo_coeff")
    dm = np.zeros((ns, nk, nao, nao), dtype=complex)
    spin = mol.spin
    if is_x2c:
        dm[0, 0] = mo_coeff[0, 0, :, :nocc] @ mo_coeff[0, 0, :, :nocc].conj().T
    elif ns == 1:
        nocc_s = nocc // 2
        dm[0, 0] = 2.0 * (mo_coeff[0, 0, :, :nocc_s] @ mo_coeff[0, 0, :, :nocc_s].conj().T)
    else:
        nocc_a = (nocc + spin) // 2
        nocc_b = (nocc - spin) // 2
        dm[0, 0] = mo_coeff[0, 0, :, :nocc_a] @ mo_coeff[0, 0, :, :nocc_a].conj().T
        dm[1, 0] = mo_coeff[1, 0, :, :nocc_b] @ mo_coeff[1, 0, :, :nocc_b].conj().T


# ---------------------------------------------------------------------------
# Compute Vhf and Vxc via PySCF get_veff using the converged density matrix.
# Both include the Hartree term; it cancels in the QP correction Vhf - Vxc.
# When xc=None/'hf', mf_dft_obj == mf_hf_obj and the correction vanishes.
# ---------------------------------------------------------------------------
use_xc = (args.xc is not None and args.xc.lower() not in ('hf', 'none'))

if is_x2c:
    mf_hf_obj = mscf.GHF(mol).x2c1e()
    mf_dft_obj = mdft.GKS(mol).x2c1e() if use_xc else mf_hf_obj
elif ns == 1:
    mf_hf_obj = mscf.RHF(mol)
    mf_dft_obj = mdft.RKS(mol) if use_xc else mf_hf_obj
else:
    mf_hf_obj = mscf.UHF(mol)
    mf_dft_obj = mdft.UKS(mol) if use_xc else mf_hf_obj

if use_xc:
    mf_dft_obj.xc = args.xc
    mf_dft_obj.verbose = 0
mf_hf_obj.verbose = 0

print("Computing Vhf and Vxc (xc='{}')".format(args.xc))

if ns == 2 and not is_x2c:
    dm_inp = dm[:, 0].real                                              # (2, nao, nao)
    vhf_val = np.asarray(mf_hf_obj.get_veff(dm=dm_inp), dtype=complex) # (2, nao, nao)
    vxc_val = np.asarray(mf_dft_obj.get_veff(dm=dm_inp), dtype=complex)
    mf_vxc_hf = vhf_val[:, np.newaxis, :, :]                           # (ns, 1, nao, nao)
    mf_vxc_dft = vxc_val[:, np.newaxis, :, :]
else:
    dm_inp = dm[0, 0].real                                              # (nao, nao)
    vhf_val = np.asarray(mf_hf_obj.get_veff(dm=dm_inp), dtype=complex) # (nao, nao)
    vxc_val = np.asarray(mf_dft_obj.get_veff(dm=dm_inp), dtype=complex)
    mf_vxc_hf = vhf_val[np.newaxis, np.newaxis, :, :]                  # (1, 1, nao, nao)
    mf_vxc_dft = vxc_val[np.newaxis, np.newaxis, :, :]


# ---------------------------------------------------------------------------
# Transform all quantities to MO basis using DFT MO coefficients from input.h5
# ---------------------------------------------------------------------------
print("Transforming quantities to MO basis")
mo_coeff_adj = np.einsum('skpq -> skqp', mo_coeff.conj())

Sk_mo = np.einsum('skpr, skrt, sktq -> skpq', mo_coeff_adj, Sk, mo_coeff, optimize=True)
Sigma_tk_mo = np.einsum('skab, tskbc, skcd -> tskad', mo_coeff_adj, Sigma_tk, mo_coeff, optimize=True)
G_tk_mo = np.einsum('skab, tskbc, skcd -> tskad', mo_coeff_adj, G_tk, mo_coeff, optimize=True)
vhf_mo = np.einsum('skpr, skrt, sktq -> skpq', mo_coeff_adj, mf_vxc_hf, mo_coeff, optimize=True)
vxc_mo = np.einsum('skpr, skrt, sktq -> skpq', mo_coeff_adj, mf_vxc_dft, mo_coeff, optimize=True)

# DFT Fock is diagonal in the MO basis with eigenvalues from HF/mo_energy
Fk_mo = np.zeros((ns, nk, nao, nao), dtype=complex)
for s in range(ns):
    for k in range(nk):
        Fk_mo[s, k] = np.diag(fock_eigs[s, k])

print("fock_eigs (spin 0, k 0):", fock_eigs[0, 0])


# ---------------------------------------------------------------------------
# MB_post: used here only for the IR tau->iw transform of the self-energy
# ---------------------------------------------------------------------------
mbo = mb.MB_post(
    fock=Fk_mo, sigma=Sigma_tk_mo, mu=mu, gtau=None, S=Sk_mo,
    kmesh=kmesh_scaled, beta=T_inv, ir_file=ir_file
)

Sigma_tk_int = mbo.sigma
Sk_int = mbo.S


# ---------------------------------------------------------------------------
# Pade-Thiele analytic continuation of the self-energy diagonal (per spin)
# ---------------------------------------------------------------------------
Sigma_tk_diag = np.einsum('tskii -> tski', Sigma_tk_int)

iwsample = mbo.ir.wsample
nw = iwsample.shape[0]
iwsample_pos = iwsample[iwsample > 0]
iw_pos_for_pade = np.zeros((nao, len(iwsample_pos)))
for i in range(nao):
    iw_pos_for_pade[i, :] = iwsample_pos[:]

Sigma_iw_diag = mbo.ir.tau_to_w(Sigma_tk_diag)
Sigma_iw_positive = Sigma_iw_diag[iwsample > 0]

# Subsample Matsubara frequencies fed into the Pade fit
nskip = 1
if nw // 2 < 40:
    idx = np.arange(1, nw // 2, 1)
else:
    idx1 = np.arange(1, 20, nskip)
    idx2 = np.arange(idx1[-1] + nskip, nw // 2, 1)
    idx = np.concatenate((idx1, idx2))

iw_inp = iw_pos_for_pade[:, idx]
Sigma_iw_inp = Sigma_iw_positive[idx]

print("Pade interpolation for Sigma — iw_inp shape:", iw_inp.shape)
t3 = time.time()

omega_fit_list = []
pade_coeff_list = []
for s in range(ns):
    sig_s = np.einsum('wka -> aw', Sigma_iw_inp[:, s, :, :])   # (nao, nfreq)
    coeff_s, omega_s = AC_pade_thiele_diag(sig_s, 1j * iw_inp)
    omega_fit_list.append(np.asarray(omega_s))
    pade_coeff_list.append(np.asarray(coeff_s))

omega_fit = np.array(omega_fit_list)       # (ns, nao, nfreq)
pade_coeff = np.array(pade_coeff_list)     # (ns, nao, nfreq)
pade_coeff = pade_coeff[:, np.newaxis]     # (ns, 1, nao, nfreq)

print("Pade-Thiele AC completed in: {:.2f} s".format(time.time() - t3))


# ---------------------------------------------------------------------------
# Self-consistent quasiparticle equation:
#   eps_QP = eps_DFT + Re[Sigma_c(eps_QP)] + Vhf_pp - Vxc_pp
# ---------------------------------------------------------------------------
sc_qp_eigs = fock_eigs.copy().astype(complex)


def quasiparticle(omega, s, k, p):
    sigmaR = pade_thiele(omega - mu, omega_fit[s, p], pade_coeff[s, k, :, p]).real
    return omega - (fock_eigs[s, k, p].real + sigmaR + vhf_mo[s, k, p, p].real - vxc_mo[s, k, p, p].real)


au2ev = 27.21139
for sp_idx in range(ns):
    for k_idx in range(nk):
        for mo_idx in range(nao):
            try:
                e = newton(
                    lambda w: quasiparticle(w, sp_idx, k_idx, mo_idx),
                    fock_eigs[sp_idx, k_idx, mo_idx].real,
                    tol=1e-6, maxiter=100, full_output=False
                )
                print(au2ev * e)
                sc_qp_eigs[sp_idx, k_idx, mo_idx] = e
            except RuntimeError:
                print("QP not converged: sp={}, k={}, mo={}".format(sp_idx, k_idx, mo_idx))
            except ValueError:
                print("ValueError (AC range?): sp={}, k={}, mo={}".format(sp_idx, k_idx, mo_idx))

# Print IPs near the Fermi level for spin 0 (clip to active-space orbital count)
homo = min(nocc // 2, nao)
print("G0W0 IP:", au2ev * sc_qp_eigs[0, 0, max(0, homo - 4):homo + 1].real)


# ---------------------------------------------------------------------------
# Save output
# ---------------------------------------------------------------------------
f = h5py.File(output, 'w')
f["S-k"] = Sk_int
f["Fock-k"] = Fk_mo
f["Selfenergy/data"] = Sigma_tk_int
f["Selfenergy/mesh"] = tau_mesh
f["mu"] = mu
f["quasiparticle/sc"] = sc_qp_eigs
f.close()
