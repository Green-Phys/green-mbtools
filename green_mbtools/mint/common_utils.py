import argparse
import logging
import os

import h5py
import numpy as np
import pyscf.lib.chkfile as chk
from io import StringIO
from numba import jit
from pyscf import gto as mgto
from pyscf import scf as mscf
from pyscf import dft as mdft
from pyscf import df as mdf
from pyscf.pbc import tools, gto, df, scf, dft

import importlib.metadata as imd

from . import integral_utils as int_utils

def construct_rmesh(nkx, nky, nkz):
    '''
    Construct real space grid centered around (0,0,0)
    '''
    Lx, Ly, Lz = (nkx-1)//2, (nky-1)//2, (nkz-1)//2 # nk=6, L=2
    leftx, lefty, leftz = (nkx-1)%2, (nky-1)%2, (nkz-1)%2 # left = 1

    rx = np.linspace(-Lx, Lx+leftx, nkx, endpoint=True) # -2,-1,0,1,2,3
    ry = np.linspace(-Ly, Ly+lefty, nky, endpoint=True)
    rz = np.linspace(-Lz, Lz+leftz, nkz, endpoint=True)
    RX, RY, RZ = np.meshgrid(rx, ry, rz)
    rmesh = np.array([RX.flatten(), RY.flatten(), RZ.flatten()]).T

    return rmesh

def extract_ase_data(a, atoms):
    '''
    For a given data in XYZ format generate parameters in ASE format
    
    :param a: string containing lattice vectros in XYZ format
    :param atoms: string containin atoms positions in XYZ format
    '''
    symbols = []
    positions = []
    lattice_vectors = np.genfromtxt(StringIO(a),delimiter=',',dtype=np.float64)
    for atom in atoms.splitlines():
        if len(atom) == 0 or len(atom.split()) != 4:
            continue
        atom = atom.strip()
        symbol = atom.split()[0]
        position = atom.split(symbol)[1].strip()
        symbols.append(symbol)
        position = [float(p) for p in position.split()]
        positions.append(np.dot(np.linalg.inv(lattice_vectors), position).tolist())
    return (lattice_vectors, symbols, positions)

def print_high_symmetry_points(cell, args):
    '''
    For a given unit cell and simulation parameters generate and print list of lattice special points
    '''
    import ase.spacegroup
    lattice_vectors, symbols, positions = extract_ase_data(args.a, args.atom)
    cc = ase.spacegroup.crystal(symbols, positions, cellpar=ase.geometry.cell_to_cellpar(lattice_vectors))
    space_group = ase.spacegroup.get_spacegroup(cc)
    lat = cc.cell.get_bravais_lattice()
    special_points = lat.get_special_points()
    print("List of special points: {}".format(special_points))

def check_high_symmetry_path(cell, args):
    '''
    Check that selected high-symmetry path is correct for the chosen simulation parameters

    :param cell: unit cell object
    :param args: simulation parameters
    '''
    if args.high_symmetry_path is None:
        return
    import ase.spacegroup
    lattice_vectors, symbols, positions = extract_ase_data(args.a, args.atom)
    logging.info(f"parse: {lattice_vectors} {symbols} {positions}")
    cc = ase.spacegroup.crystal(symbols, positions, cellpar=ase.geometry.cell_to_cellpar(lattice_vectors))
    space_group = ase.spacegroup.get_spacegroup(cc)
    lat = cc.cell.get_bravais_lattice()
    special_points = lat.get_special_points()
    path = args.high_symmetry_path
    for sp in special_points.keys():
        path = path.replace(sp, "")
    path = path.replace(",", "")
    if path != "":
        raise RuntimeError(("Chosen high symmetry path {} has invalid special points {}. Valid "
                            "special points are {} ").format(args.high_symmetry_path, path, special_points.keys()))


def high_symmetry_path(cell, args):
    '''
    Compute high-symmetry k-path

    :param cell: unit-cell object
    :param args: simulation parameters
    :return: Points on the chosen high-symmetry path, corresponding \
        non-interacting Hamiltonian and overlap matrix, and linear k-point
        axis and labels (used in band structure plots)
    '''
    if args.high_symmetry_path is None:
        return [None, None, None]
    import ase
    lattice_vectors, _, _ = extract_ase_data(args.a, args.atom)
    kpath = ase.dft.kpoints.bandpath(
        args.high_symmetry_path, lattice_vectors,
        npoints=args.high_symmetry_path_points
    )
    kmesh = cell.get_abs_kpts(kpath.kpts)
    if args.x2c == 2:
        new_mf = args.mean_field(cell, kmesh).density_fit().x2c1e()
    elif args.x2c == 1:
        new_mf = args.mean_field(cell, kmesh).density_fit().sfx2c1e()
    else:
        new_mf = args.mean_field(cell, kmesh).density_fit()
    H0_hs = new_mf.get_hcore()
    Sk_hs = new_mf.get_ovlp()
    # get symmetry labels
    # lin_kpt_axis = a tuple of (
    #   linear axis for plotting kpoints,
    #   special point location along the linear axis,
    #   symmetry point labels
    # )
    lin_kpt_axis = kpath.get_linear_kpoint_axis()
    return [kmesh, H0_hs, Sk_hs, lin_kpt_axis]


def transform(Z, X, X_inv):
    '''
    Transform Z into X basis
    :param Z: Object to be transformed
    :param X: Transformation matrix
    :param X_inv: Inverse transformation matrix
    :return: Z in new basis
    '''
    Z_X = np.zeros(Z.shape, dtype=np.complex128)
    maxdiff = -1
    for ss in range(Z.shape[0]):
        for ik in range(Z.shape[1]):
            Z_X[ss,ik] = np.einsum('ij,jk...,kl->il...', X[ik], Z[ss, ik], X[ik].T.conj())

            Z_restore = np.dot(X_inv[ik], np.dot(Z_X[ss, ik], X_inv[ik].conj().T))
            diff = np.max(np.abs(Z[ss, ik] - Z_restore))
            maxdiff = max(maxdiff, diff)

            if not np.allclose(Z[ss, ik], Z_restore, atol=1e-12, rtol=1e-12) :
                error = "Orthogonal transformation failed. Max difference between origin and restored quantity is {}".format(np.max(np.abs(Z[ss,ik] - Z_restore)))
                raise RuntimeError(error)
    logging.info(f"Maximum difference between Z and Z_restore {maxdiff}")
    return Z_X

def fold_back_to_1stBZ(kpts):
    '''
    Map each k-point from a given list of scaled k-points into the first Brillouin zone

    :param kpts: list of k-points
    '''
    nkpts = len(kpts)
    for i, ik in enumerate(kpts):
        kpts[i] = np.array([wrap_1stBZ(kk) for kk in ik])
    return kpts

def inversion_sym(kmesh_scaled):
    '''
    For a given list of the scaled k-point in the full Brillouin zone select k-points that are 
    equivalent by the time-reversal symmetry and return them and their corresponding weight and index
    '''
    ind = np.arange(np.shape(kmesh_scaled)[0])
    weight = np.zeros(np.shape(kmesh_scaled)[0])
    for i, ki in enumerate(kmesh_scaled):
        ki = [wrap_1stBZ(l) for l in ki]
        kmesh_scaled[i] = ki

    # Time-reversal symmetry
    Inv = (-1) * np.identity(3)
    for i, ki in enumerate(kmesh_scaled):
        ki = np.dot(Inv,ki)
        ki = [wrap_1stBZ(l) for l in ki]
        for l, kl in enumerate(kmesh_scaled[:i]):
            if np.allclose(ki,kl):
                ind[i] = l
                break

    uniq = np.unique(ind, return_counts=True)
    for i, k in enumerate(uniq[0]):
        weight[k] = uniq[1][i]
    ir_list = uniq[0]

    # Mark down time-reversal-reduced k-points
    conj_list = np.zeros(len(kmesh_scaled))
    for i, k in enumerate(ind):
        if i != k:
            conj_list[i] = 1

    return ir_list, ind, weight, conj_list

def wrap_k(k):
    while k < 0 :
        k = 1 + k
    while (k - 9.9999999999e-1) > 0.0 :
        k = k - 1
    return k

def parse_basis(basis_list):
    '''
    Parse information about chosen basis sets
    '''
    logging.debug(f"{basis_list}, {len(basis_list) % 2}")
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
    '''
    parse geometry of the system
    '''
    res = ""
    if os.path.exists(g) :
        with open(g) as gf:
            res = gf.read()
    else:
        res = g
    return res

def save_data(args, mycell, mf, kmesh, ind, weight, num_ik, ir_list, conj_list, Nk, nk, NQ, F, S, T, hf_dm, madelung, Zs, last_ao):
    '''
    Save data in Green/WeakCoupling format into a hdf5 file
    '''
    kptij_idx, kij_conj, kij_trans, kpair_irre_list, num_kpair_stored, kptis, kptjs = int_utils.integrals_grid(mycell, kmesh)
    logging.info(f"number of reduced k-pairs: {num_kpair_stored}")
    inp_data = h5py.File(args.output_path, "w")
    inp_data["grid/k_mesh"] = kmesh
    inp_data["grid/k_mesh_scaled"] = mycell.get_scaled_kpts(kmesh)
    inp_data["grid/index"] = ind
    inp_data["grid/weight"] = weight
    inp_data["grid/ink"] = num_ik
    inp_data["grid/nk"] = nk
    inp_data["grid/ir_list"] = ir_list
    inp_data["grid/conj_list"] = conj_list
    inp_data["grid/conj_pairs_list"] = kij_conj
    inp_data["grid/trans_pairs_list"] = kij_trans
    inp_data["grid/kpair_irre_list"] = kpair_irre_list
    inp_data["grid/kpair_idx"] = kptij_idx
    inp_data["grid/num_kpair_stored"] = num_kpair_stored
    inp_data["HF/Nk"] = Nk
    inp_data["HF/nk"] = nk
    inp_data["HF/Energy"] = mf.e_tot
    inp_data["HF/Energy_nuc"] = mf.energy_nuc()
    inp_data["HF/Fock-k"] = F.view(np.float64).reshape(F.shape[0], F.shape[1], F.shape[2], F.shape[3], 2)
    inp_data["HF/Fock-k"].attrs["__complex__"] = np.int8(1)
    inp_data["HF/S-k"] = S.view(np.float64).reshape(S.shape[0], S.shape[1], S.shape[2], S.shape[3], 2)
    inp_data["HF/S-k"].attrs["__complex__"] = np.int8(1)
    inp_data["HF/H-k"] = T.view(np.float64).reshape(T.shape[0], T.shape[1], T.shape[2], T.shape[3], 2)
    inp_data["HF/H-k"].attrs["__complex__"] = np.int8(1)
    inp_data["HF/madelung"] = madelung
    inp_data["HF/mo_energy"] = mf.mo_energy
    inp_data["HF/mo_coeff"] = mf.mo_coeff
    inp_data["mulliken/Zs"] = Zs
    inp_data["mulliken/last_ao"] = last_ao
    inp_data["params/nao"] = mycell.nao_nr()
    inp_data["params/nso"] = S.shape[2]
    inp_data["params/ns"] = S.shape[0]
    inp_data["params/nel_cell"] = mycell.nelectron
    inp_data["params/nk"] = kmesh.shape[0]
    inp_data["params/NQ"] = NQ
    inp_data.attrs["__green_version__"] = imd.version("green_mbtools")
    inp_data.close()
    chk.save(args.output_path, "Cell", mycell.dumps())
    inp_data = h5py.File("dm.h5", "w")
    inp_data["HF/dm-k"] = hf_dm.view(np.float64).reshape(hf_dm.shape[0], hf_dm.shape[1], hf_dm.shape[2], hf_dm.shape[3], 2)
    inp_data["HF/dm-k"].attrs["__complex__"] = np.int8(1)
    inp_data["dm_gamma"] = hf_dm[:, 0, :, :]
    inp_data.close()


def orthogonalize(mydf, orth, X_k, X_inv_k, F, T, hf_dm, S):
    '''
    Transform Fock-matrix, non-interacting Hamiltonian, density matrix and overlap matrix into an orthogonal basis
    '''
    maxdiff = -1
    old_shape = [-1, -1]
    for ik, k in enumerate(mydf.kpts):
        if orth == 0:
            X_inv_k.append(np.eye(F.shape[2], dtype=np.complex128))
            X_k.append(np.eye(F.shape[2], dtype=np.complex128))
            continue

        s_ev, s_eb = np.linalg.eigh(S[0, ik])

        # Remove all eigenvalues < threshold
        istart = s_ev.searchsorted(1e-9)
        s_sqrtev = np.sqrt(s_ev[istart:])

        x_pinv = s_eb[:, istart:] * s_sqrtev
        x = (s_eb[:, istart:].conj() * 1 / s_sqrtev).T
        n_ortho, n_nonortho = x.shape
        if old_shape[0] >= 0 and n_ortho != old_shape[0] and n_nonortho != old_shape[1]:
            raise RuntimeError("Error!!! Different k-point have different number of orthogonal basis.")

        old_shape[0] = n_ortho
        old_shape[1] = n_nonortho

        X_inv_k.append(x_pinv.copy())
        X_k.append(x.copy())

        diff = np.eye(n_nonortho) - np.dot(x, x_pinv)
        diff_max = np.max(np.abs(diff))
        maxdiff = max(maxdiff, diff_max)
    logging.info(f"max diff from identity {maxdiff}")
    X_inv_k = np.asarray(X_inv_k).reshape(F.shape[1:])
    X_k = np.asarray(X_k).reshape(F.shape[1:])
    # Orthogonalization
    if orth == 1:
        F = transform(F, X_k, X_inv_k)
        # S     = transform(S, X_k, X_inv_k)
        T = transform(T, X_k, X_inv_k)
        hf_dm = transform(hf_dm, X_inv_k, X_k)

        S = np.array([np.eye(F.shape[-1], dtype=np.complex128)] * F.shape[1])
        S = np.array([S, S])

    return X_k, X_inv_k, S, F, T, hf_dm

def add_common_params(parser):
    '''
    Define common command line arguments for Green python module
    '''
    parser.add_argument("--atom", type=parse_geometry, help="poistions of atoms", required=True)
    parser.add_argument("--Nk", type=int, default=0, help="number of plane-waves in each direction for integral evaluation")
    parser.add_argument("--basis", type=str, nargs="*", help="basis sets definition. First specify atom then basis for this atom", required=True)
    parser.add_argument("--auxbasis", type=str, nargs="*", default=[None], help="auxiliary basis")
    parser.add_argument("--ecp", type=str, nargs="*", default=[None], help="effective core potentials")
    parser.add_argument("--xc", type=str, nargs="*", default=[None], help="XC functional")
    parser.add_argument("--dm0", type=str, default=None, help="initial guess for density matrix")
    parser.add_argument("--df_int", type=int, default=1, help="prepare density fitting integrals or not")
    parser.add_argument("--int_path", type=str, default="df_int", help="path to store ewald corrected integrals")
    parser.add_argument("--hf_int_path", type=str, default="df_hf_int", help="path to store hf integrals")
    parser.add_argument("--output_path", type=str, default="input.h5", help="output file with initial data")
    parser.add_argument("--orth", type=int, default=0, help="Transform to orthogonal basis or not. 0 - no orthogonal transformation, 1 - data is in orthogonal basis.")
    parser.add_argument("--beta", type=float, default=None, help="Emperical parameter for even-Gaussian auxiliary basis")
    parser.add_argument("--active_space", type=int, nargs='+', default=None, help="active space orbitals")
    parser.add_argument("--spin", type=int, default=0, help="Local spin")
    parser.add_argument("--restricted", type=lambda x: (str(x).lower() in ['true','1', 'yes']), default='false', help="Spin restricted calculations.")
    parser.add_argument("--memory", type=int, default=700, help="Memory bound for integral chunk in MB")
    parser.add_argument("--grid_only", type=lambda x: (str(x).lower() in ['true','1', 'yes']), default='false', help="Only recompute k-grid points")
    parser.add_argument("--diffuse_cutoff", type=float, default=0.0, help="Remove the diffused Gaussians whose exponents are less than the cutoff")
    parser.add_argument("--damping", type=float, default=0.0, help="Damping factor for mean-field iterations")
    parser.add_argument("--max_iter", type=int, default=100, help="Maximum number of iterations in the SCF loop")
    parser.add_argument("--keep_cderi", type=lambda x: (str(x).lower() in ['true','1', 'yes']), default='false', help="Keep generated cderi files.")
    parser.add_argument("--job", choices=["init", "sym_path", "ewald_corr"], default="init", nargs="+")
    parser.add_argument(
        "--x2c", type=int, default=0, choices=[0, 1, 2],
        help="enable X2C calculations (0: non-rel., 1: sfX2C1e, 2: X2C1e)"
    )


def add_pbc_params(parser):
    '''
    Define PBC-specific command line arguments for Green python module
    '''
    parser.add_argument("--a", type=parse_geometry, help="lattice geometry", required=True)
    parser.add_argument("--nk", type=int, help="number of k-points in each direction", required=True)
    parser.add_argument("--pseudo", type=str, nargs="*", default=[None], help="pseudopotential")
    parser.add_argument("--shift", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="mesh shift")
    parser.add_argument("--center", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="mesh center")
    parser.add_argument("--symm", type=lambda x: (str(x).lower() in ['true','1', 'yes']), default='true', help="Use inversion symmetry")
    parser.add_argument("--print_high_symmetry_points", default=False, action='store_true', help="Print available high symmetry points for current system and exit.")
    parser.add_argument("--high_symmetry_path", type=str, default=None, help="High symmetry path")
    parser.add_argument("--high_symmetry_path_points", type=int, default=0, help="Number of points for high symmetry path")
    parser.add_argument("--finite_size_kind", choices=["ewald", "gf2", "gw", "gw_s", "coarse_grained"], default="ewald", nargs="+", 
                              help="Two body finite-size correction. Be default computes the second set of integrals that include simple ewald correction.")

def init_mol_params():
    '''
    Initialize argparse.ArgumentParser for Green/WeakCoupling python module and return a prased parameters map with parameters specific for molecular calculations
    '''
    parser = argparse.ArgumentParser(description="Green/WeakCoupling initialization script")
    add_common_params(parser)
    args = parser.parse_args()
    args.basis = parse_basis(args.basis)
    args.auxbasis = parse_basis(args.auxbasis)
    args.ecp = parse_basis(args.ecp)
    args.xc = parse_basis(args.xc)
    if args.x2c == 2 and args.restricted:
        raise RuntimeError("X2C calculation can not be spin restricted")
    if args.xc is not None:
        if args.x2c == 2:
            args.mean_field = mdft.GKS
        else:
            args.mean_field = mdft.RKS if args.restricted else mdft.UKS
    else:
        if args.x2c == 2:
            args.mean_field = mscf.GHF
        else:
            args.mean_field = mscf.RHF if args.restricted else mscf.UHF
    args.ns = 1 if args.restricted or args.x2c == 2 else 2
    # parameters needed to create empty grid
    args.a = [[1,0,0],[0,1,0],[0,0,1]]
    args.nk = 1
    args.shift =  [0.,0.,0.]
    args.center = [0.,0.,0.]
    return args

def init_pbc_params():
    '''
    Initialize argparse.ArgumentParser for Green/WeakCoupling python module and return a prased parameters map with parameters specific for periodic calculations
    '''
    parser = argparse.ArgumentParser(description="Green/WeakCoupling initialization script")
    add_common_params(parser)
    add_pbc_params(parser)
    args = parser.parse_args()
    args.basis = parse_basis(args.basis)
    args.auxbasis = parse_basis(args.auxbasis)
    args.ecp = parse_basis(args.ecp)
    args.pseudo = parse_basis(args.pseudo)
    args.xc = parse_basis(args.xc)
    if args.x2c == 2 and args.restricted:
        raise RuntimeError("X2C calculation can not be spin restricted")
    if args.xc is not None:
        if args.x2c == 2:
            args.mean_field = dft.KGKS
        else:
            args.mean_field = dft.KRKS if args.restricted else dft.KUKS
    else:
        if args.x2c == 2:
            args.mean_field = scf.KGHF
        else:
            args.mean_field = scf.KRHF if args.restricted else scf.KUHF
    args.ns = 1 if args.restricted or args.x2c else 2
    return args

def mol_cell(args):
    '''
    Initialize PySCF unit cell object for a given system
    '''
    c = mgto.M(
        atom = args.atom,
        unit = 'A',
        basis = args.basis,
        ecp = args.ecp,
        verbose = 7,
        spin = args.spin
    )
    return c

def pbc_cell(args):
    '''
    Initialize PySCF unit cell object for a given system
    '''
    c = gto.M(
        a = args.a,
        atom = args.atom,
        unit = 'A',
        basis = args.basis,
        ecp = args.ecp,
        pseudo = args.pseudo,
        verbose = 7,
        spin = args.spin,
#        exp_to_discard = args.diffuse_cutoff
    )
    _a = c.lattice_vectors()
    c.exp_to_discard = args.diffuse_cutoff
    c.build()
    if np.linalg.det(_a) < 0:
        raise "Lattice are not in right-handed coordinate system. Please correct your lattice vectors"
    return c

def wrap_1stBZ(k):
    '''
    wrap scaled k into [-0.5,0.5) range
    :param k: value of k-point at some dimension
    '''
    while k < -0.5 :
        k = k + 1
    while (k - 4.9999999999e-1) > 0.0 :
        k = k - 1
    return k

def init_k_mesh(args, mycell):
    '''
    init k-points mesh for GDF

    :param args: script arguments
    :param mycell: unit cell for simulation
    :return: kmesh, k_ibz, ir_list, conj_list, weight, ind, num_ik
    '''
    if args.center is None:
       args.center = [0,0,0]
    if args.shift is None:
       args.shift = [0,0,0]
    kmesh = mycell.make_kpts([args.nk, args.nk, args.nk], scaled_center=args.center)
    for i, kk in enumerate(kmesh):
        ki = kmesh[i]
        ki = mycell.get_scaled_kpts(ki) + args.shift
        ki = [wrap_k(l) for l in ki]
        kmesh[i] = mycell.get_abs_kpts(ki)
    for i, ki in enumerate(kmesh):
        ki = mycell.get_scaled_kpts(ki)
        ki = [wrap_k(l) for l in ki]
        ki = mycell.get_abs_kpts(ki)
        kmesh[i] = ki

    logging.debug(kmesh)
    logging.info(mycell.get_scaled_kpts(kmesh))

    if not args.symm :
        nkpts = kmesh.shape[0]
        weight = np.ones(nkpts)
        ir_list = np.array(range(nkpts))
        ind = np.array(range(nkpts))
        conj_list = np.zeros(nkpts)
        k_ibz = np.copy(kmesh)
        num_ik = nkpts
        return kmesh, k_ibz, ir_list, conj_list, weight, ind, num_ik

    logging.info("Compute irreducible k-points")


    k_ibz = mycell.make_kpts([args.nk,args.nk,args.nk], scaled_center=args.center)
    ind = np.arange(np.shape(k_ibz)[0])
    weight = np.zeros(np.shape(k_ibz)[0])
    for i, ki in enumerate(k_ibz):
        ki = mycell.get_scaled_kpts(ki)
        ki = [wrap_1stBZ(l) for l in ki]
        k_ibz[i] = ki

    # Time-reversal symmetry
    Inv = (-1) * np.identity(3)
    for i, ki in enumerate(k_ibz):
        ki = np.dot(Inv,ki)
        ki = [wrap_1stBZ(l) for l in ki]
        for l, kl in enumerate(k_ibz[:i]):
            if np.allclose(ki,kl):
                k_ibz[i] = kl
                ind[i] = l
                break

    uniq = np.unique(ind, return_counts=True)
    for i, k in enumerate(uniq[0]):
        weight[k] = uniq[1][i]
    ir_list = uniq[0]

    # Mark down time-reversal-reduced k-points
    conj_list = np.zeros(args.nk**3)
    for i, k in enumerate(ind):
        if i != k:
            conj_list[i] = 1
    num_ik = np.shape(uniq[0])[0]

    return kmesh, k_ibz, ir_list, conj_list, weight, ind, num_ik



def read_dm(dm0, dm_file):
    '''
    Read density matrix from smaller kmesh
    '''
    nao  = dm0.shape[-1]
    nkpts = dm0.shape[1]
    dm   = np.zeros((2,nao,nao),dtype=np.complex128)
    f    = h5py.File(dm_file, 'r')
    dm[:,:,:] = f['/dm_gamma'][:]
    f.close()
    dm_kpts = np.repeat(dm[:,None, :, :], nkpts, axis=1)
    return dm_kpts

def solve_mean_field(args, mydf, mycell):
    '''
    Obtain pySCF mean-field solution for a given parameters, unit-cell object and density-fitting object
    '''
    logging.info("Solve Mean-field")
    # prepare and solve DFT
    if args.x2c == 0:
        mf = args.mean_field(mycell, mydf.kpts).density_fit()
    elif args.x2c == 1:
        mf = args.mean_field(mycell, mydf.kpts).density_fit().sfx2c1e()
    elif args.x2c == 2:
        mf = args.mean_field(mycell, mydf.kpts).density_fit().x2c1e()
    if args.xc is not None:
        mf.xc = args.xc
    #mf.max_memory = 10000
    mydf._cderi = "cderi.h5"
    mf.kpts = mydf.kpts
    mf.with_df = mydf
    mf.diis_space = 16
    mf.damp = args.damping
    mf.max_cycle = args.max_iter
    mf.chkfile = 'tmp.chk'
    if os.path.exists("tmp.chk"):
        init_dm = mf.from_chk('tmp.chk')
        mf.kernel(init_dm)
    elif args.dm0 is not None:
        init_dm = mf.get_init_guess()
        init_dm = read_dm(init_dm, args.dm0)
        mf.kernel(init_dm)
    else:
        mf.kernel()
    if args.x2c < 2:
        mf.analyze()
    return mf

def solve_mol_mean_field(args, mydf, mycell):
    '''
    Obtain pySCF mean-field solution for a given parameters, unit-cell object and density-fitting object
    '''
    logging.info("Solve LDA")
    # prepare and solve DFT

    if args.x2c == 0:
        mf = args.mean_field(mycell).density_fit()
    elif args.x2c == 1:
        mf = args.mean_field(mycell).density_fit().sfx2c1e()
    elif args.x2c == 2:
        mf = args.mean_field(mycell).x2c1e()
    if args.xc is not None:
        mf.xc = args.xc
    # mf.max_memory = 10000
    # mydf._cderi = "cderi.h5"
    if args.x2c == 2:
        tmp_mf = mscf.RHF(mycell).density_fit() if args.restricted else mscf.UHF(mycell).density_fit()
        tmp_mf.with_df._cderi_to_save = "cderi_mol.h5"
        tmp_mf.with_df.build()
        tmp_mf = None
    else:
        mf.with_df._cderi_to_save = "cderi_mol.h5"
        mf.with_df.build()
    mf.diis_space = 16
    mf.damp = args.damping
    mf.max_cycle = args.max_iter
    mf.chkfile = 'tmp.chk'
    if os.path.exists("tmp.chk"):
        init_dm = mf.from_chk('tmp.chk')
        mf.kernel(init_dm)
    elif args.dm0 is not None:
        init_dm = mf.get_init_guess()
        init_dm = read_dm(init_dm, args.dm0)
        mf.kernel(init_dm)
    else:
        mf.kernel()
    if args.x2c < 2:
        mf.analyze()
    return mf


def store_k_grid(args, mycell, kmesh, k_ibz, ir_list, conj_list, weight, ind, num_ik):
    '''
    Store reciprocal space information into a hdf5 file in Green'WeakCoupling format
    '''
    inp_data = h5py.File(args.output_path, "a")
    nk = kmesh.shape[0]
    ink = k_ibz.shape[0]
    kptij_idx, kij_conj, kij_trans, kpair_irre_list, num_kpair_stored, kptis, kptjs = int_utils.integrals_grid(mycell, kmesh)
    logging.info(f"number of reduced k-pairs: {num_kpair_stored}")
    if not "grid" in inp_data:
        inp_data.create_group("grid")
    grid_grp = inp_data["grid"]
    data = [kmesh, mycell.get_scaled_kpts(kmesh), ind, weight, num_ik, nk, ir_list, conj_list,
               kij_conj, kij_trans, kpair_irre_list, kptij_idx, num_kpair_stored]
    names = ["k_mesh", "k_mesh_scaled", "index", "weight", "ink", "nk", "ir_list", "conj_list",
                "conj_pairs_list", "trans_pairs_list", "kpair_irre_list", "kpair_idx", "num_kpair_stored" ]
    for i, name in enumerate(names):
        if name in grid_grp:
            grid_grp[name][...] = data[i]
        else:
            grid_grp[name] = data[i]
    inp_data.close()

def construct_mol_gdf(args, mycell):
    '''
    Construct Gaussian Density Fitting obejct for a given parameters and unit cell.
    We make sure to disable range-separeting implementation
    '''
    # Use gaussian density fitting to get fitted densities
    mydf = df.GDF(mycell, mycell.kpts)
    if args.auxbasis is not None:
        mydf.auxbasis = args.auxbasis
    elif args.beta is not None:
        mydf.auxbasis = df.aug_etb(mycell, beta=args.beta)
    # Coulomb kernel mesh
    if args.Nk > 0:
        mydf.mesh = [args.Nk, args.Nk, args.Nk]
    return mydf


def construct_gdf(args, mycell, kmesh=None):
    '''
    Construct Gaussian Density Fitting obejct for a given parameters and unit cell.
    We make sure to disable range-separeting implementation
    '''
    # Use gaussian density fitting to get fitted densities
    mydf = df.GDF(mycell)
    if hasattr(mydf, "_prefer_ccdf"):
        mydf._prefer_ccdf = True  # Disable RS-GDF switch for new pyscf versions 
    if args.auxbasis is not None:
        mydf.auxbasis = args.auxbasis
    elif args.beta is not None:
        mydf.auxbasis = df.aug_etb(mycell, beta=args.beta)
    # Coulomb kernel mesh
    if args.Nk > 0:
        mydf.mesh = [args.Nk, args.Nk, args.Nk]
    if kmesh is not None:
        mydf.kpts = kmesh
    return mydf


def compute_ewald_correction(args, cell, kmesh, filename):
    # Use gaussian density fitting to get fitted densities
    mydf = df.GDF(cell)
    if args.auxbasis is not None:
        mydf.auxbasis = args.auxbasis
    elif args.beta is not None:
        mydf.auxbasis = df.aug_etb(cell, beta=args.beta)
    # Coulomb kernel mesh
    if args.Nk > 0:
        mydf.mesh = [args.Nk, args.Nk, args.Nk]
    int_utils.compute_ewald_correction(args, mydf, kmesh, cell.nao_nr(), filename)

def compute_df_int_dca(args, mycell, kmesh, lattice_kmesh, nao, X_k):
    '''
    Generate density-fitting integrals for correlated methods using q-averaging over the super-lattice points to compensate finite-size error
    
    :param args: simulation parameters
    :param mycell: unit cell object
    :param kmesh: reciprocal space grid
    :param lattice_kmesh: super-lattice k-points
    :param nao: number of atomic orbitals in the unit cell
    :param X_k: trasformation matrix for projection onto an orthogonal space
    '''

    if not bool(args.df_int):
        return
    mydf = construct_gdf(args, mycell, kmesh)
    # Use Ewald for divergence treatment
    mydf.exxdiv = 'ewald'
    weighted_coulG_old = df.GDF.weighted_coulG
    df.GDF.weighted_coulG = int_utils.weighted_coulG_ewald
    old_get_coulG = tools.get_coulG
    tools.get_coulG = lambda cell, k=np.zeros(3), exx=False, mf=None, mesh=None, Gv=None, wrap_around=True, omega=None, **kwargs: int_utils.get_coarsegrained_coulG(lattice_kmesh, cell, k, exx, mf, mesh, Gv,
              wrap_around, omega, **kwargs)

    kij_conj, kij_trans, kpair_irre_list, kptij_idx, num_kpair_stored = int_utils.compute_integrals(mycell, mydf, kmesh, nao, X_k, args.int_path, "cderi_ewald_dca.h5", False)

    mydf = None
    mydf = construct_gdf(args, mycell, kmesh)
    df.GDF.weighted_coulG = weighted_coulG_old
    int_utils.compute_integrals(mycell, mydf, kmesh, nao, X_k, args.hf_int_path, "cderi_dca.h5", False)

    tools.get_coulG = old_get_coulG

