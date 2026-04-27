import os

import h5py
import logging
import numpy as np
from pyscf.df import addons
from pyscf.pbc import tools, gto
from pyscf.pbc.lib import kpts as libkpts

from . import gdf_s_metric as gdf_S
from . import common_utils as comm
from . import integral_utils as int_utils
from . import symmetry_utils as symm_utils
from ..pesto import ft



class pyscf_init:
    '''Initialization class for Green project

    Attributes
    ----------
    args : map
        simulation parameters
    cell : pyscf.pbc.cell
        unit cell object
    kmesh : numpy.ndarray
        Monkhorst-Pack reciprocal space grid
    '''

    def __init__(self, args):
        '''
        Initialize PySCF interoperation class

        Parameters
        ----------
        args: map
            simulation parameters
        '''
        self.args = args
        if self.args.Nk is None:
            self.args.Nk = 0
        if self.args.spin is None:
            self.args.spin = 0
        if self.args.damping is None:
            self.args.damping = 0
        if self.args.max_iter is None:
            self.args.max_iter = 100
        self.cell = self.cell_object()
        
    
    def compute_df_int(self, nao, X_k):
        raise NotImplementedError("Please Implement this method")
    def mf_object(self, mydf=None):
        raise NotImplementedError("Please Implement this method")
    def df_object(self, mydf=None):
        raise NotImplementedError("Please Implement this method")
    def cell_object(self):
        raise NotImplementedError("Please Implement this method")
    def mean_field_input(self, mydf=None):
        raise NotImplementedError("Please Implement this method")



class pyscf_pbc_init (pyscf_init):
    """Initialization class for periodic / solid-state systems for the Green project
    """
    def __init__(self, args=None):
        super().__init__(comm.init_pbc_params() if args is None else args)
        self.kmesh, self.k_ibz, self.ir_list, self.conj_list, self.weight, self.ind, self.num_ik, self.kstruct = \
            comm.init_k_mesh(self.args, self.cell)

    def mean_field_input(self, mydf=None):
        """Solve a given mean-field problem and store the solution in the Green/WeakCoupling format
        
        Parameters
        ----------
        mydf : pyscf.pbc.df
            pyscf density-fitting object, will be generated if None
        """

        # Generate integrals for DFT and MBPT calculations
        if mydf is None:
            mydf = self.df_object()

        if os.path.exists("cderi.h5"):
            mydf._cderi = "cderi.h5"
        else:
            mydf._cderi_to_save = "cderi.h5"
            mydf.build()
        # number of k-points in each direction for Coulomb integrals
        nk       = np.prod(self.args.nk)
        # number of k-points in each direction to evaluate Coulomb kernel
        Nk       = self.args.Nk

        # number of orbitals per cell
        nao = self.cell.nao_nr()
        nso = 2*self.cell.nao_nr() if self.args.x2c == 2 else self.cell.nao_nr()
        Zs = np.asarray(self.cell.atom_charges())
        logging.info(f"Number of atoms: {Zs.shape[0]}")
        logging.info(f"Effective nuclear charge of each atom: {Zs}")
        atoms_info = np.asarray(self.cell.aoslice_by_atom())
        last_ao = atoms_info[:,3]
        logging.info(f"aoslice_by_atom = {atoms_info}")
        logging.info(f"Last AO index for each atom = {last_ao}")

        if self.args.grid_only:
            comm.store_k_grid(self.args, self.cell, self.kmesh, self.k_ibz, self.ir_list, self.conj_list, self.weight, self.ind, self.num_ik)
            auxcell = addons.make_auxmol(self.cell, mydf.auxbasis)
            # NOTE: if args.orth = 1, we will not be able to transform the k_sym_transform_ao yet.
            comm.store_kstruct_ops_info(self.args, self.cell, self.kmesh, self.kstruct)
            comm.store_auxcell_kstruct_ops_info(self.args, auxcell, self.kmesh)
            return

        '''
        Generate integrals for mean-field calculations
        '''
        auxcell = addons.make_auxmol(self.cell, mydf.auxbasis)
        NQ = auxcell.nao_nr()
    
        mf = self.mf_object(mydf)
    
        # Get Overlap and Fock matrices
        hf_dm = mf.make_rdm1().astype(dtype=np.complex128)
        S     = mf.get_ovlp().astype(dtype=np.complex128)
        T     = mf.get_hcore().astype(dtype=np.complex128)
        if self.args.xc is not None:
            vhf = mf.get_veff().astype(dtype=np.complex128)
        else:
            vhf = mf.get_veff(dm_kpts=hf_dm).astype(dtype=np.complex128)
        F = mf.get_fock(T,S,vhf,hf_dm).astype(dtype=np.complex128)
    
        if len(F.shape) == 3:
            F     = F.reshape((1,) + F.shape)
            hf_dm = hf_dm.reshape((1,) + hf_dm.shape)
        S = np.array((S, ) * self.args.ns)
        T = np.array((T, ) * self.args.ns)
    
        X_k = []
        X_inv_k = []

        # Orthogonalization matrix
        X_k, X_inv_k, S, F, T, hf_dm = comm.orthogonalize(mydf, self.args.orth, X_k, X_inv_k, F, T, hf_dm, S)
        # Save data into Green Software package input format.
        comm.save_data(
            self.args, self.cell, mf, self.kmesh, self.ind, self.weight, self.num_ik, self.ir_list, self.conj_list,
            Nk, nk, NQ, F, S, T, hf_dm, tools.pbc.madelung(self.cell, self.kmesh), Zs, last_ao
        )
        # Save symmetry operations info for main and auxiliary unit cells
        comm.store_kstruct_ops_info(self.args, self.cell, self.kmesh, self.kstruct, X_k=X_k, X_inv_k=X_inv_k,)
        comm.store_auxcell_kstruct_ops_info(self.args, auxcell, self.kmesh)

        # Diagnose whether self-consistent quantities obey k-space symmetry.
        if self.args.space_symm or self.args.tr_symm:
            symm_utils.check_kspace_symmetry_breaking(self.args.output_path, ["HF/H-k", "HF/S-k", "HF/Fock-k"])

        # Store density-fitted integrals
        if bool(self.args.df_int) :
            self.compute_df_int(nao, X_k)

    def compute_df_int(self, nao, X_k):
        '''
        Generate density-fitting (DF) three-center Coulomb integrals for correlated methods.

        Two separate integral sets are produced and written to disk:

        1. Mean-field integrals (written to ``args.hf_int_path``):
           Standard DF integrals L^Q_{pq}(k_i, k_j) for all symmetry-
           irreducible k-point pairs, computed with the bare Coulomb kernel.
           These are used in the mean-field and Hartree-Fock steps.

        2. Correlated integrals (written to ``args.int_path``):
           Same DF integrals but with a finite-size correction applied to
           the diagonal (k_i == k_j) pairs.  The correction strategy depends
           on ``args.finite_size_kind``:

           - ``gf2`` / ``gw`` / ``gw_s``: delegates to
             ``compute_twobody_finitesize_correction()``, which uses the
             GF2 Ewald subtraction scheme or the GW plane-wave transformation
             respectively, then returns early.

           - ``ewald`` (default): builds a second set of three-center integrals
             with the Ewald Coulomb kernel via ``green_igen.df._make_j3c`` and
             passes them to ``compute_integrals`` as ``cderi_name2``; the
             diagonal pairs in the output are then replaced by the
             Ewald-corrected values.

        Parameters
        ----------
        nao : int
            Number of non-relativistic atomic orbitals per k-point.
            Always ``cell.nao_nr()`` regardless of the X2C level, because
            the Coulomb integrals are non-relativistic.
        X_k : list of ndarray
            Per-k-point Löwdin orthogonalisation matrices X(k) = S(k)^{-1/2}.
            Empty list when orthogonalisation is disabled (``args.orth == 0``).
        '''
        import datetime
        def _ts(tag):
            print(f"[{datetime.datetime.now().isoformat()}] compute_df_int: {tag}", flush=True)

        _ts(f"ENTER — nao={nao}, finite_size_kind={self.args.finite_size_kind}, nk={len(self.kmesh)}")

        # --- Step 1: mean-field integrals (bare Coulomb kernel) --------------
        _ts("STEP 1 — constructing GDF object for mean-field integrals")
        mydf = comm.construct_gdf(self.args, self.cell, self.kmesh)
        _ts("STEP 1 — calling compute_integrals (bare Coulomb) → hf_int_path")
        int_utils.compute_integrals(self.args, self.cell, mydf, self.kmesh, nao, X_k, self.args.hf_int_path, "cderi.h5", True, True)
        mydf = None
        _ts("STEP 1 — done")

        # --- Step 2: correlated integrals with finite-size correction --------
        # GF2/GW corrections use a separate code path that handles the
        # correction internally; the plain Ewald correction is handled below.
        if 'gf2' in self.args.finite_size_kind or 'gw' in self.args.finite_size_kind or 'gw_s' in self.args.finite_size_kind:
            _ts(f"STEP 2 — delegating to compute_twobody_finitesize_correction (finite_size_kind={self.args.finite_size_kind})")
            self.compute_twobody_finitesize_correction()
            if not self.args.keep_cderi:
                os.remove("cderi.h5")
                os.system("sync")
            _ts("STEP 2 — done, returning early")
            return

        # --- Step 3: Ewald correction via green_igen._make_j3c ---------------
        # Build a second GDF object and construct three-center integrals with
        # the Ewald Coulomb kernel for the diagonal k-pairs (k_i == k_j) only.
        # These are written to cderi_ewald.h5 and later substituted for the
        # diagonal entries in the correlated integral set.
        #
        # The Ewald kernel is installed by monkey-patching gdf.GDF.weighted_coulG
        # on the class (not the instance) because green_igen._make_j3c resolves
        # the method through the class hierarchy.  The original method is saved
        # before the patch and unconditionally restored afterwards so that no
        # subsequent GDF construction in this session is affected.
        from pyscf.pbc import df as gdf
        import green_igen.df as gggdf

        _ts("STEP 3 — constructing GDF object for Ewald-corrected integrals")
        mydf = comm.construct_gdf(self.args, self.cell, self.kmesh)
        mydf.exxdiv = 'ewald'
        auxcell = gggdf.make_modrho_basis(mydf.cell, mydf.auxbasis,
                                          mydf.exp_to_discard)
        kptij_lst = np.asarray([(ki, ki) for ki in self.kmesh])
        _ts(f"STEP 3 — auxcell built: naux={auxcell.nao_nr()}, kptij_lst shape={kptij_lst.shape}")

        # Save → patch → build → restore.
        _ts("STEP 3 — patching weighted_coulG → Ewald kernel, calling _make_j3c")
        weighted_coulG_old = gdf.GDF.weighted_coulG
        gdf.GDF.weighted_coulG = int_utils.weighted_coulG_ewald
        gggdf._make_j3c(mydf, self.cell, auxcell, kptij_lst, "cderi_ewald.h5")
        gdf.GDF.weighted_coulG = weighted_coulG_old  # always restore
        _ts("STEP 3 — _make_j3c done, weighted_coulG restored")

        # Build correlated integrals; diagonal pairs come from cderi_ewald.h5.
        _ts("STEP 3 — calling compute_integrals (Ewald-corrected) → int_path")
        int_utils.compute_integrals(self.args, self.cell, mydf, self.kmesh, nao, X_k, self.args.int_path, "cderi.h5", True, self.args.keep_cderi, cderi_name2="cderi_ewald.h5")
        _ts("STEP 3 — done, compute_df_int complete")

    def evaluate_high_symmetry_path(self):
        if self.args.print_high_symmetry_points:
            comm.print_high_symmetry_points(self.args)
            return
        if self.args.high_symmetry_path is None:
            raise RuntimeError("Please specify high-symmetry path")
        if self.args.high_symmetry_path is not None:
            try:
                comm.check_high_symmetry_path(self.args)
            except RuntimeError as e:
                logging.error("\n\n\n")
                logging.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                logging.error("!!!!!!!!! Cannot compute high-symmetry path !!!!!!!!!")
                logging.error("!! Correct or Disable high-symmetry path evaluation !")
                logging.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                logging.error(e)
                exit(-1)
        kmesh_hs, Hk_hs, Sk_hs, lin_kpt_axis = comm.high_symmetry_path(
            self.cell, self.args
        )
        xpath, special_points, special_labels = lin_kpt_axis
        inp_data = h5py.File(self.args.output_path, "a")
        logging.debug(kmesh_hs)
        logging.debug(self.cell.get_scaled_kpts(kmesh_hs))
        inp_data["high_symm_path/k_mesh"] = self.cell.get_scaled_kpts(kmesh_hs)
        inp_data["high_symm_path/r_mesh"] = ft.construct_rmesh(*self.args.nk)
        inp_data["high_symm_path/Hk"] = Hk_hs
        inp_data["high_symm_path/Sk"] = Sk_hs
        inp_data["high_symm_path/xpath"] = xpath
        inp_data["high_symm_path/special_points"] = special_points
        inp_data["high_symm_path/special_labels"] = special_labels

    def compute_twobody_finitesize_correction(self, mydf=None):
        if not os.path.exists(self.args.hf_int_path):
            os.mkdir(self.args.hf_int_path)
        if 'gf2' in self.args.finite_size_kind :
            comm.compute_ewald_correction(self.args, self.cell, self.kmesh, self.args.hf_int_path + "/df_ewald.h5")
        if 'gw' in self.args.finite_size_kind :
            self.evaluate_gw_correction(mydf)
            
    
    def evaluate_gw_correction(self, mydf=None):
        if mydf is None:
            mydf = comm.construct_gdf(self.args, self.cell, self.kmesh)
        mydf.build()

        # ? the construct_gdf function being called above uses Coulomb metric, but corrections here are in overlap metric
        use_space_symm = self.args.space_symm and self.args.x2c < 2
        j2c_sqrt, uniq_qpts = gdf_S.make_j2c_sqrt(mydf, self.cell, use_space_symm, self.args.tr_symm)
        
        ''' Transformation matrix from auxiliary basis to plane-wave '''
        AqQ, q_reduced, q_scaled_reduced = gdf_S.transformation_PW_to_auxbasis(
            mydf, self.cell, j2c_sqrt, uniq_qpts, use_space_symm, self.args.tr_symm
        )
        
        q_abs = np.array([np.linalg.norm(qq) for qq in q_reduced])
        q_abs = np.array([round(qq, 8) for qq in q_abs])
        
        # Different prefactors for the GW finite-size correction for testing
        # In practice, the madelung constant is used, which decays as (1/nk).
        X = (6*np.pi**2)/(self.cell.vol*len(self.kmesh))
        X = (2.0/np.pi) * np.cbrt(X)
        
        X2 = 2.0 * np.cbrt(1.0/(self.cell.vol*len(self.kmesh)))
        
        f = h5py.File(self.args.hf_int_path + "/AqQ.h5", 'w')
        f["AqQ"] = AqQ
        f["qs"] = q_reduced
        f["qs_scaled"] = q_scaled_reduced
        f["q_abs"] = q_abs
        f["X"] = X
        f["X2"] = X2
        f["madelung"] = tools.pbc.madelung(self.cell, self.kmesh)
        f.close()

    def mf_object(self, mydf=None):
        return comm.solve_mean_field(self.args, mydf, self.cell)

    def df_object(self, mydf=None):
        return comm.construct_gdf(self.args, self.cell, self.kmesh)

    def cell_object(self):
        return comm.pbc_cell(self.args)

class pyscf_mol_init (pyscf_init):
    '''Initialization class for molecular systems in the Green project
    '''
    def __init__(self, args=None):
        super().__init__(comm.init_mol_params() if args is None else args)
        self.kmesh = np.array([[0.,0.,0.]])
        self.k_ibz = np.array([[0.,0.,0.],])
        self.ir_list = np.array([0])
        self.conj_list= np.array([0])
        self.weight= np.array([1.0])
        self.ind= np.array([0])
        self.num_ik = 1
        self.kcell = gto.Cell(verbose=0)
        self.kcell.a = [[1,0,0],[0,1,0],[0,0,1]]
        self.kcell.atom = self.cell.atom
        self.kcell.spin = self.cell.spin
        self.kcell.charge = self.cell.charge
        self.kcell.unit = 'A'
        self.kcell.basis = self.cell.basis
        self.kcell.kpts = self.kcell.make_kpts([1, 1, 1])
        self.kcell.ecp = self.cell.ecp
        self.kcell.build()
        self.kstruct = libkpts.make_kpts(self.kcell, self.kmesh, space_group_symmetry=False, time_reversal_symmetry=False)

    def mean_field_input(self, mydf=None):
        '''
        Solve a give mean-field problem and store the solution in the Green/WeakCoupling format
        
        Parameters
        ----------
        mydf : pyscf.df
            pyscf density-fitting object, will be generated if None
        '''
        if mydf is None:
            mydf = self.df_object()
#comm.construct_gdf(self.args, self.cell, self.kmesh)

        # number of k-points in each direction for Coulomb integrals
        nk       = np.prod(self.args.nk)
        # number of k-points in each direction to evaluate Coulomb kernel
        Nk       = self.args.Nk

        # number of orbitals per cell
        nao = self.cell.nao_nr()
        nso = 2*self.cell.nao_nr() if self.args.x2c == 2 else self.cell.nao_nr()
        Zs = np.asarray(self.cell.atom_charges())
        logging.info(f"Number of atoms: {Zs.shape[0]}")
        logging.info(f"Effective nuclear charge of each atom: {Zs}")
        atoms_info = np.asarray(self.cell.aoslice_by_atom())
        last_ao = atoms_info[:,3]
        logging.info(f"aoslice_by_atom = {atoms_info}")
        logging.info(f"Last AO index for each atom = {last_ao}")

        '''
        Generate integrals for mean-field calculations
        '''
        auxcell = addons.make_auxmol(self.cell, mydf.auxbasis)
        NQ = auxcell.nao_nr()
    
        mf = self.mf_object(mydf)
    
        # Get Overlap and Fock matrices
        hf_dm = mf.make_rdm1()
        S     = mf.get_ovlp().astype(dtype=np.complex128)
        T     = mf.get_hcore().astype(dtype=np.complex128)
        if self.args.xc is not None:
            vhf = mf.get_veff().astype(dtype=np.complex128)
        else:
            vhf = mf.get_veff(dm=hf_dm).astype(dtype=np.complex128)
        hf_dm = hf_dm.astype(dtype=np.complex128)
        F = mf.get_fock(T,S,vhf,hf_dm).astype(dtype=np.complex128)


        nk_tot = np.prod(self.args.nk)
        F = F.reshape((self.args.ns, nk_tot, nso, nso))
        hf_dm = hf_dm.reshape((self.args.ns, nk_tot, nso, nso))
        S = S.reshape((nk_tot, nso, nso))
        T = T.reshape((nk_tot, nso, nso))
    
        if len(F.shape) == 3:
            F     = F.reshape((1,) + F.shape)
            hf_dm = hf_dm.reshape((1,) + hf_dm.shape)
        S = np.array((S, ) * self.args.ns)
        T = np.array((T, ) * self.args.ns)

        X_k = []
        X_inv_k = []

        # Orthogonalization matrix
        X_k, X_inv_k, S, F, T, hf_dm = comm.orthogonalize(mydf, self.args.orth, X_k, X_inv_k, F, T, hf_dm, S)
        # Save data into Green Software package input format. Here we set Madelung constant to 0 as there is not long range divergence for molecule
        comm.save_data(self.args, self.kcell, mf, self.kmesh, self.ind, self.weight, self.num_ik, self.ir_list, self.conj_list, Nk, nk, NQ, F, S, T, hf_dm, 0.0, Zs, last_ao)
        comm.store_mol_symmetry_info(self.args, self.kcell, auxcell, self.kmesh)
        if bool(self.args.df_int):
            self.compute_df_int(nao, X_k)

    def compute_df_int(self, nao, X_k):
        '''
        Generate density-fitting integrals for correlated methods
        '''
        h_in = h5py.File("cderi_mol.h5", 'r')
        h_out = h5py.File("cderi.h5", 'w')

        j3c_obj = h_in["/j3c"]
        if not isinstance(j3c_obj, h5py.Dataset):  # not a dataset
            if isinstance(j3c_obj, h5py.Group):  # pyscf >= 2.1
                h_in.copy(h_in["/j3c"], h_out, "j3c/0")
            else:
                raise ValueError("Unknown structure of cderi_mol.h5. Perhaps, PySCF upgrade went badly...")
        else:  # pyscf < 2.1
            h_in.copy(h_in["/j3c"], h_out, "j3c/0/0")

        kptij = np.zeros((1, 2, 3))
        h_out["j3c-kptij"] = kptij

        h_in.close()
        h_out.close()
        mydf = comm.construct_gdf(self.args, self.kcell, self.kmesh)
        int_utils.compute_integrals(self.args, self.kcell, mydf, self.kmesh, nao, X_k, "df_hf_int", "cderi.h5", True, self.args.keep_cderi)
        mydf = None

    def df_object(self, mydf=None):
        return comm.construct_mol_gdf(self.args, self.kcell)

    def mf_object(self, mydf=None):
        return comm.solve_mol_mean_field(self.args, mydf, self.cell)

    def cell_object(self):
        return comm.mol_cell(self.args)
