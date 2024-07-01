import os

import h5py
import logging
import numpy as np
from pyscf.df import addons
from pyscf.pbc import tools, gto

from . import gdf_s_metric as gdf_S
from . import common_utils as comm
from . import integral_utils as int_utils



class pyscf_init:
    '''
    Attributes
    ----------
    args : map
        argument's map
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
            argument's map
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
    '''
    '''
    def __init__(self, args=None):
        super().__init__(comm.init_pbc_params() if args is None else args)
        self.kmesh, self.k_ibz, self.ir_list, self.conj_list, self.weight, self.ind, self.num_ik = comm.init_k_mesh(self.args, self.cell)

    def mean_field_input(self, mydf=None):
        '''
        Solve a give mean-field problem and store the solution in the Green/WeakCoupling format
        
        Parameters
        ----------
        mydf : pyscf.pbc.df
            pyscf density-fitting object, will be generated if None
        '''
        if mydf is None:
            mydf = self.df_object()

        if os.path.exists("cderi.h5"):
            mydf._cderi = "cderi.h5"
        else:
            mydf._cderi_to_save = "cderi.h5"
            mydf.build()
        # number of k-points in each direction for Coulomb integrals
        nk       = self.args.nk ** 3
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
            vhf = mf.get_veff(hf_dm).astype(dtype=np.complex128)
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
        comm.save_data(self.args, self.cell, mf, self.kmesh, self.ind, self.weight, self.num_ik, self.ir_list, self.conj_list, Nk, nk, NQ, F, S, T, hf_dm, tools.pbc.madelung(self.cell, self.kmesh), Zs, last_ao)
        if bool(self.args.df_int) :
            self.compute_df_int(nao, X_k)

    def compute_df_int(self, nao, X_k):
        '''
        Generate density-fitting integrals for correlated methods
        '''
        mydf = comm.construct_gdf(self.args, self.cell, self.kmesh)
        int_utils.compute_integrals(self.args, self.cell, mydf, self.kmesh, nao, X_k, "df_hf_int", "cderi.h5", True, self.args.keep_cderi)
        mydf = None

        if 'gf2' in self.args.finite_size_kind or 'gw' in self.args.finite_size_kind or 'gw_s' in self.args.finite_size_kind:
            self.compute_twobody_finitesize_correction()
            return

        mydf = comm.construct_gdf(self.args, self.cell, self.kmesh)
        # Use Ewald for divergence treatment
        mydf.exxdiv = 'ewald'
        import importlib.util as iu
        new_pyscf = iu.find_spec('pyscf.pbc.df.gdf_builder') is not None
        if new_pyscf :
            import pyscf.pbc.df.gdf_builder as gdf
            weighted_coulG_old = gdf._CCGDFBuilder.weighted_coulG
            gdf._CCGDFBuilder.weighted_coulG = int_utils.weighted_coulG_ewald
        else:
            from pyscf.pbc import df as gdf
            weighted_coulG_old = gdf.GDF.weighted_coulG
            gdf.GDF.weighted_coulG = int_utils.weighted_coulG_ewald
    
        #kij_conj, kij_trans, kpair_irre_list, kptij_idx, num_kpair_stored = 
        int_utils.compute_integrals(self.args, self.cell, mydf, self.kmesh, nao, X_k, "df_int", "cderi_ewald.h5", True, self.args.keep_cderi)
        if new_pyscf :
            gdf._CCGDFBuilder.weighted_coulG = weighted_coulG_old
        else:
            gdf.GDF.weighted_coulG = weighted_coulG_old

    def evaluate_high_symmetry_path(self):
        if self.args.print_high_symmetry_points:
            comm.print_high_symmetry_points(self.cell, self.args)
            return
        if self.args.high_symmetry_path is None:
            raise RuntimeError("Please specify high-symmetry path")
        if self.args.high_symmetry_path is not None:
            try:
                comm.check_high_symmetry_path(self.cell, self.args)
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
        inp_data["high_symm_path/r_mesh"] = comm.construct_rmesh(self.args.nk, self.args.nk, self.args.nk)
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

        j3c, kptij_lst, j2c_sqrt, uniq_kpts = gdf_S.make_j3c(mydf, self.cell, j2c_sqrt=True, exx=False)
        
        ''' Transformation matrix from auxiliary basis to plane-wave '''
        AqQ, q_reduced, q_scaled_reduced = gdf_S.transformation_PW_to_auxbasis(mydf, self.cell, j2c_sqrt, uniq_kpts)
        
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
    '''
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
        nk       = self.args.nk ** 3
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
        hf_dm = mf.make_rdm1().astype(dtype=np.complex128)
        S     = mf.get_ovlp().astype(dtype=np.complex128)
        T     = mf.get_hcore().astype(dtype=np.complex128)
        if self.args.xc is not None:
            vhf = mf.get_veff().astype(dtype=np.complex128)
        else:
            vhf = mf.get_veff(hf_dm).astype(dtype=np.complex128)
        F = mf.get_fock(T,S,vhf,hf_dm).astype(dtype=np.complex128)


        F = F.reshape((self.args.ns, self.args.nk, nso, nso))
        hf_dm = hf_dm.reshape((self.args.ns, self.args.nk, nso, nso))
        S = S.reshape((self.args.nk, nso, nso))
        T = T.reshape((self.args.nk, nso, nso))
    
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
