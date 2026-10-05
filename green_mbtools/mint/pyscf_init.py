import os

import h5py
import logging
import numpy as np
from pyscf.df import addons
from pyscf.gto.ecp import core_configuration
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

    def init_core(self):
        '''
        This method computes the number of core orbitals for this system.
        It also returns a reordering list that allows to go from the pyscf AO ordering to a core + valence ordering.
        The number of core orbitals for different types of atoms can be given as an argument (see add_common_params).
        '''

        increasing_ordering = ['1s', '2s',  '2px', '2py', '2pz', '3s', '3px', '3py', '3pz', '4s', '3dxy', '3dyz', '3dz^2', '3dxz', '3dx2-y2', '4px', '4py', '4pz', '5s', '4dxy', '4dyz', '4dz^2', '4dxz', '4dx2-y2', '5px', '5py', '5pz', '6s', '4f-3', '4f-2', '4f-1', '4f+0', '4f+1', '4f+2', '4f+3', '5dxy', '5dyz', '5dz^2', '5dxz', '5dx2-y2', '6px', '6py', '6pz', '7s', '5f-3', '5f-2', '5f-1', '5f+0', '5f+1', '5f+2', '5f+3', '6dxy', '6dyz', '6dz^2', '6dxz', '6dx2-y2',  '7px', '7py', '7pz']
        default_nb_core_elec = {
            # Period 1
            "H": 0, "He": 0,
            # Period 2 (core = He 2)
            "Li": 2, "Be": 2, "B": 2, "C": 2, "N": 2, "O": 2, "F": 2, "Ne": 2,
            # Period 3 (core = Ne 10)
            "Na": 10, "Mg": 10, "Al": 10, "Si": 10, "P": 10, "S": 10, "Cl": 10, "Ar": 10,
            # Period 4 (core = Ar 18)
            "K": 18, "Ca": 18, "Sc": 18, "Ti": 18, "V": 18, "Cr": 18, "Mn": 18,"Fe": 18, "Co": 18, "Ni": 18, "Cu": 18, "Zn": 18,"Ga": 18, "Ge": 18, "As": 18, "Se": 18, "Br": 18, "Kr": 18,
            # Period 5 (core = Kr 36)
            "Rb": 36, "Sr": 36, "Y": 36, "Zr": 36, "Nb": 36, "Mo": 36, "Tc": 36,"Ru": 36, "Rh": 36, "Pd": 36, "Ag": 36, "Cd": 36,"In": 36, "Sn": 36, "Sb": 36, "Te": 36, "I": 36, "Xe": 36,
            # Period 6 (core = Xe 54)
            "Cs": 54, "Ba": 54,"La": 54, "Ce": 54, "Pr": 54, "Nd": 54, "Pm": 54, "Sm": 54, "Eu": 54,"Gd": 54, "Tb": 54, "Dy": 54, "Ho": 54, "Er": 54, "Tm": 54, "Yb": 54, "Lu": 54,"Hf": 54, "Ta": 54, "W": 54, "Re": 54, "Os": 54, "Ir": 54, "Pt": 54,"Au": 54, "Hg": 54,"Tl": 54, "Pb": 54, "Bi": 54, "Po": 54, "At": 54, "Rn": 54,
            # Period 7 (core = Rn 86)
            "Fr": 86, "Ra": 86,"Ac": 86, "Th": 86, "Pa": 86, "U": 86, "Np": 86, "Pu": 86, "Am": 86,"Cm": 86, "Bk": 86, "Cf": 86, "Es": 86, "Fm": 86, "Md": 86, "No": 86, "Lr": 86,"Rf": 86, "Db": 86, "Sg": 86, "Bh": 86, "Hs": 86, "Mt": 86, "Ds": 86,"Rg": 86, "Cn": 86,"Nh": 86, "Fl": 86, "Mc": 86, "Lv": 86, "Ts": 86, "Og": 86,
}

        # Read input to change the default size of core
        if self.args.nb_core_elec is not None:
            for atom, n_elec_core in self.args.nb_core_elec:
                default_nb_core_elec[atom] = n_elec_core

        # AO ranges per atom (valid for spherical and cartesian bases) and AO labels as
        # (atom_id, symbol, "nl", "m") tuples, e.g. (0, "C", "2p", "x") -> "2px".
        aoslices = self.cell.aoslice_by_atom()
        ao_labels = ["".join(lab[2:]) for lab in self.cell.ao_labels(fmt=False)]
        
        list_core_idx = []
        list_val_idx = []
        for atm_idx in range(self.cell.natm):
            symbol = self.cell.atom_symbol(atm_idx)
            ao_start, ao_end = int(aoslices[atm_idx, 2]), int(aoslices[atm_idx, 3])
            atom_labels = ao_labels[ao_start:ao_end]

            if symbol not in default_nb_core_elec:
                raise ValueError(
                    f"init_core: no default number of core electrons for atom '{symbol}' (atom {atm_idx}). "
                    f"Set it explicitly with --nb_core_elec {symbol},<n> (use 0 for ghost atoms)."
                )
            size_core = default_nb_core_elec[symbol]
            if size_core % 2 != 0 or size_core // 2 > len(increasing_ordering):
                raise ValueError(
                    f"init_core: invalid number of core electrons {size_core} for atom '{symbol}'; "
                    f"it must be even and at most {2 * len(increasing_ordering)}."
                )

            # Electrons already removed by an ECP / pseudopotential. pyscf labels the remaining
            # shells past that core (e.g. Si with a Ne-core pseudo starts at 3s), so the shells
            # it removed must be skipped here too.
            n_ecp = self.cell.atom_nelec_core(atm_idx)
            if size_core <= n_ecp:
                core_orbs = []
            else:
                removed_shells = set()
                if n_ecp > 0:
                    ecp_conf = core_configuration(n_ecp, atom_symbol=self.cell.atom_pure_symbol(atm_idx))
                    for l, nshell in enumerate(ecp_conf):
                        removed_shells |= {f"{n}{'spdf'[l]}" for n in range(l + 1, l + 1 + nshell)}
                core_orbs = [o for o in increasing_ordering[:size_core // 2] if o[:2] not in removed_shells]
                if 2 * len(core_orbs) != size_core - n_ecp:
                    raise ValueError(
                        f"init_core: the {n_ecp}-electron ECP of atom '{symbol}' is not compatible with a "
                        f"{size_core}-electron core. Set the core explicitly with --nb_core_elec {symbol},<n>."
                    )

            # Positions of the core orbitals inside the AO list
            core_positions = []
            for orb in core_orbs:
                if orb not in atom_labels:
                    raise ValueError(
                        f"init_core: core orbital '{orb}' of atom '{symbol}' (atom {atm_idx}) is not in its basis "
                        f"(the basis may be cartesian). Reduce its core with --nb_core_elec {symbol},<n>."
                    )
                core_positions.append(ao_start + atom_labels.index(orb))
            # Positions of the valence orbitals inside the AO list
            val_positions = [i for i in range(ao_start, ao_end) if i not in core_positions]

            list_core_idx += core_positions
            list_val_idx += val_positions

        self.ncore = len(list_core_idx)
        nao = self.cell.nao_nr()

        if self.args.x2c == 2:
            # Spinor (2-component) matrices have dimension 2*nao; the core reordering is only
            # defined for nao-dimensional (spin-free) matrices so far.
            if self.ncore > 0:
                raise NotImplementedError(
                    "init_core: frozen core is not implemented for --x2c 2 yet. "
                    "Use --nb_core_elec <element>,0 for every element to disable it."
                )
            self.orb_reordering = list(range(2 * nao))
        elif self.args.orth in ("mo", "natural", "fno"):
            # Orbital bases sorted by energy/occupation: the core orbitals already come first,
            # so no reordering is needed.
            self.orb_reordering = list(range(nao))
        else:
            self.orb_reordering = list_core_idx + list_val_idx

        logging.info(f"Number of core orbitals: {self.ncore}")
        logging.info(f"Core + valence AO reordering: {self.orb_reordering}")

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
        self.init_core()

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
            # NOTE: if args.orth != "none", we will not be able to transform the k_sym_transform_ao yet.
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

        # Orthogonalization matrix. For X2C (--x2c=2) the spinor S is
        # block-diagonal in spin so Löwdin variants give a block-diagonal
        # X whose AO block is a valid ERI rotation; MO and natural
        # rotations would have non-block-diagonal X in general and are
        # refused.
        if self.args.x2c == 2 and self.args.orth not in ("none", "lowdin", "symmetric_lowdin"):
            raise NotImplementedError(
                "ortho not supported for 2-component / x2c1e calculations "
                "with mode={!r}; allowed modes are 'none', 'lowdin', "
                "'symmetric_lowdin'.".format(self.args.orth)
            )
        # Time reversal ALWAYS on for the X build (independent of --tr_symm):
        # the df-integral pair reduction always folds by k->-k conjugation.
        # NOTE: this TR-always kstruct is used ONLY to build X. The exported
        # symmetry operators (store_kstruct_ops_info below) follow
        # self.kstruct (tr_symm=args.tr_symm); the two agree for tr_symm=true
        # and the TR-conjugation branch is inert for tr_symm=false.
        #
        # Build the symmetry decomposition of self.kmesh itself (NOT the
        # q=k1-k2 difference mesh that build_q_struct produces). orthogonalize
        # indexes S/F with ortho_sym_kstruct.ibz2bz, so ortho_sym_kstruct.kpts must match
        # self.kmesh in order; make_kpts on self.kmesh guarantees this, whereas
        # the difference mesh reorders (Gamma-centered) or, for shifted meshes,
        # is a different set of points entirely.
        ortho_sym_kstruct = libkpts.make_kpts(
            self.cell, self.kmesh,
            space_group_symmetry=False,
            time_reversal_symmetry=True)
        X_k, X_inv_k, S, F, T, hf_dm = comm.orthogonalize(
            mydf, self.args.orth, X_k, X_inv_k, F, T, hf_dm, S,
            sym_kstruct=ortho_sym_kstruct, mycell=self.cell, spinor=self.args.x2c==2)
        # Save data into Green Software package input format.
        comm.save_data(
            self.args, self.cell, mf, self.kmesh, self.ind, self.weight, self.num_ik, self.ir_list, self.conj_list,
            Nk, nk, NQ, F, S, T, hf_dm, tools.pbc.madelung(self.cell, self.kmesh), Zs, last_ao, self.ncore, self.orb_reordering
        )
        # Save symmetry operations info for main and auxiliary unit cells
        comm.store_kstruct_ops_info(self.args, self.cell, self.kmesh, self.kstruct, X_k=X_k, X_inv_k=X_inv_k,)
        comm.store_auxcell_kstruct_ops_info(self.args, auxcell, self.kmesh)
        # Save the AO->orthogonal basis transformation so tooling can move the
        # stored (orthogonalized) quantities back to the AO basis.
        comm.store_orth_transform(self.args, X_k, X_inv_k)

        # Diagnose whether self-consistent quantities obey k-space symmetry.
        if self.args.space_symm or self.args.tr_symm:
            symm_utils.check_kspace_symmetry_breaking(self.args.output_path, ["HF/H-k", "HF/S-k", "HF/Fock-k"])

        # Store density-fitted integrals
        if bool(self.args.df_int) :
            self.compute_df_int(nao, X_k)

    def compute_df_int(self, nao, X_k):
        '''
        Generate density-fitting (DF) three-center Coulomb integrals for correlated methods.

        This routine always produces the mean-field DF integral set written to
        ``args.hf_int_path``. A second, correlated DF integral set written to
        ``args.int_path`` is generated here only for the ``ewald`` finite-size
        correction path.

        1. Mean-field integrals (written to ``args.hf_int_path``):
           Standard DF integrals L^Q_{pq}(k_i, k_j) for all symmetry-
           irreducible k-point pairs, computed with the bare Coulomb kernel.
           These are used in the mean-field and Hartree-Fock steps.

        2. Finite-size correction handling:

           - ``gf2`` / ``gw`` / ``gw_s``: delegates to
             ``compute_twobody_finitesize_correction()``, which uses the
             GF2 Ewald subtraction scheme or the GW plane-wave transformation
             respectively, then returns early. In these branches,
             ``compute_integrals(..., basename=args.int_path, ...)`` is not
             called by this function.

           - ``ewald`` (default): builds a second set of three-center integrals
             with the Ewald Coulomb kernel via ``green_igen.df._make_j3c`` and
             passes them to ``compute_integrals`` as ``cderi_name2``; the
             diagonal pairs in the output are then replaced by the
             Ewald-corrected values and written to ``args.int_path``.

        Parameters
        ----------
        nao : int
            Number of non-relativistic atomic orbitals per k-point.
            Always ``cell.nao_nr()`` regardless of the X2C level, because
            the Coulomb integrals are non-relativistic.
        X_k : list of ndarray
            Per-k-point orthogonalisation matrices X(k). The specific form
            depends on ``args.orth``:

            * ``"lowdin"`` — canonical Löwdin, ``X(k) = Lambda^{-1/2} V†``
              (rectangular when small eigenvalues of S are dropped).
            * ``"symmetric_lowdin"`` — Hermitian Löwdin, ``X(k) = S(k)^{-1/2}``
              (square; treats sub-tol eigenvalues pseudo-inversely).
            * ``"mo"`` — canonical MOs, ``X(k) = C(k)†`` with
              ``X_inv = S(k) @ C(k)``.
            * ``"natural"`` — natural orbitals, ``X(k) = C_NO(k)†`` with
              ``X_inv = S(k) @ C_NO(k)`` and ``C_NO`` the S-orthonormal
              eigenvectors of ``S^{-1/2} dm S^{-1/2}``.

            When orthogonalisation is disabled (``args.orth == "none"``),
            ``X_k`` contains identity transforms for each k-point rather
            than an empty list.
        '''
        # --- Step 1: mean-field integrals (bare Coulomb kernel) --------------
        mydf = comm.construct_gdf(self.args, self.cell, self.kmesh)
        int_utils.compute_integrals(self.args, self.cell, mydf, self.kmesh, nao, X_k, self.args.hf_int_path, "cderi.h5", True, True)
        mydf = None

        # --- Step 2: correlated integrals with finite-size correction --------
        # GF2/GW corrections use a separate code path that handles the
        # correction internally; the plain Ewald correction is handled below.
        if 'gf2' in self.args.finite_size_kind or 'gw' in self.args.finite_size_kind or 'gw_s' in self.args.finite_size_kind:
            self.compute_twobody_finitesize_correction(X_k=X_k)
            if not self.args.keep_cderi:
                os.remove("cderi.h5")
                os.system("sync")
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

        mydf = comm.construct_gdf(self.args, self.cell, self.kmesh)
        mydf.exxdiv = 'ewald'
        auxcell = gggdf.make_modrho_basis(mydf.cell, mydf.auxbasis,
                                          mydf.exp_to_discard)
        kptij_lst = np.asarray([(ki, ki) for ki in self.kmesh])

        # Save → patch → build → restore.
        weighted_coulG_old = gdf.GDF.weighted_coulG
        gdf.GDF.weighted_coulG = int_utils.weighted_coulG_ewald
        gggdf._make_j3c(mydf, self.cell, auxcell, kptij_lst, "cderi_ewald.h5")
        gdf.GDF.weighted_coulG = weighted_coulG_old  # always restore

        # Build correlated integrals; diagonal pairs come from cderi_ewald.h5.
        int_utils.compute_integrals(self.args, self.cell, mydf, self.kmesh, nao, X_k, self.args.int_path, "cderi.h5", True, self.args.keep_cderi, cderi_name2="cderi_ewald.h5")

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

    def compute_twobody_finitesize_correction(self, mydf=None, X_k=None):
        if not os.path.exists(self.args.hf_int_path):
            os.mkdir(self.args.hf_int_path)
        if 'gf2' in self.args.finite_size_kind :
            comm.compute_ewald_correction(
                self.args, self.cell, self.kmesh,
                self.args.hf_int_path + "/df_ewald.h5",
                X_k=X_k,
            )
        if 'gw' in self.args.finite_size_kind :
            # AqQ is a plane-wave ↔ aux-basis map with no AO indices, so it
            # does not need the AO→ortho rotation that the Coulomb integrals
            # require. The mbpt GW correction consumes AqQ together with the
            # already-rotated V on disk.
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
        self.init_core()


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

        # Orthogonalization matrix. For X2C (--x2c=2) the spinor S is
        # block-diagonal in spin so Löwdin variants give a block-diagonal
        # X whose AO block is a valid ERI rotation; MO and natural
        # rotations would have non-block-diagonal X in general and are
        # refused.
        if self.args.x2c == 2 and self.args.orth not in ("none", "lowdin", "symmetric_lowdin"):
            raise NotImplementedError(
                "ortho not supported for 2-component / x2c1e calculations "
                "with mode={!r}; allowed modes are 'none', 'lowdin', "
                "'symmetric_lowdin'.".format(self.args.orth)
            )
        X_k, X_inv_k, S, F, T, hf_dm = comm.orthogonalize(mydf, self.args.orth, X_k, X_inv_k, F, T, hf_dm, S,
                                                          sym_kstruct=self.kstruct, mycell=self.kcell,
                                                          spinor=self.args.x2c==2)
        # Save data into Green Software package input format. Here we set Madelung constant to 0 as there is
        # no long range divergence for molecule
        comm.save_data(self.args, self.kcell, mf, self.kmesh, self.ind, self.weight, self.num_ik, self.ir_list,
                       self.conj_list, Nk, nk, NQ, F, S, T, hf_dm, 0.0, Zs, last_ao, self.ncore, self.orb_reordering)
        comm.store_mol_symmetry_info(self.args, self.kcell, auxcell, self.kmesh)
        # Save the AO->orthogonal basis transformation so tooling can move the
        # stored (orthogonalized) quantities back to the AO basis.
        comm.store_orth_transform(self.args, X_k, X_inv_k)
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
