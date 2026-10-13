import numpy as np
import h5py
from green_mbtools.version import require_input_version


# Reference input-file version for SEET
REF_INPUT_VERSION = "1.1.0"

class seet_init:
    '''
    SEET pre-processing class.
    Computes proper orthogonal transformation and projection matricies

    Attributes
    ----------
    args : map
        simulation parameters
    '''

    def __init__(self, args):
        '''
        Initialize SEET pre-processing class

        Parameters
        ----------
        args: map
            simulation parameters
        '''
        self.args = args


    def get_input_data(self):
        '''
        Read weak-coupling solution
        '''
        from green_mbtools.pesto import mb

        with h5py.File(self.args.input_file, "r") as inp_data:
            input_version   = inp_data.attrs.get("__green_version__")
            require_input_version(input_version, REF_INPUT_VERSION)
            kmesh           = inp_data["symmetry/k/mesh"][()]
            kmesh_sc        = inp_data["symmetry/k/mesh_scaled"][()]
            S     = inp_data["HF/S-k"][()]
            T     = inp_data["HF/H-k"][()]

            # Sigma1 / G_tau are stored on the space-group irreducible wedge.
            # Expand them onto the full BZ using the AO symmetry transforms
            # (U X U^dagger) together with time-reversal conjugation.
            conj_list = inp_data["symmetry/k/tr_conj"][()]
            ibz2bz    = inp_data["symmetry/k/ibz2bz"][()]
            bz2ibz    = inp_data["symmetry/k/bz2ibz"][()]
            k_sym_ao  = inp_data["symmetry/k/k_sym_transform_ao"][()]

        with h5py.File(self.args.gf2_input_file, "r") as gf2_inp_data:
            last_gf2_iter = gf2_inp_data["iter"][()]
            S1     = gf2_inp_data["iter{}/Sigma1".format(last_gf2_iter)][()].view(np.complex128)
            if len(S1.shape) == 5:
                S1     = S1.reshape(S1.shape[:-1])

            G_tau = gf2_inp_data["iter{}/G_tau/data".format(last_gf2_iter)][()].view(np.complex128)
            if len(G_tau.shape) == 6:
                dm_s  = - G_tau[G_tau.shape[0]-1,:].reshape(G_tau.shape[1:-1])
            else :
                dm_s  = - G_tau[G_tau.shape[0]-1,:].reshape(G_tau.shape[1:])

        S1   = mb.to_full_bz(S1, conj_list, ibz2bz, bz2ibz, 1, k_sym_ao)
        dm_s = mb.to_full_bz(dm_s, conj_list, ibz2bz, bz2ibz, 1, k_sym_ao)
        # Spin-average on the full BZ (after the per-k rotation) so the
        # local density matrix inherits the same symmetry treatment.
        dm   = 0.5 * (dm_s[0] + dm_s[1])
        F = S1 + T

        return F, S, T, dm, dm_s, kmesh, kmesh_sc
