import numpy as np
import sys
import scipy.linalg as LA
import h5py
import time
from functools import reduce
from threading import Thread

from . import ortho_utils

class seet_init:
    '''
    SEET pre-processing class
    computes proper orthogonal transformation and projection matricies

    Attributes
    ----------
    args : map
        argument's map
    '''

    def __init__(self, args):
        '''
        Initialize SEET pre-processing class

        Parameters
        ----------
        args: map
            argument's map
        '''
        self.args = args


    def to_full_bz(self, X, conj_list, ir_list, bz_index, k_ind):
        '''
        Project moment-dependent quantity onto full Brillouin zone
        '''
        index_list = np.zeros(bz_index.shape, dtype=int)
        for i,irn in enumerate(ir_list):
            index_list[irn] = i
        old_shape = X.shape
        new_shape = np.copy(old_shape)
        new_shape[k_ind] = conj_list.shape[0]
        Y = np.zeros(new_shape, dtype=X.dtype)
        for ik, kk in enumerate(bz_index):
            k = index_list[kk]
            if k_ind == 0:
                Y[ik, ::] = X[k, ::].conj() if conj_list[ik] else X[k, ::]
            elif k_ind == 1:
                Y[:,ik, ::] = X[:,k, ::].conj() if conj_list[ik] else X[:,k, ::]
            elif k_ind == 2:
                Y[:,:,ik, ::] = X[:,:,k, ::].conj() if conj_list[ik] else X[:,:,k, ::]
        return Y

    def get_input_data(self):
        '''
        Read weak-coupling solution
        '''
        inp_data = h5py.File(self.args.input_file, "r")

        gf2_inp_data = h5py.File(self.args.gf2_input_file, "r")
        last_gf2_iter = gf2_inp_data["iter"][()]
        S1     = gf2_inp_data["iter{}/Sigma1".format(last_gf2_iter)][()].view(np.complex128)
        if len(S1.shape) == 5:
            S1     = S1.reshape(S1.shape[:-1])

        gf2_inp_data = h5py.File(self.args.gf2_input_file, "r")
        last_gf2_iter = gf2_inp_data["iter"][()]
        G_tau = gf2_inp_data["iter{}/G_tau/data".format(last_gf2_iter)][()].view(np.complex128)
        if len(G_tau.shape) == 6:
            dm    = - G_tau[G_tau.shape[0]-1,:].reshape(G_tau.shape[1:-1])
        else :
            dm    = - G_tau[G_tau.shape[0]-1,:].reshape(G_tau.shape[1:])
        dm_s  = np.copy(dm)
        dm    = (dm[0] + dm[1])*0.5
        gf2_inp_data.close()

        e_nuc           = inp_data["HF/Energy_nuc"][()]
        nk              = inp_data["HF/nk"][()]
        kmesh           = inp_data["grid/k_mesh"][()]
        kmesh_sc        = inp_data["grid/k_mesh_scaled"][()]
        reduced_mesh    = inp_data["grid/k_mesh"][()]
        reduced_mesh_sc = inp_data["grid/k_mesh_scaled"][()]
        if "grid/weight" in inp_data:
            weight          = inp_data["grid/weight"][()]
            conj_list       = inp_data["grid/conj_list"][()]
            ir_list = inp_data["grid/ir_list"][()]
            bz_index = inp_data["grid/index"][()]
        else:
            weight = [1] * kmesh.shape[0]
            conj_list = [0] * kmesh.shape[0]
            ir_list = range(kmesh.shape[0])
            bz_index = range(kmesh.shape[0])
        S     = inp_data["HF/S-k"][()].view(np.complex128)
        S     = S.reshape(S.shape[:-1])
        T     = inp_data["HF/H-k"][()].view(np.complex128)
        T     = T.reshape(T.shape[:-1])
        if self.args.from_ibz:
            # T = to_full_bz(T, conj_list, ir_list, bz_index, 1)
            S1 = self.to_full_bz(S1, conj_list, ir_list, bz_index, 1)
            dm_s = self.to_full_bz(dm_s, conj_list, ir_list, bz_index, 1)
            dm = self.to_full_bz(dm, conj_list, ir_list, bz_index, 0)

        inp_data.close()
        F = S1 + T

        return F, S, T, dm, dm_s, e_nuc, nk, kmesh, kmesh_sc, reduced_mesh, reduced_mesh_sc, weight, conj_list, ir_list, bz_index
