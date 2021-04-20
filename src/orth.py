from functools import reduce
import numpy as np
import scipy.linalg as LA

'''
Orthogonalization utilities
'''

def sao_orth(X, S, type=None):
    if type != 'g' and type != 'f':
        raise ValueError("Valid transformation types are 'g' for density, 'f' for Fock")
    ns  = S.shape[0]
    nk  = S.shape[1]
    nao = S.shape[2]
    original_shape = X.shape
    X = X.reshape(-1, ns, nk, nao, nao)
    X_orth = np.zeros(X.shape, dtype=X.dtype)
    for d in range(X.shape[0]):
        for s in range(ns):
            for ik in range(nk):
                S12 = LA.sqrtm(S[s,ik])
                if type == 'g':
                    X_orth[d,s,ik] = reduce(np.dot, (S12, X[d,s,ik], S12))
                elif type == 'f':
                    S12_inv = np.linalg.inv(S12)
                    X_orth[d,s,ik] = reduce(np.dot, (S12_inv, X[d,s,ik], S12_inv))
    X_orth = X_orth.reshape(original_shape)
    return X_orth


def SAO_matrices(S):
    nk = S.shape[0]
    nao = S.shape[1]
    X_k = []
    X_inv_k = []

    for ik in range(nk):
        Sk = S[ik]
        x_pinv = LA.sqrtm(Sk)
        x = LA.inv(x_pinv)
        n_ortho, n_nonortho = x.shape

        # store transformation basis for current k-point
        xx_inv = x_pinv.conj().T
        xx = x.conj().T
        X_inv_k.append(xx_inv.copy())
        X_k.append(xx.copy())

        # check that direct and inverse symmetries are consistent
        assert (np.allclose(np.dot(x, x_pinv), np.eye(x.shape[0])))
        assert (np.allclose(np.dot(X_inv_k[ik].conj().T, X_inv_k[ik]), S[ik]))

    return X_inv_k, X_k


def transform(Z, X, X_inv):
    '''
    Transform Z into X basis
    Z_X = X^* Z X
    :param Z: Object to be transformed
    :param X: Transformation matrix
    :param X_inv: Inverse transformation matrix
    :return: Z in new basis
    '''
    Z_X = np.zeros(Z.shape, dtype=np.complex)
    for ik in range(Z.shape[0]):
        Z_X[ik] = transform_per_k(Z[ik, :], X[ik], X_inv[ik])

    return Z_X


def transform_per_k(Z, X, X_inv):
    '''
    Transform Z into X basis
    Z_X = X^* Z X
    :param Z: Object to be transformed
    :param X: Transformation matrix
    :param X_inv: Inverse transformation matrix
    :return: Z in new basis
    '''
    # Z_X = np.einsum('ij,jk...,kl->il...', X, Z, X.T.conj())
    Z_X = np.dot(X.conj().T, np.dot(Z, X))

    Z_restore = np.dot(X_inv.conj().T, np.dot(Z_X, X_inv))
    if not np.allclose(Z, Z_restore):
        error = "Orthogonal transformation failed. Max difference between origin and restored quantity is {}".format(
            np.max(Z - Z_restore))
        raise RuntimeError(error)
    return Z_X
