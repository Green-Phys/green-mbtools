from functools import reduce
import numpy as np
import scipy.linalg as LA

"""
Orthogonalization utilities
"""


def canonical_matrices(S, thr=1e-7, type='f'):
    '''Löwdin's canonical orthogonalization'''
    # Form vectors for normalized overlap matrix
    Sval, Svec = LA.eigh(S)
    X = Svec[:, Sval >= thr] / np.sqrt(Sval[Sval >= thr])
    if type == 'f':
        return X
    elif type == 'g':
        X = LA.pinv(X)
        return X.T.conj()
    else:
        raise ValueError(
            "Invalid transofrmation type. Only 'f'/'g' type for \
            Fock/Green's function only."
        )


def canonical_orth(H, S, thr=1e-7, type='f'):
    '''Löwdin's canonical orthogonalization'''
    # if type != 'f':
    #     raise ValueError(
    #         "Invalid transformation type. "
    #         "Only 'f' type for Fock is supported currently."
    #     )
    print("Canonical orthogonalization with threshold = {}.".format(thr))
    ns = S.shape[0]
    nk = S.shape[1]
    nao = S.shape[2]
    original_shape = H.shape
    H = H.reshape(-1, ns, nk, nao, nao)
    H_orth = np.zeros(H.shape, dtype=H.dtype)
    for s in range(ns):
        for ik in range(nk):
            cond = np.linalg.cond(S[s, ik])
            if cond > 1e7:
                print(
                    "Warning: Condition number = {} is larger than"
                    " 1e7.".format(cond)
                )
            X = canonical_matrices(S[s, ik], thr, type)
            # nbands <= nao due to linear dependency
            nbands = X.shape[1]
            for d in range(H.shape[0]):
                H_orth[d, s, ik, :nbands, :nbands] = reduce(
                    np.dot, (X.T.conj(), H[d, s, ik], X)
                )
    H_orth = H_orth.reshape(original_shape)

    return H_orth


def sao_orth(X, S, type=None):
    """Symmetrized AO basis.
    """
    if not(type == 'g' or type == 'f'):
        raise ValueError(
            "Valid transformation types are 'g' for density, 'f' for Fock"
        )
    ns = S.shape[0]
    nk = S.shape[1]
    nao = S.shape[2]
    original_shape = X.shape
    X = X.reshape(-1, ns, nk, nao, nao)
    X_orth = np.zeros(X.shape, dtype=X.dtype)
    S12 = np.zeros(S.shape, dtype=S.dtype)
    for s in range(ns):
        for ik in range(nk):
            cond = np.linalg.cond(S[s, ik])
            if cond > 1e7:
                print(
                    "Warning: Condition number, {}, is larger than 1e7. "
                    "Possible numerical instability could appear. \n"
                    "Consider to use the canonical orthogonalization "
                    "instead.".format(cond)
                )
            if type == 'g':
                S12[s, ik] = LA.sqrtm(S[s, ik])
            elif type == 'f':
                S12[s, ik] = np.linalg.inv(LA.sqrtm(S[s, ik]))

    for d in range(X.shape[0]):
        for s in range(ns):
            for ik in range(nk):
                X_orth[d, s, ik] = reduce(
                    np.dot, (S12[s, ik], X[d, s, ik], S12[s, ik])
                )
    X_orth = X_orth.reshape(original_shape)
    return X_orth


def SAO_matrices(S):
    nk = S.shape[0]
    X_k = []
    X_inv_k = []

    for ik in range(nk):
        Sk = S[ik]
        x_pinv = LA.sqrtm(Sk)
        x = LA.inv(x_pinv)

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
    Z_X = np.zeros(Z.shape, dtype=complex)
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

    Z_restore = X_inv.conj().T @ Z_X @ X_inv
    if not np.allclose(Z, Z_restore):
        error = "Orthogonal transformation failed. "\
            "Max difference between origin and restored quantity "\
            "is {}".format(np.max(Z - Z_restore))
        raise RuntimeError(error)
    return Z_X
