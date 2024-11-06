import numpy as np
import scipy.linalg as LA
from functools import reduce

def wrap_k(k):
    while k < 0 :
        k = 1 + k
    while (k - 9.9999999999e-1) > 0.0 :
        k = k - 1
    return k

def transform_per_k(Z, X, X_inv):
    '''
    Transform Z into X basis
    Z_X = X^* Z X
    :param Z: Object to be transformed
    :param X: Transformation matrix
    :param X_inv: Inverse transformation matrix
    :return: Z in new basis
    '''
    #Z_X = np.zeros(Z.shape, dtype=np.complex128)
    #Z_X = np.einsum('ij,jk...,kl->il...', X, Z, X.T.conj())
    Z_X = np.dot(X.conj().T, np.dot(Z, X))

    Z_restore = np.dot(X_inv.conj().T, np.dot(Z_X, X_inv))
    if not np.allclose(Z, Z_restore) :
        error = "Orthogonal transformation failed. Max difference between origin and restored quantity is {}".format(np.max(Z - Z_restore))
        raise RuntimeError(error)
    return Z_X

def transform(Z, X, X_inv):
    '''
    Transform Z into X basis
    Z_X = X^* Z X
    :param Z: Object to be transformed
    :param X: Transformation matrix
    :param X_inv: Inverse transformation matrix
    :return: Z in new basis
    '''
    Z_X = np.zeros(Z.shape, dtype=np.complex128)
    for ik in range(Z.shape[0]):
        Z_X[ik] = transform_per_k(Z[ik,:], X[ik], X_inv[ik])

    return Z_X

def normalize(Sv):
    Svo = np.zeros(Sv.shape, dtype=Sv.dtype)
    for iv, v in enumerate(Sv.T):
        ss = np.sign(v[0])*(-1)**iv
        Svo[:,iv] = ss*v.T
    return Svo

def avg(kpts, X):
    '''
    Average momentum dependent quantity 
    '''
    xxx = (X[kpts[0]] + X[kpts[1 % len(kpts)]].conj())/2
    for ikk, kk in enumerate(kpts):
        X[kk] = (xxx if (ikk % 2) == 0 else xxx.conj())

def symmetrical_orbitals(S, dm, F_up, F_dn, T_up, T_dn, kmesh, Debug_print=False):
    X_k = []
    X_inv_k = []
    pairing = {}
    unique_kpts = []
    # generate list of unique k-points by applying an inversion symmetry
    for ik, k in enumerate(kmesh):
        for ikp, kp in enumerate(kmesh):
            minus_k = [wrap_k(-kk) for kk in k]
            if np.allclose(minus_k, kp) and ikp not in pairing.keys():
                pairing[ikp] = ik
                unique_kpts.append(ik)
        if ik not in pairing.keys():
            pairing[ik] = ik
    # find the list of k-points that are the same by inversion symmetry and apply the k-symmetrization
    for ik in unique_kpts:
        kpts = []
        for ikp in pairing.keys():
            if pairing[ikp] == ik:
                kpts.append(ikp)
        avg(kpts, dm)
        avg(kpts, F_up)
        avg(kpts, F_dn)
        avg(kpts, T_up)
        avg(kpts, T_dn)

    dmmm = []

    # Compute transformation matrices
    for ik, k in enumerate(kmesh):
        # check whether matrices should be real by inversion symmetry
        if np.allclose(k, [wrap_k(-kk) for kk in k]):
            if Debug_print:
                print(k)
            dmk = dm[ik].real
            Sk = S[ik].real
        else:
            dmk = dm[ik]
            Sk = S[ik]


        x_pinv = LA.sqrtm(Sk)
        x = LA.inv(x_pinv)


        n_ortho, n_nonortho = x.shape

        if Debug_print:
            print("S", np.allclose(np.dot(x_pinv, x_pinv.conj().T), S[ik]))
            print("X X^-1", np.allclose(np.dot(x, x_pinv), np.eye(x.shape[0])))
            print("X* X^-1*", np.allclose(np.dot(x.conj().T, x_pinv.conj().T), np.eye(x.shape[0])))

        # check that direct and inverse symmetries are consistent
        assert(np.allclose(np.dot(x, x_pinv), np.eye(x.shape[0])))

        # store transformation basis for current k-point
        xx_inv = x_pinv.conj().T.astype(np.complex128)
        xx = x.conj().T.astype(np.complex128)
        X_inv_k.append(xx_inv.copy())
        X_k.append(xx.copy())

        minus_k = [wrap_k(-kk) for kk in k]

        for ikk, kk in enumerate(kmesh):
            if np.allclose(kk, minus_k) and ikk < ik:
                # X_inv_k.append(np.copy(X_inv_k[ikk]))
                # X_k.append(np.copy(X_k[ikk]))
                if Debug_print:
                    print("============== reduced point ================")
                    print("S", np.allclose(S[ikk].conj(), S[ik]))
                    print("Sev", np.allclose(LA.eigvalsh(S[ikk]), LA.eigvalsh(S[ik])))
                    # print LA.eigvalsh(S[ikk]), "\n\n=====\n\n", LA.eigvalsh(S[ik])
                    print("S", np.allclose(np.dot(X_inv_k[ikk].T, X_inv_k[ikk].conj()), S[ik]))
                    print("S2", np.allclose(X_inv_k[ikk], xx_inv.conj().T.conj()))
                assert(np.allclose(np.dot(X_inv_k[ik].conj().T, X_inv_k[ik]), S[ik]))
                assert(np.allclose(np.dot(X_inv_k[ikk].T, X_inv_k[ikk].conj()), S[ik]))
                X_inv_k[ik] = np.copy(X_inv_k[ikk].conj())
                X_k[ik] = np.copy(X_k[ikk].conj())
                continue

    # save density matrix eigen values distribution per each k-point
    np.savetxt("dm_k".format(ik), np.array(dmmm).T)
    np.savetxt("dm_band_k".format(ik), np.array(dmmm))
    return X_inv_k, X_k

def canonical_orbitals(S, dm, F_up, F_dn, T_up, T_dn, kmesh, Debug_print=False):
    X_k = []
    X_inv_k = []
    pairing = {}
    unique_kpts = []
    # generate list of unique k-points by applying an inversion symmetry
    for ik, k in enumerate(kmesh):
        for ikp, kp in enumerate(kmesh):
            minus_k = [wrap_k(-kk) for kk in k]
            if np.allclose(minus_k, kp) and ikp not in pairing.keys():
                pairing[ikp] = ik
                unique_kpts.append(ik)
        if ik not in pairing.keys():
            pairing[ik] = ik
    # find the list of k-points that are the same by inversion symmetry and apply the k-symmetrization
    for ik in unique_kpts:
        kpts = []
        for ikp in pairing.keys():
            if pairing[ikp] == ik:
                kpts.append(ikp)
        avg(kpts, dm)
        #avg(kpts, S)
        avg(kpts, F_up)
        avg(kpts, F_dn)
        avg(kpts, T_up)
        avg(kpts, T_dn)

    dmmm = []

    # Compute transformation matrices
    for ik, k in enumerate(kmesh):
        # check whether matrices should be real by inversion symmetry
        if np.allclose(k, [wrap_k(-kk) for kk in k]):
            if Debug_print:
                print(k)
            dmk = dm[ik].real
            Sk = S[ik].real
        else:
            dmk = dm[ik]
            Sk = S[ik]


        s_ev, s_eb = np.linalg.eigh(Sk)

        # Remove all eigenvalues < threshold
        istart = s_ev.searchsorted(1e-9)
        s_sqrtev = np.sqrt(s_ev[istart:])

        # Moore-Penrose pseudoinverse of X:  (X^+ * X)^(-1) * X^+
        # TODO: use least squares instead
        x_pinv = s_eb[:, istart:] * s_sqrtev
        x = (s_eb[:, istart:].conj() * 1 / s_sqrtev).T
        x = LA.inv(x_pinv)

        n_ortho, n_nonortho = x.shape
        if Debug_print:
            print("S", np.allclose(np.dot(x_pinv, x_pinv.conj().T), S[ik]))
            print("X X^-1", np.allclose(np.dot(x, x_pinv), np.eye(x.shape[0])))
            print("X* X^-1*", np.allclose(np.dot(x.conj().T, x_pinv.conj().T), np.eye(x.shape[0])))

        # check that direct and inverse symmetries are consistent
        assert(np.allclose(np.dot(x, x_pinv), np.eye(x.shape[0])))

        # store transformation basis for current k-point
        xx_inv = x_pinv.conj().T
        xx = x.conj().T
        X_inv_k.append(xx_inv.copy())
        X_k.append(xx.copy())

        minus_k = [wrap_k(-kk) for kk in k]
        for ikk, kk in enumerate(kmesh):
            if np.allclose(kk, minus_k) and ikk < ik:
                # X_inv_k.append(np.copy(X_inv_k[ikk]))
                # X_k.append(np.copy(X_k[ikk]))
                if Debug_print:
                    print("============== reduced point ================")
                    print("S", np.allclose(S[ikk].conj(), S[ik]))
                    print("Sev", np.allclose(LA.eigvalsh(S[ikk]), LA.eigvalsh(S[ik])))
                    # print LA.eigvalsh(S[ikk]), "\n\n=====\n\n", LA.eigvalsh(S[ik])
                    print("S", np.allclose(np.dot(X_inv_k[ikk].T, X_inv_k[ikk].conj()), S[ik]))
                    print("S2", np.allclose(X_inv_k[ikk], xx_inv.conj().T.conj()))
                assert(np.allclose(np.dot(X_inv_k[ik].conj().T, X_inv_k[ik]), S[ik]))
                assert(np.allclose(np.dot(X_inv_k[ikk].T, X_inv_k[ikk].conj()), S[ik]))
                X_inv_k[ik] = np.copy(X_inv_k[ikk].conj())
                X_k[ik] = np.copy(X_k[ikk].conj())
                continue

    # save density matrix eigen values distribution per each k-point
    np.savetxt("dm_k".format(ik), np.array(dmmm).T)
    np.savetxt("dm_band_k".format(ik), np.array(dmmm))
    return X_inv_k, X_k

def natural_orbitals(S, dm, F_up, F_dn, T_up, T_dn, kmesh, Debug_print=False):
    '''
    Compute transformation matricies for natural orbtial basis from spin-averaged density matrix
    ---
    
    '''
    X_k = []
    X_inv_k = []
    pairing = {}
    unique_kpts = []
    # generate list of unique k-points by applying an inversion symmetry
    for ik, k in enumerate(kmesh):
        for ikp, kp in enumerate(kmesh):
            minus_k = [wrap_k(-kk) for kk in k]
            if np.allclose(minus_k, kp) and ikp not in pairing.keys():
                pairing[ikp] = ik
                unique_kpts.append(ik)
        if ik not in pairing.keys():
            pairing[ik] = ik
    # find the list of k-points that are the same by inversion symmetry and apply the k-symmetrization
    for ik in unique_kpts:
        kpts = []
        for ikp in pairing.keys():
            if pairing[ikp] == ik:
                kpts.append(ikp)
        avg(kpts, dm)
        avg(kpts, F_up)
        avg(kpts, F_dn)
        avg(kpts, T_up)
        avg(kpts, T_dn)

    dmmm = []

    # Compute transformation matrices
    for ik, k in enumerate(kmesh):
        # check whether matrices should be real by inversion symmetry
        if np.allclose(k, [wrap_k(-kk) for kk in k]):
            dmk = dm[ik].real
            Sk = S[ik].real
        else:
            dmk = dm[ik]
            Sk = S[ik]

        # compute natural orbital basis
        Sd, Sv = LA.eigh(dmk,np.linalg.inv(Sk))
        Sv = Sv.conj().T.astype(np.complex128)
        Svi = LA.inv(Sv)

        # Check that transformations for the inversion symmetries are consistent
        if ik != pairing[ik] and not np.allclose(reduce(np.dot, (Sv, dm[ik], Sv.conj().T)), reduce(np.dot, (Sv.conj(), dm[pairing[ik]], Sv.T)),
                          atol=1e-7):
            print(False, np.max(np.abs(reduce(np.dot, (Sv, dm[ik], Sv.conj().T)) - reduce(np.dot, (Sv.conj(), dm[pairing[ik]], Sv.T))) ))
            print(ik, k, pairing[ik], kmesh[pairing[ik]])
            print("A", reduce(np.dot, (Sv, dm[ik], Sv.conj().T)))
            print("B", reduce(np.dot, (Sv.conj(), dm[pairing[ik]], Sv.T)))
        else:
            pass

        # save density matrix eigenvalues for future analysis
        dmmm.append(np.diag(reduce(np.dot, (Sv, dm[ik], Sv.conj().T))).real)

        # here we assume that the direct transformation is for quantities like Self-energy and Fock matrix
        # and the inverse transformation if for quantities like density matrix and Green's function
        x_pinv = Sv
        x  = Svi

        # check that direct and inverse symmetries are consistent
        assert(np.allclose(np.dot(x, x_pinv), np.eye(x.shape[0])))

        dmkd = transform_per_k(dmk, x_pinv.conj().T, x.conj().T)
        assert(np.allclose(dmkd, np.diag(np.diag(dmkd))))

        # store transformation basis for current k-point
        X_inv_k.append(x_pinv.copy())
        X_k.append(x.copy())

    # save density matrix eigen values distribution per each k-point
    np.savetxt("dm_k".format(ik), np.array(dmmm).T)
    np.savetxt("dm_band_k".format(ik), np.array(dmmm))
    return X_inv_k, X_k
