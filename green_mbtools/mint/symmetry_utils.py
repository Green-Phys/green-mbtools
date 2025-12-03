import numpy as np
from pyscf.pbc.symm.symmetry import _get_phase


def _get_rotation_mat(cell, kpt_scaled_ibz, op, Dmats, ignore_phase=False, tol=1e-10, x2c=False):
    atm_map, phases = _get_phase(cell, op, kpt_scaled_ibz, ignore_phase, tol)

    if not x2c:
        dim = cell.nao_nr() # set dimension to nao_nr
    else:
        raise NotImplementedError("X2C symmetry operations are not implemented yet.")

    mat = np.zeros([dim, dim], dtype=np.complex128)
    aoslice = cell.aoslice_by_atom()
    for iatm in range(cell.natm):
        jatm = atm_map[iatm]
        if iatm != jatm:
            #sanity check
            nao_i = aoslice[iatm][3] - aoslice[iatm][2]
            nao_j = aoslice[jatm][3] - aoslice[jatm][2]
            assert(nao_i == nao_j)
            nshl_i = aoslice[iatm][1] - aoslice[iatm][0]
            nshl_j = aoslice[jatm][1] - aoslice[jatm][0]
            assert(nshl_i == nshl_j)
            for ishl in range(nshl_i):
                shl_i = ishl + aoslice[iatm][0]
                shl_j = ishl + aoslice[jatm][0]
                l_i = cell._bas[shl_i,1]
                l_j = cell._bas[shl_j,1]
                assert(l_i == l_j)
        phase = phases[iatm]
        ao_off_i = aoslice[iatm][2]
        ao_off_j = aoslice[jatm][2]
        shlid_0 = aoslice[iatm][0]
        shlid_1 = aoslice[iatm][1]
        for ishl in range(shlid_0, shlid_1):
            l = cell.bas_angular(ishl)
            Dmat = Dmats[l] * phase
            if not cell.cart:
                nao = 2 * l + 1
            else:
                nao = (l+1) * (l+2) // 2
            nc = cell.bas_nctr(ishl)
            for _ in range(nc):
                mat[ao_off_j:ao_off_j+nao, ao_off_i:ao_off_i+nao] = Dmat
                ao_off_i += nao
                ao_off_j += nao
        assert ao_off_i == aoslice[iatm][3]
        assert ao_off_j == aoslice[jatm][3]
    return mat