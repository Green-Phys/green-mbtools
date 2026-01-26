import numpy as np
import scipy.linalg as LA
import h5py
from pathlib import Path
from green_mbtools.mint import pyscf_pbc_init
from green_mbtools.mint import common_utils as comm
import os
import shutil


def test_symmetry():
    """
    Docstring for test_symmetry
    """
    tmp_dir = Path(__file__).parent / "tmp"
    ref_file = Path(__file__).parent / "test_data" / "H2_pbc" / "UHF" / "input.h5"

    # ensure a clean slate
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()
    old_cwd = Path.cwd()  # working directory for pytests
    os.chdir(tmp_dir)
    # prepare scratch locations
    output_h5 = tmp_dir / "input.h5"

    # Reference data
    fref = h5py.File(ref_file, "r")
    Fock = fref["HF/Fock-k"][()].view(complex)
    Fock = Fock.reshape(Fock.shape[:-1])
    Sk = fref["HF/S-k"][()].view(complex)
    Sk = Sk.reshape(Sk.shape[:-1])
    Hk = fref["HF/H-k"][()].view(complex)
    Hk = Hk.reshape(Hk.shape[:-1])
    fref.close()

    # build parameters
    params = [
        "--atom", "H -0.25 -0.25 -0.25\nH  0.25  0.25  0.25",
        "--a", "4.0655, 0.0,    0.0\n0.0,    4.0655, 0.0\n0.0,    0.0,    4.0655\n",
        "--basis", "gth-dzvp-molopt-sr", "--pseudo", "gth-pbe",
        "--output_path", str(output_h5),
        "--df_int", "0",
        "--nk", "3", "--grid_only", "true"
    ]

    # run mean‑field generation
    args = comm.init_pbc_params(params=params)
    pyscf_init = pyscf_pbc_init(args)
    pyscf_init.mean_field_input()

    # compare symmetry mesh and data sets
    fout = h5py.File(output_h5, "r")
    group = fout["grid"]
    nk = group["nk"][()]
    ink = group["ink"][()]
    bz_2_ibz_index = group["index"][()]
    kspace_orep = group["kspace_orep"][()]
    kspace_orep_aux = group["kspace_orep_aux"][()]
    fout.close()

    # number of irreducible k-points
    assert ink == 6

    # Check Fock symmetry
    for i in range(nk):
        ibz = bz_2_ibz_index[i]
        Uop = kspace_orep[i]
        # Overlap
        Sk_recon = Uop @ Sk[0, ibz] @ Uop.conj().T
        diff_S = LA.eigvalsh(Sk_recon) - LA.eigvalsh(Sk[0, i])
        assert np.max(np.abs(diff_S)) < 1e-8
        # Hamiltonian
        Hk_recon = Uop @ Hk[0, ibz] @ Uop.conj().T
        diff_H = LA.eigvalsh(Hk_recon, Sk_recon) - LA.eigvalsh(Hk[0, i], Sk[0, i])
        assert np.max(np.abs(diff_H)) < 1e-8
        # Fock
        # NOTE: because Vxc is constructed on real grid, its symmetrization is not perfect.
        # Threfore, the agreement in Fock matrix is usually around 1e-5 to 1e-6
        Fk_recon = Uop @ Fock[0, ibz] @ Uop.conj().T
        diff_F = LA.eigvalsh(Fk_recon, Sk[0, i]) - LA.eigvalsh(Fock[0, i], Sk[0, i])
        assert np.max(np.abs(diff_F)) < 1e-5

    # Check condition number of aux-space transformation operators
    for i in range(nk):
        Uop_aux = kspace_orep_aux[i]
        cond_num = np.linalg.cond(Uop_aux)
        assert cond_num < 1e6  # condition number should be reasonable
