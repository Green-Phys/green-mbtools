from mbanalysis.src.analyt_cont import caratheodory_run
import mbanalysis.caratheodory as carath_exe
import numpy as np
import pytest

#
# Test function for Nevanlinna analytic continuation
#


def test_carath_exe_on_dirac_delta_spectrum(data_path):
    """Test basic functionality of Caratheodory AC for a simple
    1x1 Green's function matrix.
    """
    data_dir = pytest.test_data_dir + '/caratheodory'

    #
    # Test case 1: Dirac delta centered at w = 0
    #

    # Generate imaginary axis data for f(z) = 1 / z
    iw_vals = np.linspace(0.1, 10, 100)
    G_iw = 1 / (1j * iw_vals)
    G_iw = G_iw.reshape((G_iw.shape[0], 1, 1, 1, 1))

    # Parameters for Caratheodory analytic continuation
    outdir = data_dir + '/DiracCarath'
    n_real = 101
    w_min = -0.1
    w_max = 0.1
    eta = 0.01

    freqs, G_w, A_w = caratheodory_run(
        G_iw, iw_vals, outdir=outdir, n_real=n_real,
        w_min=w_min, w_max=w_max, eta=eta
    )

    # Generate referencec data
    f_max = freqs[np.argmax(A_w)]
    G_w_ref = 1 / (freqs + 1j * eta)
    A_w_ref = -np.imag(G_w_ref) / np.pi
    G_w_ref = G_w_ref.reshape(G_w.shape)
    A_w_ref = A_w_ref.reshape(A_w.shape)

    assert f_max == 0
    assert np.linalg.norm(G_w - G_w_ref) < 1e-5
    # Spectral func relation
    assert np.linalg.norm(A_w - A_w_ref) < 1e-5

    #
    # Test case 2: Dirac delta centerd at z0 = 1 + 1e-4j
    #

    # Generate imaginary axis data for f(z) = 1 / (z - z0)
    iw_vals = np.linspace(0.1, 10, 100)
    G_iw = 1 / (1j * iw_vals - 1)
    G_iw = G_iw.reshape((G_iw.shape[0], 1, 1, 1, 1))

    # Parameters for Caratheodory analytic continuation
    outdir = data_dir + '/ShiftDiracCarath'
    n_real = 201
    w_min = -0.5
    w_max = 1.5
    eta = 0.01

    freqs, G_w, A_w = caratheodory_run(
        G_iw, iw_vals, outdir=outdir, n_real=n_real,
        w_min=w_min, w_max=w_max, eta=eta
    )

    # Generate referencec data
    f_max = freqs[np.argmax(A_w)]
    G_w_ref = 1 / (freqs + 1j * eta - 1)
    A_w_ref = -np.imag(G_w_ref) / np.pi
    G_w_ref = G_w_ref.reshape(G_w.shape)
    A_w_ref = A_w_ref.reshape(A_w.shape)

    assert f_max == 1
    assert np.linalg.norm(G_w - G_w_ref) < 1e-5
    assert np.linalg.norm(A_w - A_w_ref) < 1e-5


def test_carath_exe_on_matrix(data_path):
    """Test Caratheodory for complex valued matrices
    For this, we will use Jiani's Hubbard dimer test set.
    """

    # data directory
    data_dir = pytest.test_data_dir
    data_dir += '/caratheodory'

    # Input data is stored in file GF_imag
    input_data = data_dir + '/GF_imag'
    print(input_data)
    nw = 3
    nao = 4

    # Parameters for continuation
    n_real = 21
    w_min = -1
    w_max = 1
    eta = 0.01
    use_custom_grid = 0  # False
    gf_c_fname = data_dir + '/GF_c_test'
    gf_a_fname = data_dir + '/GF_A_test'
    carath_exe.caratheodory(
        input_data, nw, nao, gf_c_fname, gf_a_fname,
        use_custom_grid, 'grid_file', n_real, w_min, w_max, eta
    )

    gf_c_test_data = np.loadtxt(gf_c_fname)
    gf_a_test_data = np.loadtxt(gf_a_fname)
    gf_c_ref_data = np.loadtxt(data_dir + '/GF_c')
    gf_a_ref_data = np.loadtxt(data_dir + '/GF_A')

    freqs = gf_c_test_data[:, 0]
    gf_c_test = np.zeros((n_real, nao, nao), dtype=complex)
    gf_a_test = gf_a_test_data[:, 1]
    gf_c_ref = np.zeros((n_real, nao, nao), dtype=complex)
    gf_a_ref = gf_a_ref_data[:, 1]
    for jw in range(n_real):
        # complex matrix
        xdata = gf_c_test_data[jw, 1:].view(complex)
        gf_c_test[jw, :, :] = xdata.reshape((nao, nao))
        xdata = gf_c_ref_data[jw, 1:].view(complex)
        gf_c_ref[jw, :, :] = xdata.reshape((nao, nao))

    # check that Sigma_w is complex number
    assert freqs[0] == w_min
    assert freqs[-1] == w_max
    assert len(freqs) == n_real
    assert gf_c_test.all() == gf_c_ref.all()
    assert gf_a_test.all() == gf_a_ref.all()


def test_caratheodory_custom_freqs(data_path):
    """Test custom frequency functionality for Caratheodory, where the analytic
    continuation is performed on user specified frequencies.
    """

    # data directory
    data_dir = pytest.test_data_dir + '/caratheodory'

    # Test case: Dirac delta centered at w = 0
    # Generate imaginary axis data for f(z) = 1 / z
    iw_vals = np.linspace(0.1, 10, 100)
    G_iw = 1 / (1j * iw_vals)
    G_iw = G_iw.reshape((G_iw.shape[0], 1, 1, 1, 1))

    # Parameters for Caratheodory analytic continuation
    outdir = data_dir + '/DiracCarath'
    n_real = 101
    w_min = -0.1
    w_max = 0.1
    eta = 0.01
    real_freqs = np.linspace(w_min, w_max, n_real)

    freqs, G_w, A_w = caratheodory_run(
        G_iw, iw_vals, outdir=outdir, custom_freqs=real_freqs, eta=eta
    )

    # check the output freqs against input
    assert np.allclose(freqs, real_freqs, rtol=1e-10)

    # Generate referencec data
    f_max = freqs[np.argmax(A_w)]
    G_w_ref = 1 / (freqs + 1j * eta)
    A_w_ref = -np.imag(G_w_ref) / np.pi
    G_w_ref = G_w_ref.reshape(G_w.shape)
    A_w_ref = A_w_ref.reshape(A_w.shape)

    assert f_max == 0
    assert np.linalg.norm(G_w - G_w_ref) < 1e-5
    # Spectral func relation
    assert np.linalg.norm(A_w - A_w_ref) < 1e-5
