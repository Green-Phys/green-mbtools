import numpy as np
import cvxpy as cp
import baryrat
from scipy.optimize import minimize


def cvx_optimize(poles, GM_matrix, zM):
    """Top level target error function in the ES approach
    to Nevanlinna, i.e.,
        Err (poles) = min_{X} || Gimag (iw) - Gapprox [poles, X] (iw) ||
    Input Arguments:
        poles           :   pole positions
        GM_matrix       :   Exact Matsubara Green's function data in the shape
                            (num_imag, num_orb, num_orb)
        zM              :   Matsubara frequencies
    Returns:
        opt_error       :   Optimal value of Err (poles) for fixed poles
        np_X_vec        :   Numpy array rep of optimal X_vec in the shape
                            (num_poles, num_orb, num_orb)
        np_G_approxx    :   Numpy array rep of approximated optimal Green's
                            function for the specified Matsubara frequencies.
                            The shape of the array is
                            (num_imag, num_orb, num_orb)
    """
    # number of imag frequency points
    num_imag = len(zM)
    # Get the number of poles.
    num_poles = len(poles)
    # number of orbitals
    num_orb = GM_matrix.shape[-1]

    # Initialize the cvxpy variables X_vec.
    # These variables are positive-semidefinite (PSD) and Hermitian matrices.
    # Loosely speaking, the X variables are of the shape
    #   X_vec = np.zeros((Num_poles, N_orb, N_orb), dtype=complex)
    X_vec = cp.Variable((num_poles, num_orb * num_orb), complex=False)

    # Operator for transposse of X_vec matrices
    # We will use this operator to enforce Hermitian nature.
    X_trans = np.zeros((num_orb * num_orb, num_orb * num_orb))
    for i in range(num_orb * num_orb):
        j = num_orb * (i % num_orb) + i // num_orb
        X_trans[j, i] = 1.0

    # PSD and Hermitian constraint
    constr = []
    for i in range(num_poles):
        constr.append(
            cp.reshape(X_vec[i, :], (num_orb, num_orb)) >> 0
        )

    #
    # Define the objective function
    #

    # construct iw - poles matrix
    iw_pole_matrix = np.zeros((num_imag, num_poles), dtype=complex)
    for iw in range(num_imag):
        for m in range(num_poles):
            iw_pole_matrix[iw, m] = 1 / (1j * zM[iw] - poles[m])

    # Construct G_approx of shape ((N_orb, N_orb, Num_poles), dtype=complex)
    G_approx = iw_pole_matrix @ X_vec \
        + iw_pole_matrix @ cp.conj(X_vec) @ X_trans
    obj_qty = cp.norm(
        G_approx - GM_matrix.reshape((num_imag, num_orb * num_orb)),
        p='fro'
    )

    #
    # Define cvxy's Objective problem and solve itjj
    #

    prob = cp.Problem(cp.Minimize(obj_qty), constr)
    opt_error = prob.solve(eps=1e-6)
    # print("Constraint values: ")
    # constr_values = [constr[i].dual_value for i in range(num_poles)]
    # print(constr_values)

    #
    # Get results and store in np_X_vec
    #

    # X vectors
    np_X_vec = X_vec.value
    np_X_vec += np.conj(np_X_vec) @ X_trans
    np_X_vec = np_X_vec.reshape((num_poles, num_orb, num_orb))
    # approximated Green's function on iw points
    np_G_approx = np.einsum('wp, pij -> wij', iw_pole_matrix, np_X_vec)

    return opt_error, np_X_vec, np_G_approx


def cvx_optimize_spectral(poles, GM_diags, zM):
    """Top level target error function in the ES approach
    to Nevanlinna, i.e.,
        Err (poles) = min_{X} || Gimag (iw) - Gapprox [poles, X] (iw) ||
    This function only computes the spectral function, not the full G matrix.
    Input Arguments:
        poles           :   pole positions
        GM_diags        :   Exact Matsubara Green's function diagonal data
                            in the shape (num_imag, num_orb)
        zM              :   Matsubara frequencies
    Returns:
        opt_error       :   Optimal value of Err (poles) for fixed poles
        np_X_vec        :   Numpy array rep of optimal X_vec in the shape
                            (num_poles, num_orb)
        np_G_approxx    :   Numpy array rep of approximated optimal Green's
                            function for the specified Matsubara frequencies.
                            The shape of the array is
                            (num_imag, num_orb)
    """
    # number of imag frequency points
    num_imag = len(zM)
    # Get the number of poles.
    num_poles = len(poles)
    # number of orbitals
    num_orb = GM_diags.shape[-1]

    # Initialize the cvxpy variables X_vec.
    # These variables are positive-semidefinite (PSD) and Hermitian matrices.
    # Loosely speaking, the X variables are of the shape
    #   X_vec = np.zeros((Num_poles, N_orb, N_orb), dtype=complex)
    X_vec = cp.Variable((num_poles, num_orb), nonneg=True)

    #
    # Define the objective function
    #

    # construct iw - poles matrix
    iw_pole_matrix = np.zeros((num_imag, num_poles), dtype=complex)
    for iw in range(num_imag):
        for m in range(num_poles):
            iw_pole_matrix[iw, m] = 1 / (1j * zM[iw] - poles[m])

    # Construct G_approx of shape ((N_orb, N_orb, Num_poles), dtype=complex)
    G_approx = iw_pole_matrix @ X_vec
    obj_qty = cp.norm(
        G_approx - GM_diags.reshape((num_imag, num_orb)),
        p='fro'
    )

    #
    # Define cvxy's Objective problem and solve itjj
    #

    prob = cp.Problem(cp.Minimize(obj_qty))
    opt_error = prob.solve()
    # print("Constraint values: ")
    # constr_values = [constr[i].dual_value for i in range(num_poles)]
    # print(constr_values)

    #
    # Get results and store in np_X_vec
    #

    # X vectors
    np_X_vec = X_vec.value
    np_X_vec = np_X_vec.reshape((num_poles, num_orb))
    # approximated Green's function on iw points
    np_G_approx = iw_pole_matrix @ np_X_vec

    return opt_error, np_X_vec, np_G_approx


def cvx_gradient(poles, GM_matrix, zM):
    """Computes the gradient for top-level optimization of pole positions
    in ES approach to Nevanlinna
        d Err / d poles
    where,
        Err (poles) = min_{X} || Gimag (iw) - Gapprox [poles, X] (iw) ||
    Specifically, G_approx is defined as
        G_approx = sum_m X_vec(:,:,m)/(1j*zM - poles(m))
        X_vec is shape (N_orb, N_orb, num_poles) and is just an array of
        optimized X_l matrices, for given vallue of pole positions poles
    Args:
        poles           :   pole positions
        GM_matrix       :   Exact Matsubara Green's function data in the shape
                            (num_imag, num_orb, num_orb)
        zM              :   Matsubara frequencies
    Returns:
        grad            :   Gradient of E with respect to poles
    """

    #
    # Initialization
    #

    # number of imag frequency points
    num_imag = len(zM)
    # Get the number of poles.
    Num_poles = len(poles)

    # Get the norm of obj.
    err_value, X_vec, G_approx = cvx_optimize(poles, GM_matrix, zM)
    Gdiff = GM_matrix - G_approx

    # compute gradient
    grad = np.zeros((Num_poles))

    # Loop over all Matsubara frequencies
    for iw in range(num_imag):
        # Initialize the variable Ghere.
        dG_iw = Gdiff[iw]

        # Loop over all poles.
        for k in range(Num_poles):
            # Add the contribution from the pole k to grad.
            gradhere = X_vec[k, :, :] / (
                (1j * zM[iw] - poles[k])**2
            )
            grad[k] += np.real(np.einsum('ij, ij ->', dG_iw, gradhere))

    # Scale the gradient by -1/y.
    grad = -grad / err_value

    return grad


def run_es(
    iw_vals, G_iw, re_w_vals, diag=True, eta=0.01, eps_pol=1,
    ofile='Xw_real.txt'
):
    """Pole estimation and semi-definite relaation algorithm based on
    Huang, Gull and Lin, 10.1103/PhysRevB.107.075151
    Input args
        iw_vals     :   imaginary frequency values (value only, without iota)
        X_iw        :   matsubara quantity to be analytically continued
        re_w_vals   :   real frequency grid to perform continuation on
        eta         :   broadening
        eps_pol     :   max imag part of the poles to be considered
    """

    # step 1: AAA algorithm for pole estimation
    if len(G_iw.shape) == 1:
        G_iw = G_iw.reshape(G_iw.shape + (1, ))
        g_trace = G_iw * 1.0
    elif len(G_iw.shape) == 2:
        g_trace = G_iw * 1.0
    elif len(G_iw.shape) == 3:
        g_trace = np.einsum('wpp -> w', G_iw)
    else:
        raise ValueError(
            'expect X_iw shape to be (nw, nao) or (nw, nao, nao), '
            + 'but received {}'.format(G_iw.shape)
        )
    res = baryrat.aaa(1j * iw_vals, g_trace)
    poles, _ = res.polres()
    poles = poles[np.abs(np.imag(poles)) < eps_pol]
    poles = np.asarray(np.real(poles))
    poles = np.unique(np.sort(poles))

    # step 2: semi-definite relaxation
    # optimize the poles
    if diag:
        res = minimize(
            lambda x: cvx_optimize_spectral(x, G_iw, iw_vals)[0], poles,
            tol=1e-6, options={
                "maxiter": 20,
                "eps": 1e-6,
            }
        )
        poles_opt = res.x
        _, X_vec, _ = cvx_optimize_spectral(poles_opt, G_iw, iw_vals)
    else:
        res = minimize(
            lambda x: cvx_optimize(x, G_iw, iw_vals)[0], poles,
            jac=lambda x: cvx_gradient(x, G_iw, iw_vals),
            tol=1e-6, options={
                "maxiter": 20,
                "eps": 1e-6,
            }
        )
        poles_opt = res.x
        _, X_vec, _ = cvx_optimize(poles_opt, G_iw, iw_vals)

    # Calculate the spectrum
    Greens_calc = np.zeros(
        (len(re_w_vals), ) + G_iw.shape[1:], dtype=complex
    )
    for i_re, re_w in enumerate(re_w_vals):
        Greenhere = np.zeros(G_iw.shape[1:], dtype=complex)
        for i_pol in range(len(poles_opt)):
            Greenhere += X_vec[i_pol] / (
                re_w + eta * 1j - poles_opt[i_pol]
            )
        Greens_calc[i_re] = Greenhere

    np.savetxt(ofile, Greens_calc.reshape(len(re_w_vals), -1))

    return
