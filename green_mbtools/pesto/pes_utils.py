import numpy as np
import cvxpy as cp
import baryrat
from scipy.optimize import minimize


def cvx_matrix_projection(zM, GM_matrix, w_cut=10, n_real=201, ofile='Giw', solver='SCS', **kwargs):
    """Projection of noisy Green's function matrix data on to Nevanlinna manifold.
    Once this is performed, analytic continuation either using Nevanlinna
    or ES becomes easier.
    The main idea is to obtain a projected Matsubara Green's function of the form

        G(iw_m) = sum_n P_n / (iw_m - w_n)

    Parameters
    ----------
    zM : numpy.ndarray
        n_imag number of imaginary frequency values
    GM_matrix : numpy.ndarray
        Exact Matsubara Green's function data in the shape (n_imag, n_orb, n_orb)
    w_cut : float, optional
        cut-off to consider for real axis, by default 10.0
    n_real : int, optional
        number of real frequencies to consider, by default 201
    ofile : str, optional
        Output file in which the green's function data will be dumped, by default 'Giw'
    solver : str, optional
        CVXPy solver to use, by default 'SCS'
    **kwargs : dict, optional
        dictionary of keyword arguments for CVXPy solver

    Returns
    -------
    numpy.ndarray
        Projected Green's function
    """
    # number of imag frequency points
    n_imag = len(zM)

    # number of orbitals
    n_orb = GM_matrix.shape[-1]
    assert GM_matrix.shape == (n_imag, n_orb, n_orb)

    # real frequency grid
    w_grid = np.linspace(-w_cut, w_cut, n_real)

    # Initialize the cvxpy variables P_vec
    # These variables are positive-semidefinite (PSD)
    P_vec = cp.Variable((n_real, n_orb * n_orb), complex=True)

    # PSD and Hermitian constraint
    constr = []
    for i in range(n_real):
        constr.append(
            cp.reshape(P_vec[i, :], (n_orb, n_orb)) >> 0
        )

    #
    # Define the objective function
    #

    # real and imaginary mesh matrix
    zw_matrix = np.zeros((n_imag, n_real), dtype=complex)
    for i, iw in enumerate(zM):
        for j, w in enumerate(w_grid):
            zw_matrix[i, j] = 1 / (iw - w)

    G_approx = zw_matrix @ P_vec
    G_diff = G_approx - GM_matrix.reshape((n_imag, n_orb * n_orb))
    obj_qty = cp.norm(G_diff, p='fro')

    #
    # Define cvxy's Objective problem and solve it
    #

    prob = cp.Problem(cp.Minimize(obj_qty), constr)
    # NOTE: using default convergence criterion for the solver
    opt_error = prob.solve(solver=solver, **kwargs)
    print("Error CVXPy optimization: ", opt_error)
    print("Objective value after optimization: ", obj_qty.value)

    #
    # Get results for the projected imaginary-time Greens function
    #

    np_P_vec = P_vec.value
    G_iw_proj = zw_matrix @ np_P_vec
    np.savetxt(ofile, G_iw_proj)

    return


def cvx_diag_projection(zM, GM_diag, w_cut=10, n_real=201, ofile='Giw', solver='SCS', **kwargs):
    """Projection of noisy Green's function diagonal data on to Nevanlinna manifold.
    Once this is performed, analytic continuation either using Nevanlinna
    or ES becomes easier.
    The main idea is to obtain a projected Matsubara Green's function of the form

        G(iw_m) = sum_n P_n / (iw_m - w_n)

    Parameters
    ----------
    zM : numpy.ndarray
        n_imag number of imaginary frequency values
    GM_matrix : numpy.ndarray
        Exact Matsubara Green's function data in the shape (n_imag, n_orb, n_orb)
    w_cut : float, optional
        cut-off to consider for real axis, by default 10.0
    n_real : int, optional
        number of real frequencies to consider, by default 201
    ofile : str, optional
        Output file in which the green's function data will be dumped, by default 'Giw'
    solver : str, optional
        CVXPy solver to use, by default 'SCS'
    **kwargs : dict, optional
        dictionary of keyword arguments for CVXPy solver

    Returns
    -------
    numpy.ndarray
        Projected Green's function
    """
    # number of imag frequency points
    n_imag = len(zM)

    # number of orbitals
    n_orb = GM_diag.shape[-1]
    assert GM_diag.shape == (n_imag, n_orb)

    # real frequency grid
    w_grid = np.linspace(-w_cut, w_cut, n_real)

    # Initialize the cvxpy variables P_vec
    # These variables are non-negative vectors
    P_vec = cp.Variable((n_real, n_orb), nonneg=True)

    #
    # Define the objective function
    #

    # real and imaginary mesh matrix
    zw_matrix = np.zeros((n_imag, n_real), dtype=complex)
    for i, iw in enumerate(zM):
        for j, w in enumerate(w_grid):
            zw_matrix[i, j] = 1 / (iw - w)

    G_approx = zw_matrix @ P_vec
    G_diff = G_approx - GM_diag
    obj_qty = cp.norm(G_diff, p='fro')

    #
    # Define cvxy's Objective problem and solve it
    #

    prob = cp.Problem(cp.Minimize(obj_qty))
    # NOTE: using default convergence criterion for the solver
    opt_error = prob.solve(solver=solver, **kwargs)
    print("Error CVXPy optimization: ", opt_error)
    print("Objective value after optimization: ", obj_qty.value)

    #
    # Get results for the projected imaginary-time Greens function
    #

    np_P_vec = P_vec.value
    G_iw_proj = zw_matrix @ np_P_vec
    G_iw_proj = G_iw_proj.reshape((n_imag, n_orb))
    np.savetxt(ofile, G_iw_proj)

    return


def cvx_optimize(poles, GM_matrix, zM, solver='SCS', **kwargs):
    """Top level target error function in the ES approach for matrix analytic continuation. It returns

        Err (poles) = min_{X} || Gimag (iw) - Gapprox [poles, X] (iw) ||
    
    Parameters
    ----------
    poles : numpy.ndarray
        1D array of pole positions
    GM_matrix : numpy.ndarray
        Exact Matsubara Green's function data in the shape (num_imag, num_orb, num_orb)
    zM : numpy.ndarray
        1D Matsubara frequencies
    solver : str, optional
        CVXPy solver for semi-definite relaxation in the ES continuation, by default 'SCS'
    **kwargs : dict, optional
        dictionary of keyword arguments for CVXPy solver

    Returns
    -------
    float
        Optimal value of error function Err (poles) for fixed poles
    numpy.ndarray
        optimal X_vec in the shape (num_poles, num_orb, num_orb)
    numpy.ndarray
        approximated optimal Green's function for the specified Matsubara frequencies.
        The shape of the array is (num_imag, num_orb, num_orb)
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
    # Define cvxy's Objective problem and solve it
    #

    prob = cp.Problem(cp.Minimize(obj_qty), constr)
    # NOTE: using default convergence criterion for the solver
    opt_error = prob.solve(solver=solver, **kwargs)
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


def cvx_optimize_spectral(poles, GM_diags, zM, solver='SCS', **kwargs):
    """Top level target error function in the ES approach for spectral analytic continuation. It returns

        Err (poles) = min_{X} || Diag(Gimag (iw) - Gapprox [poles, X] (iw)) ||
    
    Parameters
    ----------
    poles : numpy.ndarray
        1D array of pole positions
    GM_diags : numpy.ndarray
        Exact Matsubara Green's function diagonal data in the shape (num_imag, num_orb)
    zM : numpy.ndarray
        1D Matsubara frequencies
    solver : str, optional
        CVXPy solver for semi-definite relaxation in the ES continuation, by default 'SCS'
    **kwargs : dict, optional
        dictionary of keyword arguments for CVXPy solver

    Returns
    -------
    float
        Optimal value of error function Err (poles) for fixed poles
    numpy.ndarray
        optimal X_vec in the shape (num_poles, num_orb)
    numpy.ndarray
        approximated optimal Green's function diagonals for the specified Matsubara frequencies.
        The shape of the array is (num_imag, num_orb)
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
    # NOTE: using default convergence criterion for the solver
    opt_error = prob.solve(solver=solver, **kwargs)
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
    """Computes the gradient for top-level optimization of pole positions in ES analytic continuation
    i.e., d Err / d poles where,

        `Err (poles) = min_{X} || Gimag (iw) - Gapprox [poles, X] (iw) ||`

    Specifically, G_approx is defined as

        `G_approx = sum_m X_vec(:,:,m)/(1j*zM - poles(m))`
        `X_vec is shape (N_orb, N_orb, num_poles) and is just an array of`
        `optimized X_l matrices, for given vallue of pole positions poles`

    Parameters
    ----------
    poles : numpy.ndarray
        pole positions
    GM_matrix : numpy.ndarray
        Exact Matsubara Green's function data in the shape (num_imag, num_orb, num_orb)
    zM : numpy.ndarray
        Matsubara frequencies

    Returns
    -------
    numpy.ndarray
        Gradient of E with respect to poles
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
    ofile='Xw_real.txt', solver='SCS', **kwargs
):
    """Pole estimation and semi-definite (PES) relaation algorithm based on
    Huang, Gull and Lin, 10.1103/PhysRevB.107.075151. The output is saved in ofile.

    Parameters
    ----------
    iw_vals : numpy.ndarray
        1D array of imaginary frequency values (value only, without i)
    G_iw : numpy.ndarray
        matsubara quantity to be analytically continued
    re_w_vals : numpy.ndarray
        real frequency grid to perform continuation on
    diag : bool, optional
        perform continuation for diagonal entries only if set to True, by default True
    eta : float, optional
        broadening parameter, by default 0.01
    eps_pol : float, optional
        maximum imag part of the poles to be considered, by default 1
    ofile : str, optional
        Path or name of output file for storing the continued data
    solver : str, optional
        CVXPy solver to use (options: SCS, MOSEK, CLARABEL, etc.), by default 'SCS'
    **kwargs : dict, optional
        dictionary of keyword arguments for CVXPy solver
    """

    # step 1: AAA algorithm for pole estimation
    if len(G_iw.shape) == 1:
        G_iw = G_iw.reshape(G_iw.shape + (1, ))
        g_trace = G_iw * 1.0
    elif len(G_iw.shape) == 2:
        g_trace = np.einsum('wa -> w', G_iw)
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
            lambda x: cvx_optimize_spectral(
                x, G_iw, iw_vals, solver, **kwargs
            )[0], poles, tol=1e-6, options={
                "maxiter": 30,
                "eps": 1e-7,
            }
        )
        poles_opt = res.x
        opt_error, X_vec, _ = cvx_optimize_spectral(
            poles_opt, G_iw, iw_vals, solver, **kwargs
        )
    else:
        res = minimize(
            lambda x: cvx_optimize(x, G_iw, iw_vals, solver, **kwargs)[0],
            poles, jac=lambda x: cvx_gradient(x, G_iw, iw_vals),
            tol=1e-6, options={
                "maxiter": 30,
                "eps": 1e-7,
            }
        )
        poles_opt = res.x
        opt_error, X_vec, _ = cvx_optimize(
            poles_opt, G_iw, iw_vals, solver, **kwargs
        )

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
    np.savetxt(ofile.replace('.txt', '_error.txt'), np.asarray([opt_error]))

    return
