import numpy as np

from SEAL.SplineFunction import SplineFunction
from SEAL.SplineSpace import SplineSpace
from SEAL.lib import knot_averages


def variation_diminishing_spline_approximation(f, p, t):
    """
    Given a callable function f defined on the knot vector of S,
    finds the variation diminishing spline approximation (VDSA) to f
    in the spline space S.

    :param f: callable function defined on knot vector
    :param p: spline degree
    :param t: p + 1 regular knot vector with t1 = a, t_n+1 = b
    :return: the variation diminishing spline approximation to f
    """

    vdsa_coefficients = [f(tau) for tau in knot_averages(t, p)]
    return SplineFunction(p, t, vdsa_coefficients)


def least_squares_spline_approximation(parameter_values, data_values, spline_space, weights=None):
    """
    Given a set of m data points (x_i, y_i), and a SplineSpace S,
    compute the weighted least squares spline approximation to the data. 
    :type spline_space: SplineSpace
    :param parameter_values: np.ndarray, shape (m, 2)
    :param data_values: np.ndarray, shape (m, 2)
    :param spline_space: SplineSpace object
    :param weights: Optional. np.ndarray, shape (m, 1), 
    :return: SplineFunction, the least squares spline approximation
    """
    m = len(parameter_values)
    n = spline_space.n
    basis = spline_space.basis

    if not weights:
        weights = np.ones(m)

    # TODO: Make sure this is sufficient
    if isinstance(data_values, (list, tuple)) or data_values.ndim == 1:
        dim = 1
        data_values = np.reshape(data_values, (m, 1))
    else:
        _, dim = data_values.shape

    A = np.zeros(shape=(m, n, dim))
    b = np.zeros(shape=(m, dim))
    for i in range(m):
        for j in range(n):
            A[i, j] = weights[i] * basis[j](parameter_values[i])
        b[i] = weights[i] * data_values[i, :]

    coefficients = []
    for i in range(dim):
        component = np.linalg.solve(A[:, :, i].T.dot(A[:, :, i]), A[:, :, i].T.dot(b[:, i]))
        coefficients.append(component)

    coefficients = np.column_stack(coefficients)
    return spline_space(coefficients)


def least_squares_tensor_approximation(parameter_values, data_values, spline_space, weights=None):
    """
    Given a set of gridded data
    :param parameter_values: two arrays of length (m1,) and (m2,) respectively
    :param data_values: a (m1, m2, dim) ndarray
    :param spline_space: a tensor product spline space
    :param weights: Optional: two arrays of length (m1,) and (m2,) respectively
    :return: A TensorProductSplineFunction
    """
    x_values, y_values = parameter_values
    mx, my = len(x_values), len(y_values)
    nx, ny = spline_space.n
    basisx, basisy = spline_space.basis

    if weights is None:
        weights = np.ones(mx), np.ones(my)

    if len(data_values.shape) == 2:
        dim = 1
        data_values = np.reshape(data_values, (mx, my, 1))
    else:
        _, _, dim = data_values.shape

    # assemble matrices
    A = np.zeros(shape=(mx, nx, dim))
    B = np.zeros(shape=(my, ny, dim))
    G = np.zeros(shape=(mx, my, dim))
    w, v = np.sqrt(weights[0]), np.sqrt(weights[1])

    for i in range(mx):
        for q in range(nx):
            A[i, q, :] = w[i] * basisx[q](x_values[i])
    for j in range(my):
        for r in range(ny):
            B[j, r, :] = v[j] * basisy[r](y_values[j])

    for i in range(mx):
        for j in range(my):
            G[i, j, :] = w[i] * v[j] * data_values[i, j, :]

    coefficients = np.zeros(shape=(nx, ny, dim))
    for i in range(dim):
        D = np.linalg.solve(A[:, :, i].T.dot(A[:, :, i]), A[:, :, i].T.dot(G[:, :, i]))
        component = np.linalg.solve(B[:, :, i].T.dot(B[:, :, i]), B[:, :, i].T.dot(D.T))
        coefficients[:, :, i] = component.T  # TODO, check this transpose

    return spline_space(coefficients)
