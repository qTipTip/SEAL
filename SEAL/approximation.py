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


def least_squares_tensor_approximation(data_values, spline_space, weights=None):
    """
    Given a set of gridded data 
    :param data_values: 
    :param spline_space: 
    :param weights: 
    :return: 
    """
