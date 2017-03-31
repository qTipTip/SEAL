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


def least_squares_spline_approximation(data_values, spline_space, weights=None):
    """
    Given a set of m data points (x_i, y_i), and a SplineSpace S,
    compute the weighted least squares spline approximation to the data. 
    :type spline_space: SplineSpace
    :param data_values: np.ndarray, shape (m, 2)
    :param spline_space: SplineSpace object
    :param weights: Optional. np.ndarray, shape (m, 1), 
    :return: SplineFunction, the least squares spline approximation
    """

    m, _ = data_values.shape
    n = spline_space.n
    basis = spline_space.basis

    x_values, y_values = data_values.T
    if not weights:
        weights = np.ones(m)

    # initialize linear system
    A = np.zeros(shape=(m, n))
    b = np.zeros(m)

    for i in range(m):
        for j in range(n):
            A[i, j] = weights[i] * basis[j](x_values[i])
        b[i] = weights[i] * y_values[i]

    # solve linear system
    c = np.linalg.solve(A.T.dot(A), A.T.dot(b))

    return spline_space(c)
