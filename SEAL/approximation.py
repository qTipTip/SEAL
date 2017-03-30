from SEAL.SplineFunction import SplineFunction
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
