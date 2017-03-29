import numpy as np

def evaluate_non_zero_basis_splines(x, mu, t, p):
    """
   Evaluates the non-zero basis splines of degree p at the point x.
   mu denotes the index such that t_{mu} <= x < t_{mu + 1}
   :param x: parameter value
   :param mu: index
   :param t: knot vector
   :param p: spline degree
   :return: vector b of at most p + 1 non-zero splines evaluated at x
    """

    # TODO: Now that p+1 regularity is enforced, we might not need to check
    # these cases.
    n = len(t) - p - 1
    if mu - p < 0:
        # Too few knots at start of knot vector
        t = np.insert(t, 0, [t[0] - 1]*(p+1))
        new_mu = index(x, t)
        b = evaluate_non_zero_basis_splines(x, new_mu, t, p)
        return b[p:]
    elif mu > n-1:
        # Too few knots at end of knot vector
        t = np.append(t, [t[-1] + 1] * (p+1))
        new_mu = index(x, t)
        b = evaluate_non_zero_basis_splines(x, new_mu, t, p)
        return b[:-p]
    else:
        # No action needed
        b = 1
        for k in range(1, p + 1):
            # extract relevant knots
            t1 = t[mu - k + 1: mu + 1]
            t2 = t[mu + 1: mu + k + 1]
            # append 0 to end of first term, and insert 0 to start of second term
            omega = np.divide((x - t1), (t2 - t1), out=np.zeros_like(t1), where=((t2 - t1) != 0))
            b = np.append((1 - omega) * b, 0) + np.insert((omega * b), 0, 0)

        return b

def index(x, t):
    """
    Given a knot vector t, find the index mu
    such that t[mu] <= x < t[mu+1]
    :param x: Parameter value
    :param t: knot vector
    :return: 
    """
    for i in range(len(t)-1):
        if t[i] <= x < t[i+1]:
            return i
    return i

def knot_averages(t, p):
    """
    Given a knot vector of length p + n + 1,
    returns the knot averages. Main use case is
    the plotting of control polygons, and the
    variation diminishing spline approximation.
    :param t: knot vector 
    :param p: spline degree
    :return: array of knot averages
    """

    n = len(t) - p - 1
    k = np.zeros(n)
    for i in range(n):
        k[i] = sum(t[i + 1: i + p + 1]) / float(p)
    return k


def create_knots(a, b, p, n):
    """
    Returns a p+1 regular knot vector starting at a and ending at b
    with a total length of n + p + 1. A spline space on such a knot vector will have n
    basis splines.
    :param a: float, start
    :param b: float, end
    :param p: int, spline degree
    :param n: int, number of basis splines
    :return: np.ndarray, p+1 regular knot vector
    """

    interior_knots = np.linspace(a, b, num=n - p + 1)
    regular_knots = np.lib.pad(interior_knots, (p, p), mode='edge')

    return regular_knots
