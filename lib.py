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
    n = len(t) - p - 1
    if mu - p < 0:
        # Too few knots at start of knot vector
        t = np.insert(t, 0, [t[0] - 1]*(p))
        new_mu = index(x, t)
        b = evaluate_non_zero_basis_splines(x, new_mu, t, p)
        return b[p:]
    elif mu > n-1:
        # Too few knots at end of knot vector
        t = np.append(t, [t[-1] + 1] * (p))
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
