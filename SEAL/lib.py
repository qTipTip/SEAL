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
        t = np.insert(t, 0, [t[0] - 1] * (p + 1))
        new_mu = index(x, t)
        b = evaluate_non_zero_basis_splines(x, new_mu, t, p)
        return b[p:]
    elif mu > n - 1:
        # Too few knots at end of knot vector
        t = np.append(t, [t[-1] + 1] * (p + 1))
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
    if abs(x - t[-1]) <= 1.0e-14:
        # at endpoint, return last non trivial index
        # TODO: Make this less hackish
        for i in range(len(t) - 1, 0, -1):
            if t[i] < x:
                return i
    for i in range(len(t) - 1):
        if t[i] <= x < t[i + 1]:
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


def create_interpolation_knots(x_values, interpol_type='linear'):
    """
    Constructs a suitable knot vector for use in interpolation.
    Interpol_type = linear --> 2-regular knot vector
    Interpol_type = cubic --> 4-regular knot vector with double interior knots
    :param x_values: np.ndarray, m data values
    :param interpol_type: linear/cubic
    :return: np.ndarray
    """
    if interpol_type == 'linear':
        return np.lib.pad(x_values, pad_width=(1, 1), mode='edge')
    elif interpol_type == 'cubic':
        # double all knots
        x_values = np.repeat(x_values, 2)
        # pad with two values at each end
        x_values = np.lib.pad(x_values, (2, 2), mode='edge')
        return x_values


def create_cubic_hermite_coefficients(x, f, df):
    """
    Computes the 2m spline coefficients for the cubic hermite spline interpolation.
    :param x: parameter_values, 1 dimensional array_like
    :param f: function_values, k-dimensional array_like
    :param df: derivatives, k-dimensional array_like
    :return: coefficients, k-dimensional np.ndarray 
    """
    m = len(x)
    x = np.lib.pad(x, (1, 1), 'edge')

    # TODO: Should do type checking somewhere else.
    if isinstance(f, (list, tuple)) or f.ndim == 1:
        f = np.array(f).reshape((m, 1))
        df = np.array(df).reshape((m, 1))

    _, k = f.shape
    cubic_hermite_coefficients = np.zeros(shape=(2 * m, k))

    for j in range(k):
        for i in range(m):
            cubic_hermite_coefficients[2 * i, j] = f[i, j] - (1.0 / 3.0) * (x[i + 1] - x[i]) * df[i, j]
            cubic_hermite_coefficients[2 * i + 1, j] = f[i, j] + (1.0 / 3.0) * (x[i + 2] - x[i + 1]) * df[i, j]

    return cubic_hermite_coefficients


def approximate_derivatives(x_values, f_values):
    """
    Approximate the derivatives in each point using interpolating parabolas.
    :param x_values: ndarray of shape (m, )
    :param f_values: ndarray of shape (m, )
    :return: ndarray of length m, df_values
    """

    m = len(x_values)

    # TODO: Do type checking somewhere else
    # if data is given as list, tuple or ndarray w/ shape (m,)
    if isinstance(f_values, (list, tuple)) or f_values.ndim == 1:
        f_values = np.array(f_values).reshape((m, 1))
    _, k = f_values.shape
    df_values = np.zeros(shape=(m, k))

    # pad values for end points
    x_values = np.lib.pad(x_values, pad_width=(1, 1), mode='constant', constant_values=(x_values[2], x_values[m - 3]))
    x_differences = x_values[1:] - x_values[:-1]

    for j in range(k):
        f_component = f_values[:, j]
        f_component = np.lib.pad(f_component, pad_width=(1, 1), mode='constant',
                                 constant_values=(f_component[2], f_component[m - 3]))
        f_differences = f_component[1:] - f_component[:-1]
        delta_values = np.divide(f_differences, x_differences)
        for i in range(0, m):
            df_values[i, j] = (x_differences[i] * delta_values[i + 1] + x_differences[i + 1] * delta_values[i]) / (
                x_differences[i] + x_differences[i + 1])
    return df_values


def parametrize(data_values, data_type='curve', parametrization_type='uniform'):
    """
    Given a set of data points (x, y) or (x, y, z),
    compute a suitable parametrization.
    :param data_values: 
    :return: 
    """

    if data_type == 'curve':
        m, _ = data_values.shape
        parameter_values = np.zeros(m)
        for i in range(1, m):
            parameter_values[i] = parameter_values[i-1] + np.linalg.norm(data_values[i] - data_values[i-1])
        return parameter_values

    elif data_type == 'surface':
        raise NotImplementedError("Parametrization of gridded data is not yet implemented")
        m, n, _ = data_values.shape
        parameter_values_x = np.zeros(m)
        parameter_values_y = np.zeros(n)
        for i in range(1, m):
            parameter_values_x[i] = parameter_values_y[i-1] + np.linalg.norm(data_values[i] - data_values[i-1])
