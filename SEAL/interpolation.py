import numpy as np

from SEAL.SplineFunction import SplineFunction
from SEAL.lib import create_interpolation_knots, create_cubic_hermite_coefficients, approximate_derivatives


def linear_spline_interpolation(data_values):
    """
    Computes the linear spline interpolation to the given
    m data points (x_i, y_i).
    :param data_values: np.ndarray, shape (m, 2)
    :return: SplineFunction of degree 1 representing the linear spline interpolation.
    """

    x_values, y_values = data_values.T
    t_values = create_interpolation_knots(x_values, interpol_type='linear')
    p = 1

    f = SplineFunction(p, t_values, y_values)
    return f


def cubic_hermite_interpolation(data_values):
    """
    Computes the cubic hermite spline interpolation to the given m data points
    (x_i, f(x_i), f'(x_i)). Note that this is a local interpolation method.
    :param data_values: np.ndarray, shape(m, 2/3). If no derivatives given, they are approximated using
        interpolating quadratics.
    :return: SplineFunction of degree 3 representing the linear spline interpolant to the data. 
    """

    m, n = data_values.shape
    p = 3

    if n == 2:
        x_values, f_values = data_values.T
        df_values = approximate_derivatives(x_values, f_values)
        data_values = np.append(data_values, df_values[:, None], 1)
    else:
        x_values, f_values, df_values = data_values.T

    cubic_hermite_knots = create_interpolation_knots(x_values, interpol_type='cubic')
    cubic_hermite_coeff = create_cubic_hermite_coefficients(data_values)

    return SplineFunction(p, cubic_hermite_knots, cubic_hermite_coeff)
