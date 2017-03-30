from SEAL.SplineFunction import SplineFunction
from SEAL.lib import create_interpolation_knots, create_cubic_hermite_coefficients


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
    (x_i, f(x_i), f'(x_i))
    :param data_values: np.ndarray, shape(m, 3)
    :return: SplineFunction of degree 3 representing the linear spline interpolant to the data. 
    """

    m, _ = data_values.shape
    p = 3

    x_values, f_values, df_values = data_values.T
    cubic_hermite_knots = create_interpolation_knots(x_values, interpol_type='cubic')
    cubic_hermite_coeff = create_cubic_hermite_coefficients(data_values)

    return SplineFunction(p, cubic_hermite_knots, cubic_hermite_coeff)
