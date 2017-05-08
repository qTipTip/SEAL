from SEAL.SplineFunction import SplineFunction
from SEAL.lib import create_interpolation_knots, create_cubic_hermite_coefficients, approximate_derivatives


def linear_spline_interpolation(parameter_values, data_values):
    """
    Computes the linear spline interpolation to the given
    m data points (x_i, y_i).
    :param parameter_values: np.ndarray, shape (m, )
    :param data_values: np.ndarray, shape (m, 2)
    :return: SplineFunction of degree 1 representing the linear spline interpolation.
    """

    t_values = create_interpolation_knots(parameter_values, interpol_type='linear')
    p = 1

    f = SplineFunction(p, t_values, data_values)
    return f


def cubic_hermite_interpolation(parameter_values, function_values, derivatives=None):
    cubic_hermite_knots = create_interpolation_knots(parameter_values, interpol_type='cubic')
    p = 3
    if derivatives is None:
        derivatives = approximate_derivatives(parameter_values, function_values)

    cubic_hermite_coeff = create_cubic_hermite_coefficients(parameter_values, function_values, derivatives)
    return SplineFunction(p, cubic_hermite_knots, cubic_hermite_coeff)
