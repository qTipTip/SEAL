from SEAL.SplineFunction import SplineFunction
from SEAL.lib import create_interpolation_knots


def linear_spline_interpolation(data_values):
    """
    Computes the linear spline interpolation to the given
    m data points (x_i, y_i).
    :param data_values: np.ndarray, shape (m, 2)
    :return: SplineFunction of degree 1 representing the linear spline interpolation.
    """

    x_values, y_values = data_values.T
    t_values = create_interpolation_knots(x_values)
    p = 1

    f = SplineFunction(p, t_values, y_values)
    return f
