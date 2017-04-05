# Wrappers
from .SplineFunction import SplineFunction
from .SplineSpace import SplineSpace
from .TensorProductSplineFunction import TensorProductSplineFunction
from .TensorProductSplineSpace import TensorProductSplineSpace
# Approximation and interpolation
from .approximation import variation_diminishing_spline_approximation, least_squares_spline_approximation, \
    least_squares_tensor_approximation
from .interpolation import linear_spline_interpolation, cubic_hermite_interpolation
# Auxiliary methods
from .lib import evaluate_non_zero_basis_splines, index, knot_averages, create_knots, create_interpolation_knots, \
    create_cubic_hermite_coefficients, approximate_derivatives, parametrize
