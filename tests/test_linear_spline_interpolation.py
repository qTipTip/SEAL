from unittest import TestCase

import numpy as np

from SEAL.interpolation import linear_spline_interpolation


class TestLinearSplineInterpolation(TestCase):
    def test_linear_spline_interpolation(self):
        """
        Given:
            data_values = [(0, 1), (1, 0), (2, 5), (5, 3)]
        When:
            Computing the linear spline interpolant f to data_values
        Then:
            f(0) = 1, f(1) = 0, f(2) = 5, f(5) = 3
        """
        x_values = np.array([0, 1, 2, 5])
        y_values = np.array([1, 0, 5, 3])
        f = linear_spline_interpolation(x_values, y_values)

        expected_values = [1, 0, 5, 3]
        computed_values = [f(x) for x in x_values]

        for e, c in zip(expected_values, computed_values):
            self.assertAlmostEqual(e, c)
