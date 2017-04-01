from unittest import TestCase

import numpy as np

from SEAL.interpolation import cubic_hermite_interpolation


class TestCubicHermiteInterpolation(TestCase):
    def test_cubic_hermite_interpolation_supplied_derivatives(self):
        """
        Given:
            Interpolation problem:
                [a, b] = [x_1, x_m] = [0, 1]
                m = 2
                f(x) = x**4
                f'(x) = 4*x**3
        When:
            Finding the cubic spline interpolant Hf
        Then:
            Hf(x) = 2x**3 - x**2
        """

        data_values = np.array([(0, 0, 0), (1, 1, 4)])
        x_values = [0, 1]
        f_values = [0, 1]
        df_values = [0, 4]

        Hf = cubic_hermite_interpolation(x_values, f_values, df_values)

        x_values = np.linspace(0, 1, num=30)
        expected_values = [2 * x ** 3 - x ** 2 for x in x_values]
        computed_values = Hf(x_values)

        for e, c in zip(expected_values, computed_values.flat):
            self.assertAlmostEqual(e, c)
