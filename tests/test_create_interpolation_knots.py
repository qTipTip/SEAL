from unittest import TestCase

from SEAL.lib import create_interpolation_knots


class TestCreateInterpolationKnots(TestCase):
    def test_create_interpolation_linear_knots(self):
        """
        Given:
            x_values = [0, 1, 2, 3, 4, 5, 6, 7]
        When:
            Constructing 2-regular knot vector from these x_values, for 
            linear spline interpolation.
        Then:
            t = [0, 0, 1, 2, 3, 4, 5, 6, 7, 7]
        """
        x_values = [0, 1, 2, 3, 4, 5, 6, 7]
        expected_values = [0, 0, 1, 2, 3, 4, 5, 6, 7, 7]
        computed_values = create_interpolation_knots(x_values, interpol_type='linear')

        for e, c in zip(expected_values, computed_values):
            self.assertAlmostEqual(e, c)

    def test_create_interpolation_cubic_knots(self):
        """
        Given:
            x_values = [0, 1, 2, 3, 4, 5, 6, 7]
        When: 
            Constructing the 4-regular knot vector from these x_values, with
            double interior knots, for cubic spline interpolation
        Then:
            t = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7, 7]
        """
        x_values = [0, 1, 2, 3, 4, 5, 6, 7]
        expected_values = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7, 7]
        computed_values = create_interpolation_knots(x_values, interpol_type='cubic')

        for e, c in zip(expected_values, computed_values):
            self.assertAlmostEqual(e, c)