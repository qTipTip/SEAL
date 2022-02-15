from unittest import TestCase

import numpy as np

from SEAL.SplineFunction import SplineFunction


class TestSplineFunction(TestCase):
    def test__determine_coefficient_space(self):
        """
        Given:
            spline coefficients of 1, 2 and 3 dimensional type
        When:
            initializing a new spline function, and
            determining the spline coefficient space
        Then:
            scalar values --> 1 D
            2-tuples      --> 2 D
            3-tuples      --> 3 D
        """

        c_one = [0, 0, 0, 3, 5, 2, 0, 0, 0]
        c_two = [(0, 0), (0, 0), (0, 0), (3, 0), (5, 1), (2, 3), (0, 0), (0, 0), (0, 0)]
        c_three = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (1, 0, 0), (0, 0, 1), (0, 1, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)]
        p = 3
        t = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 6, 6, 6]

        expected_dimensions = [1, 2, 3]
        for expected_d, c in zip(expected_dimensions, [c_one, c_two, c_three]):
            f = SplineFunction(p, t, c)
            computed_d = f.d

            self.assertEqual(computed_d, expected_d)

    def test_call(self):
        """
        Given:
            A knot vector [0, 1, 2, 3, 4]
            A set of coefficients [3, 2, 4]
            A degree p = 1
        When:
            Evaluating the spline f at 0, 1.5, 3.5,
        Then:
            f(0) = 0
            f(1.5) = 2.5
            f(2.5) = 3
            f(3.5) = 2
        """

        t = [0, 0, 1, 2, 3, 4, 4]
        c = [0, 3, 2, 4, 0]
        p = 1
        f = SplineFunction(degree=p, knots=t, coefficients=c)

        expected_values = [0, 2.5]
        computed_values = [f(x) for x in [0, 1.5]]

        for e, c in zip(expected_values, computed_values):
            self.assertAlmostEqual(e, c)

    def test_call_two(self):
        """
        Given:
            A knot vector [0, 0, 1, 2, 3, 4, 4]
            A set of coefficients [0, 3, 2, 4, 0]
            A degree p = 1
        When:
            Evaluating the spline f at 0, 1.5, 2.5, 3.5,
        Then:
            f(0) = 0
            f(1.5) = 0.5
            f(2.5) = 3
            f(3.5) = 2
        """

        t = [0, 0, 1, 2, 3, 4, 4]
        c = [0, 3, 2, 4, 0]
        p = 1
        f = SplineFunction(degree=p, knots=t, coefficients=c)

        expected_values = [3.0, 2.0]
        computed_values = [f(x) for x in [2.5, 3.5]]

        for e, c in zip(expected_values, computed_values):
            self.assertAlmostEqual(e, c)

    def test_call_end_point(self):
        """
        Given:
        Given:
            A knot vector [0, 0, 1, 2, 3, 4, 4]
            A set of coefficients [0, 3, 2, 4, 0]
            A degree p = 1
        When:
            Evaluating the spline f at 4.0, while not defined in theory,
            we allow it here.
        Then:
            f(4.0) = 0 
        """

        t = [0, 0, 1, 2, 3, 4, 4]
        c = [0, 3, 2, 4, 0]
        p = 1
        f = SplineFunction(degree=p, knots=t, coefficients=c)
        x = 4

        expected_value = 0.0
        computed_values = float(f(x))

        self.assertAlmostEqual(expected_value, computed_values)

    def test_basis_spline(self):
        """
        Given:
            A knot vector [0, 0, 0, 1, 2, 3, 4, 5, 5, 5]
            A set of coefficients [0, 0, 1, 0, 0]
            A degree p = 2
        When:
            Evaluating the second basis spline over the range [0, 5]
        Then:
            B(x) = as expected_func below
        """

        def expected_func(x):

            if 1 <= x <= 2:
                return (x - 1) ** 2 / 2.0
            elif 2 <= x < 3:
                return (x - 1) * (3 - x) / 2.0 + (4 - x) * (x - 2) / 2.0
            elif 3 <= x < 4:
                return (4 - x) ** 2 / 2.0
            else:
                return 0.0

        t = [0, 0, 0, 1, 2, 3, 4, 5, 5, 5]
        p = 2
        c = [0, 0, 0, 1, 0, 0, 0]
        f = SplineFunction(p, t, c)

        x_values = np.linspace(t[0], t[-1], 100)

        y_computed = f(x_values)
        y_expected = [expected_func(x) for x in x_values]

        for e, c in zip(y_expected, y_computed):
            self.assertAlmostEqual(e, c)
