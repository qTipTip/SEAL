from unittest import TestCase

from SplineFunction import SplineFunction
import numpy as np


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

        c_one = [3, 5, 2]
        c_two = [(3, 0), (5, 1), (2, 3)]
        c_three = [(1, 0, 0), (0, 0, 1), (0, 1, 0)]
        p = 3
        t = [0, 1, 2, 3, 4, 5, 6]

        expected_dimensions = [1, 2, 3]
        for expected_d, c in zip(expected_dimensions, [c_one, c_two, c_three]):
            f = SplineFunction(p, t, c)
            computed_d = f.d

            self.assertEquals(computed_d, expected_d)

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

        t = [0, 1, 2, 3, 4]
        c = [3, 2, 4]
        p = 1
        f = SplineFunction(degree=p, knots=t, coefficients=c)

        expected_values = [0, 2.5]
        computed_values = [f(x) for x in [0, 1.5]]
        np.testing.assert_array_almost_equal(expected_values, computed_values)

    def test_call_two(self):
        """
        Given:
            A knot vector [0, 1, 2, 3, 4]
            A set of coefficients [3, 2, 4]
            A degree p = 1
        When:
            Evaluating the spline f at 0, 1.5, 3.5,
        Then:
            f(0) = 0
            f(1.5) = 0.5
            f(2.5) = 3
            f(3.5) = 2
        """

        t = [0, 1, 2, 3, 4]
        c = [3, 2, 4]
        p = 1
        f = SplineFunction(degree=p, knots=t, coefficients=c)

        expected_values = [3.0, 2.0]
        computed_values = [f(x) for x in [2.5, 3.5]]

        for e, c in zip(expected_values, computed_values):
            self.assertAlmostEqual(e, c)