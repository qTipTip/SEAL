from unittest import TestCase

from SplineFunction import SplineFunction


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