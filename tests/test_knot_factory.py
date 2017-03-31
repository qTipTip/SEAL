from unittest import TestCase

from SEAL.lib import create_knots


class TestKnotFactory(TestCase):
    def test_create_knots(self):
        """
        Given:
            p = 2
            n = 5
            a = 0
            b = 3
        When:
            Computing the corresponding p+1 regular knot vector
        Then:
            t = [0, 0, 0, 1, 2, 3, 3, 3]
        """

        p = 2
        n = 5
        a = 0
        b = 3

        computed_values = create_knots(a, b, p, n)
        expected_values = [0, 0, 0, 1, 2, 3, 3, 3]

        for c, e in zip(computed_values, expected_values):
            self.assertAlmostEqual(c, e)
