from unittest import TestCase

from SEAL import SplineFunction
from SEAL.lib import evaluate_blossom


class TestEvaluate_blossom(TestCase):
    def test_evaluate_blossom(self):
        p = 2
        t = [0, 0, 0, 1, 2, 2, 2]
        c = [-1, 1, -1, 1]

        f = SplineFunction(p, t, c)

        cases = [(2, 0.5), (2, 0), (3, 1.5), (3, 1)]

        for case in cases:
            mu, x = case
            x_values = [x]*p
            computed_values = evaluate_blossom(p, t, mu, c, x_values)
            expected_values = f(x)

        self.assertAlmostEqual(computed_values, expected_values)
