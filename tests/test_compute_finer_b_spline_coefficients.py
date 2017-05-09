from unittest import TestCase

from SEAL.lib import compute_fine_spline_coefficients
import numpy as np

class TestComputeFineSplineCoefficients(TestCase):
    def test_compute_fine_spline_coefficients_quadratic(self):
        """
           Given:
               p = 2
               tau = [-1, -1, -1, 0, 1, 1, 1]
               t = [-1, -1, -1, -0.5, 0, 0.5, 1, 1, 1]
               c = [1. -2, 2, -1]
           When:
               Representing the spline with coefficients c in the finer spline space
           Then:
                Fine spline coefficients b = [1, -0.5, -1, 1, 0.5, -1]
        """
        p = 2
        c = [1, -2, 2, -1]
        tau = [-1, -1, -1, 0, 1, 1, 1]
        t = [-1, -1, -1, -0.5, 0, 0.5, 1, 1, 1]

        expected_b = np.array([1, -0.5, -1, 1, 0.5, -1])
        computed_b = compute_fine_spline_coefficients(p, tau, t, c)

        for e, c in zip(expected_b, computed_b):
            self.assertAlmostEqual(e, c)