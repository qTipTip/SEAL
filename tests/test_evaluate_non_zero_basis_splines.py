from unittest import TestCase

import numpy as np

from SEAL.lib import evaluate_non_zero_basis_splines


class TestEvaluate_non_zero_basis_splines(TestCase):
    def test_evaluate_non_zero_basis_splines_regular_vector(self):
        """
        Given:
            A p + 1 extended knot vector t = [0, 0, 1, 2, 2]
            A degree p = 1
            An index mu = 1
        When:
            Evaluating the non-zero splines at the point x = 0
        Then:
            b = [1.0, 0.0]
        """

        p = 1
        t = np.array([0, 0, 1, 2, 2], dtype=np.float64)
        mu = 1
        x = 0

        expected_values = [1.0, 0.0]
        computed_values = evaluate_non_zero_basis_splines(x, mu, t, p)

        np.testing.assert_array_almost_equal(expected_values, computed_values)

    def test_evaluate_non_zero_basis_splines_non_regular_vector_start(self):
        """
        Given:
            A knot vector t = [0, 1, 2, 3, 4]
            A degree p = 1
            An index mu = 1
        When:
            Evaluating the non-zero basis splines at the point x = 0
        Then:
            b = [0]
        """
        p = 1
        t = np.array([0, 1, 2, 3, 4], dtype=np.float64)
        mu = 0
        x = 0

        expected_values = [0]
        computed_values = evaluate_non_zero_basis_splines(x, mu, t, p)
        np.testing.assert_array_almost_equal(expected_values, computed_values)

    def test_evaluate_non_zero_basis_splines_non_regular_vector_end(self):
        """
        Given:
            A knot vector t = [0, 1, 2, 3, 4]
            A degree p = 1
            An index mu = 3
        When:
            Evaluating the non-zero basis splines at the point x = 3.5
        Then:
            b = [0.5]
        """
        p = 1
        t = np.array([0, 1, 2, 3, 4], dtype=np.float64)
        mu = 3
        x = 3.5

        expected_values = [0.5]
        computed_values = evaluate_non_zero_basis_splines(x, mu, t, p)
        np.testing.assert_array_almost_equal(expected_values, computed_values)

    def test_evaluate_non_zero_basis_splines_non_regular_vector_middle(self):
        """
        Given:
            A knot vector t = [0, 1, 2, 3, 4]
            A degree p = 1
            An index mu = 1
        When:
            Evaluating the non-zero basis splines at the point x = 1.5
        Then:
            b = [2]
        """

        p = 1
        t = np.array([0, 1, 2, 3, 4], dtype=np.float64)
        mu = 1
        x = 1.5

        expected_values = [0.5, 0.5]
        computed_values = evaluate_non_zero_basis_splines(x, mu, t, p)
        np.testing.assert_array_almost_equal(expected_values, computed_values)

    def test_evaluate_non_zero_basis_splines_non_regular_vector_end_quadratic(self):
        t = np.array([0, 1, 2, 3, 4, 5], dtype=np.float64)
        p = 2
        mu = 4
        x = 4.5

        expected_values = [0.125]
        computed_values = evaluate_non_zero_basis_splines(x, mu, t, p)

        np.testing.assert_array_almost_equal(expected_values, computed_values)
