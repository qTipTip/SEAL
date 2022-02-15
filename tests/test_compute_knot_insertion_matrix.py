from unittest import TestCase

import numpy as np

from SEAL.lib import compute_knot_insertion_matrix


class TestComputeKnotInsertionMatrix(TestCase):
    def test_compute_knot_insertion_matrix(self):
        """
        Given:
            p = 2
            tau = [-1, -1, -1, 0, 1, 1, 1]
            t = [-1, -1, -1, -0.5, 0, 0.5, 1, 1, 1]
        When:
            Representing the coarse linear B spline
            as a linear combination of the fine linear B splines
        Then:
            Knot Insertion Matrix A =
                [[ 1.    0.    0.    0.  ]
                [ 0.5   0.5   0.    0.  ]
                [ 0.    0.75  0.25  0.  ]
                [ 0.    0.25  0.75  0.  ]
                [ 0.    0.    0.5   0.5 ]
                [ 0.    0.    0.    1.  ]]
        """
        p = 2
        tau = [-1, -1, -1, 0, 1, 1, 1]
        t = [-1, -1, -1, -0.5, 0, 0.5, 1, 1, 1]

        expected_A = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.5, 0.5, 0.0, 0.0],
                [0.0, 0.75, 0.25, 0.0],
                [0.0, 0.25, 0.75, 0.0],
                [0.0, 0.0, 0.5, 0.5],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        computed_A = compute_knot_insertion_matrix(p, tau, t)

        for e, c in zip(expected_A, computed_A):
            np.testing.assert_array_almost_equal(e, c)
