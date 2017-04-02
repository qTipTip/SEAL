from unittest import TestCase

from SEAL.lib import index


class TestIndex(TestCase):
    def test_index_endpoints(self):
        """
        Given:
            knot vector t = [0, 0, 0, 1, 2, 3, 3, 3]
        When:
            x = 0, 0.5, 1, 1.5, 2, 2.5, 3
        Then:
            index = 2, 2, 3, 3, 4, 4, 4
        """
        t = [0, 0, 0, 1, 2, 3, 3, 3]
        x = [0, 0.5, 1, 1.5, 2, 2.5, 3]

        expected_values = [2, 2, 3, 3, 4, 4, 4]
        computed_values = [index(x_i, t) for x_i in x]

        for e, c in zip(expected_values, computed_values):
            self.assertAlmostEqual(e, c)
