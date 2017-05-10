from unittest import TestCase

from SEAL.lib import insert_midpoints


class TestInsertMidpoints(TestCase):
    def test_insert_midpoints(self):
        """
        Given:
            p = 2
            p + 1 regular knot vector t = [0, 0, 0, 1, 2, 3, 3, 3]
        When:
            Inserting midpoints
        Then:
            refined_t = [0, 0, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3, 3]
        """
        p = 2
        t = [0, 0, 0, 1, 2, 3, 3, 3]

        expected_refined = [0, 0, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3, 3]
        computed_refined = insert_midpoints(t, p)

        for e, c in zip(expected_refined, computed_refined):
            self.assertAlmostEqual(e, c)