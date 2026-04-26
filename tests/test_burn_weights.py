import unittest

import numpy as np

from poker44.base.validator import build_weight_vector_from_scores


class BurnWeightTests(unittest.TestCase):
    def test_enforces_burn_fraction_when_miner_scores_exist(self):
        scores = np.array([10.0, 4.0, 6.0], dtype=np.float32)

        weights = build_weight_vector_from_scores(scores)

        self.assertAlmostEqual(float(weights[0]), 0.97, places=6)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
        self.assertAlmostEqual(float(weights[1]), 0.012, places=6)
        self.assertAlmostEqual(float(weights[2]), 0.018, places=6)

    def test_burns_everything_when_no_positive_miner_scores_exist(self):
        scores = np.array([0.4, 0.0, 0.0], dtype=np.float32)

        weights = build_weight_vector_from_scores(scores)

        self.assertAlmostEqual(float(weights[0]), 1.0, places=6)
        self.assertAlmostEqual(float(weights[1]), 0.0, places=6)
        self.assertAlmostEqual(float(weights[2]), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
