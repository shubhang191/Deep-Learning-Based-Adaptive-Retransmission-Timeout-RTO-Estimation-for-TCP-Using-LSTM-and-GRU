import unittest

import numpy as np

from jacobson import jacobson_rto, karn_backoff


class JacobsonTests(unittest.TestCase):
    def test_initial_values(self):
        result = jacobson_rto([100.0])
        self.assertAlmostEqual(result["srtt"][0], 100.0)
        self.assertAlmostEqual(result["rttvar"][0], 50.0)
        self.assertAlmostEqual(result["rto"][0], 300.0)

    def test_recursion_order_and_values(self):
        result = jacobson_rto([100.0, 200.0])
        # Previous SRTT=100, RTTVAR=50, and beta=1/4.
        self.assertAlmostEqual(result["rttvar"][1], 62.5)
        self.assertAlmostEqual(result["srtt"][1], 112.5)
        self.assertAlmostEqual(result["rto"][1], 362.5)

    def test_minimum_rto_can_be_clamped(self):
        result = jacobson_rto([20.0], min_rto=1000.0)
        self.assertEqual(result["rto"][0], 1000.0)

    def test_karn_backoff_helper(self):
        self.assertAlmostEqual(karn_backoff(100.0, 2), 400.0)
        self.assertAlmostEqual(karn_backoff(100.0, 10, max_rto=300.0), 300.0)


if __name__ == "__main__":
    unittest.main()
