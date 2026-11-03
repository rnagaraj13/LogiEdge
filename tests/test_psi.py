import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monitoring.psi import calculate_psi, proportions


class TestPSI(unittest.TestCase):
    def test_zero_for_identical_distribution(self):
        ref = proportions([0.8] * 100)
        cur = proportions([0.8] * 100)
        self.assertAlmostEqual(calculate_psi(ref, cur), 0.0, places=7)

    def test_positive_for_different_distribution(self):
        ref = proportions([0.9] * 100)
        cur = proportions([0.1] * 100)
        self.assertGreater(calculate_psi(ref, cur), 0.1)


if __name__ == "__main__":
    unittest.main()
