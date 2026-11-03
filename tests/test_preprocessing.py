import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.preprocessing import extract_feature_vector, moving_average


class TestPreprocessing(unittest.TestCase):
    def test_moving_average(self):
        values = [1, 2, 3, 4, 5]
        self.assertEqual(moving_average(values, width=2), [1.0, 1.5, 2.5, 3.5, 4.5])

    def test_feature_vector_has_six_values(self):
        temperatures = [4.0 + 0.01 * i for i in range(30)]
        vibrations = [0.4 + (i % 3) * 0.02 for i in range(15)]
        features = extract_feature_vector(temperatures, vibrations)
        self.assertEqual(len(features), 6)
        self.assertGreater(features[3], 0.0)
        self.assertGreaterEqual(features[4], features[3])


if __name__ == "__main__":
    unittest.main()
