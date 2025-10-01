import unittest
import numpy as np
from t1prep.utils import find_largest_cluster, smart_round

class TestUtils(unittest.TestCase):

    def test_find_largest_cluster_no_cluster(self):
        """Test that an empty mask is returned when no clusters are found."""
        binary_volume = np.zeros((10, 10, 10), dtype=bool)
        cluster_mask = find_largest_cluster(binary_volume)
        self.assertTrue(np.all(cluster_mask == False))
        self.assertEqual(cluster_mask.shape, binary_volume.shape)

    def test_find_largest_cluster_one_cluster(self):
        """Test that the largest cluster is correctly identified."""
        binary_volume = np.zeros((10, 10, 10), dtype=bool)
        binary_volume[2:5, 2:5, 2:5] = True
        cluster_mask = find_largest_cluster(binary_volume)
        self.assertTrue(np.all(cluster_mask == binary_volume))

    def test_find_largest_cluster_multiple_clusters(self):
        """Test that the largest cluster is identified among multiple clusters."""
        binary_volume = np.zeros((10, 10, 10), dtype=bool)
        binary_volume[2:4, 2:4, 2:4] = True  # Smaller cluster
        binary_volume[6:9, 6:9, 6:9] = True  # Larger cluster

        expected_mask = np.zeros((10, 10, 10), dtype=bool)
        expected_mask[6:9, 6:9, 6:9] = True

        cluster_mask = find_largest_cluster(binary_volume)
        self.assertTrue(np.all(cluster_mask == expected_mask))

    def test_smart_round(self):
        """Test the smart_round function."""
        self.assertEqual(smart_round(0.123456789), 0.12346)
        self.assertEqual(smart_round(1.123456789), 1.123)
        self.assertEqual(smart_round(10.123456789), 10.12)
        self.assertEqual(smart_round(-0.123456789), -0.12346)
        self.assertEqual(smart_round(-1.123456789), -1.123)
        self.assertEqual(smart_round(-10.123456789), -10.12)
        self.assertEqual(smart_round(0), 0.0)

if __name__ == '__main__':
    unittest.main()
