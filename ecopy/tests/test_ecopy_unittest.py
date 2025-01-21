import unittest
import numpy as np
from ecopy import load_data, wt_mean, wt_var, wt_scale, diversity, rarefy, transform

class TestECOPY(unittest.TestCase):

    def setUp(self):
        pass

    def test_wt_mean(self):
        x = [2, 3, 4, 5]
        w = [0.1, 0.1, 0.5, 0.3]
        res = wt_mean(x, w)
        self.assertEqual(res, 4.0)

    def test_wt_var(self):
        x = [2, 3, 4, 5]
        w = [0.1, 0.1, 0.5, 0.3]
        res = wt_var(x, w)
        self.assertEqual(res, 1.25)    

    def test_wt_scale(self):
        x = [2, 3, 4, 5]
        w = [0.1, 0.1, 0.5, 0.3]
        res = np.round(wt_scale(x,w), 2)
        truth = res == [-1.79, -0.89, 0, 0.89]
        self.assertEqual(truth.sum(), 4)

    def test_diversity(self):
        sp = np.array([0, 1, 2, 3, 0]).reshape(1,5)
        div = np.round(diversity(sp, num_equiv=False))
        self.assertEqual(div, 1)

    def test_rarefy(self):
        BCI = load_data('BCI')
        rareRich = np.round(rarefy(BCI, 'rarefy'))
        self.assertEqual(rareRich[1], 77)

    def test_log_median_ratio(self):
        counts_arr = np.array(([2, 3, 6],
                               [10, 11, 24]))
        # rows are different species, columns are different samples
        observed_norm = transform(counts_arr, axis=1, method='log_median_ratio')
        expected_norm = np.array([[4.217, 4.662, 4.273],
                                  [15.462, 13.986, 15.260]])
        self.assertTrue(np.array_equal(np.round(expected_norm), np.round(observed_norm)))

        # transposed data the data
        observed_norm = transform(counts_arr.transpose(), axis=0, method='log_median_ratio')
        expected_norm = np.array([[4.217, 4.662, 4.273],
                                  [15.462, 13.986, 15.260]])
        self.assertTrue(np.array_equal(np.round(expected_norm.transpose()), np.round(observed_norm)))


if __name__ == '__main__':
    unittest.main()
