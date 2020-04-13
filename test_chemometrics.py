import chemometrics as cm
import numpy as np
import unittest
import pdb

class TestAsym_ls(unittest.TestCase):
    def test_shape(self):
        """
        Test that output shape is as expected
        """
        shape_x = [10,3]
        shape_y = [10,1]
        expected_shape = (shape_x[1], shape_y[1])

        X = np.random.normal(size=shape_x)
        y = np.random.normal(size=shape_y)
        output_shape = cm.asym_ls(X, y).shape
        self.assertEqual(expected_shape, output_shape)

    def test_w_ju7symmetric_ls(self):
        """
        Test if symmetric weights results in least squares solution
        """

        shape_x = [10,3]
        shape_y = [10,1]
        X = np.random.normal(size=shape_x)
        y = np.random.normal(size=shape_y)

        beta_als = cm.asym_ls(X, y, asym_factor=0.5)
        beta_ls = np.linalg.lstsq(X, y, rcond=None)[0]
        self.assertTrue(np.all(np.isclose(beta_als, beta_ls)))

    def test_location(self):
        """
        Test if different asym_factors produce qualitatively correct effect
        """
        shape = [10,1]
        asym_factors = [0.001, 0.0011, 0.5, 0.9, 0.99]
        x = np.ones(shape)
        y = np.arange(shape[0])[:,None]
        last_beta = 0

        for factor_i in asym_factors:
            current_beta = cm.asym_ls(x, y, asym_factor=factor_i)
            self.assertTrue(current_beta>last_beta)


class TestGenerate_spectra(unittest.TestCase):
    def test_shape(self):
        """
        Test if correct shape is generated
        """
        n_wl = 200
        expected_shape = (200,)
        output_shape = cm.generate_spectra(n_wl, 2, 50).shape
        self.assertEqual(expected_shape, output_shape)


if __name__ == '__main__':
    unittest.main()
