"""
Testing chemometrics.mcr.regressors

"""

import numpy as np

from chemometrics.mcr.metrics import mse
import unittest


class TestMSE(unittest.TestCase):
    """
    Test mse calculation
    """

    def test_mse(self):
        A = np.ones((3, 3))
        B = np.eye(3)
        error = mse(None, None, A, B)
        self.assertEqual(error, 6/9)


if __name__ == '__main__':
    unittest.main()
