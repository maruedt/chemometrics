# Copyright 2020 Matthias Rüdt
#
# This file is part of chemometrics.
#
# chemometrics is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# chemometrics is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with chemometrics.  If not, see <https://www.gnu.org/licenses/>.

from .context import chemometrics as cm  # accounts for relativ path
import numpy as np
import unittest


class TestAsym_ls(unittest.TestCase):
    """
    Test cases for `asym_ls`
    """

    def test_shape(self):
        """
        Test that output shape is as expected
        """
        shape_x = [10, 3]
        shape_y = [10, 2]
        expected_shape = (shape_x[1], shape_y[1])

        X = np.random.normal(size=shape_x)
        y = np.random.normal(size=shape_y)
        output_shape = cm.asym_ls(X, y).shape
        self.assertEqual(expected_shape, output_shape)

    def test_w_ju7symmetric_ls(self):
        """
        Test if symmetric weights results in least squares solution
        """
        shape_x = [10, 3]
        shape_y = [10, 1]
        X = np.random.normal(size=shape_x)
        y = np.random.normal(size=shape_y)

        beta_als = cm.asym_ls(X, y, asym_factor=0.5)
        beta_ls = np.linalg.lstsq(X, y, rcond=-1)[0]
        self.assertTrue(np.all(np.isclose(beta_als, beta_ls)))

    def test_location(self):
        """
        Test if different asym_factors produce qualitatively correct effect
        """
        shape = [10, 1]
        asym_factors = [0.001, 0.0011, 0.5, 0.9, 0.99]
        x = np.ones(shape)
        y = np.arange(shape[0])[:, None]
        last_beta = 0

        for factor_i in asym_factors:
            current_beta = cm.asym_ls(x, y, asym_factor=factor_i)
            self.assertTrue(current_beta > last_beta)


class TestEmsc(unittest.TestCase):
    r"""
    Test the `emsc` function
    """

    def test_shape(self):
        r"""
        Check the shape of the return matrix
        """
        n_series, n_variables = (10, 50)
        # generate dummy data and background
        scaler = np.linspace(0, 10, num=n_variables)
        D = np.ones([n_series, n_variables]) * scaler[:, None].T
        background = 0.5 * D[0, :]
        background = background[:, None]
        background_list = [None, background]
        # iterate over different inputs
        for bg in background_list:
            D_pretreated, coefficients = cm.emsc(D, p_order=0, background=bg)
            self.assertTrue(D_pretreated.shape == (n_series, n_variables))
            self.assertTrue(coefficients.shape[0] == n_series)

    def test_background_subtraction(self):
        r"""
        Test wether background subtraction works
        """
        n_series, n_variables = (10, 50)
        # generate dummy data and background
        scaler = np.arange(n_variables)
        D = np.ones([n_series, n_variables]) * scaler[:, None].T
        background = 0.5 * D[0, :]
        background = background[:, None]
        D_pretreated, coefficients = cm.emsc(
            D,
            p_order=0,
            background=background
        )
        self.assertTrue(np.all(np.isclose(np.zeros([n_series, n_variables]),
                        D_pretreated)))


class Testwhittaker(unittest.TestCase):
    r"""
    Test ``whittaker`` smoother.
    """

    def test_shape(self):
        shape = (100, 50)
        penalty = 100
        diff_order = [1, 2]
        X = np.random.normal(size=shape)
        for i in diff_order:
            X_smoothed = cm.whittaker(X, penalty, constraint_order=i)
            self.assertEqual(X_smoothed.shape, shape)

    def test_null_smoothing(self):
        r"""
        Test that very weak smoothing returns itself
        """
        shape = (50, 1)
        penalty = 0
        X = np.random.normal(size=shape)
        X_smoothed = cm.whittaker(X, penalty)
        self.assertTrue(np.all(np.isclose(X, X_smoothed)))

    def test_max_smoothing(self):
        r"""
        Test that very strong smoothing leads to polynomial.
        """
        shape = (50, 1)
        penalty = 1e9
        diff_order = 1
        X = np.random.normal(size=shape) + np.arange(shape[1])
        X_smoothed = cm.whittaker(X, penalty, diff_order)
        is_close = np.isclose(X_smoothed, X.mean())
        self.assertTrue(np.all(is_close))
