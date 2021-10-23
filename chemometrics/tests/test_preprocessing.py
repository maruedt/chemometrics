# Copyright 2021 Matthias Rüdt
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

import chemometrics as cm  # accounts for relativ path
from chemometrics import preprocessing as pp
import numpy as np
import unittest
import scipy.sparse as sparse
import matplotlib
import matplotlib.pyplot as plt


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

    def test_w_symmetric_ls(self):
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
    Test the `Emsc` preprocessing.
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
            emsc = cm.Emsc(p_order=0, background=bg)
            D_pretreated = emsc.fit_transform(D)
            coefficients = emsc.coefficients_
            self.assertTrue(D_pretreated.shape == (n_series, n_variables))
            self.assertTrue(coefficients.shape[0] == n_series)

    def test_background_subtraction(self):
        r"""
        Test whether background subtraction works
        """
        n_series, n_variables = (10, 50)
        # generate dummy data and background
        scaler = np.arange(n_variables)
        D = np.ones([n_series, n_variables]) * scaler[:, None].T
        background = 0.5 * D[0, :]
        background = background[:, None]
        emsc = cm.Emsc(p_order=0, background=background)
        D_pretreated = emsc.fit_transform(D)
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
            whittaker = cm.Whittaker(penalty=penalty, constraint_order=i)
            X_smoothed = whittaker.fit_transform(X)
            self.assertEqual(X_smoothed.shape, shape)

    def test_shape_deriv(self):
        r"""
        Test that size of axis 1 of transformed decreases with derivative
        """
        shape = (100, 50)
        penalty = 100
        deriv_order = range(1, 5)
        constraint_order = 5
        X = np.random.normal(size=shape)
        for i in deriv_order:
            whittaker = cm.Whittaker(
                penalty=penalty,
                constraint_order=constraint_order,
                deriv=i
            )
            X_smoothed = whittaker.fit_transform(X)
            expected_shape = (shape[0], shape[1]-i)
            self.assertEqual(X_smoothed.shape, expected_shape)

    def test_null_smoothing(self):
        r"""
        Test that very weak smoothing returns itself
        """
        shape = (2, 50)
        penalty = 0
        X = np.random.normal(size=shape)
        whittaker = cm.Whittaker(penalty=penalty)
        X_smoothed = whittaker.fit_transform(X)
        self.assertTrue(np.all(np.isclose(X, X_smoothed)))

    def test_max_smoothing(self):
        r"""
        Test that very strong smoothing leads to expected results.
        """
        # first order penality, no derivative -> constant
        shape = (2, 50)
        penalty = 1e9
        diff_order = 1
        X = np.random.normal(size=shape) + np.arange(shape[1])[:, None].T
        whittaker = cm.Whittaker(penalty=penalty, constraint_order=diff_order)
        X_smoothed = whittaker.fit_transform(X)
        self.assertTrue(np.allclose(X_smoothed, X.mean(axis=1)[:, None]))

        # first order penality, first derivative -> zeros
        whittaker.deriv = 1
        X_smoothed = whittaker.transform(X)
        self.assertTrue(np.allclose(X_smoothed, 0, atol=1.e-5))

        # second order penality, first order derivative -> constant
        whittaker.constraint_order = 2
        X_smoothed = whittaker.fit_transform(X)
        self.assertTrue(np.allclose(X_smoothed,
                                    X_smoothed.mean(axis=1)[:, None]))

    def test_sp_diff_matrix(self):
        r"""
        Test the ``whittaker`` helper function ``_sp_diff_matrix``
        """
        mat_size = 10
        difforder_list = [1, 2, 3]
        # test for different diff orders that sparse matrix is returned and
        # the results are the same as from numpy diff.
        for difforder in difforder_list:
            diff_mat = pp._sp_diff_matrix(mat_size, diff_order=difforder)
            self.assertTrue(sparse.isspmatrix_csc(diff_mat))
            full = diff_mat.toarray()
            comp = np.diff(np.eye(mat_size), n=difforder, axis=0)
            is_close = np.isclose(full, comp)
            self.assertTrue(np.all(is_close))

    def test_calc_whittaker_h_bar(self):
        r"""
        Test that ``_calc_whittaker_h_bar`` returns a float.
        """
        n_var = [30, 300]
        penalty = 0.5
        constraint_order = 2
        for var in n_var:
            h_bar = pp._calc_whittaker_h_bar(var, penalty, constraint_order)
            self.assertIsInstance(h_bar, float)

    def test_whittaker_cve(self):
        r"""
        Test whittaker cross validation returns float.
        """
        penalty = 1e5
        n_wl = 200
        n_band = 20
        bandwidth = 1
        X = cm.generate_background(n_wl) + cm.generate_spectra(n_wl, n_band,
                                                               bandwidth)
        X = X.T
        whittaker = cm.Whittaker(penalty=penalty)
        whittaker.fit(X)
        cve = whittaker.score(X)
        self.assertIsInstance(cve, float)

    def test_autofit(self):
        r"""
        Test auto fit function of whittaker.
        """
        n_wl = 200
        n_band = 20
        bandwidth = 1
        n_samples = 50
        S = cm.generate_background(n_wl).T + cm.generate_spectra(n_wl, n_band,
                                                                 bandwidth)
        C = np.random.uniform(size=n_samples)
        X = C[:, None] * S
        X = X + np.random.normal(size=X.shape, scale=0.1)
        whittaker = cm.Whittaker()
        X_smoothed = whittaker.fit_transform(X)
        self.assertIsInstance(whittaker.penalty_, float)
        self.assertEqual(X_smoothed.shape, X.shape)

    def test_plot(self):
        r"""
        Test that plot function runs
        - returns axes
        - a similar minimum is shown as obtained by auto-optimziation
        - the penalty remains the same before and after running the function

        """
        X, _ = cm.generate_data()
        whittaker = cm.Whittaker()

        autoscore = whittaker.fit(X).score(X)
        autopenalty = whittaker.penalty_

        plt.figure()
        ax = whittaker.plot(X)
        self.assertIsInstance(ax, matplotlib.axes.Axes)

        y_plotted = ax.lines[0].get_ydata()
        min_plotted = np.min(y_plotted)
        self.assertTrue(np.abs(autoscore-min_plotted) < .5)
        self.assertFalse(np.allclose(y_plotted, min_plotted))

        self.assertEqual(whittaker.penalty_, autopenalty)


class TestAsymWhittaker(unittest.TestCase):
    r"""
    Test ``AsymWhittaker`` background correction.
    """

    def test_shape(self):
        """
        Test shape of return argument
        """
        shape = (100, 50)
        penalty = 100
        diff_order = [1, 2]
        X = np.random.normal(size=shape)
        for i in diff_order:
            aw = cm.AsymWhittaker(penalty=penalty, constraint_order=i)
            X_smoothed = aw.fit_transform(X)
            self.assertEqual(X_smoothed.shape, shape)

    def test_null_smoothing(self):
        r"""
        Test that very weak background subtraction returns 0.
        """
        shape = (1, 50)
        penalty = 0
        X = np.random.normal(size=shape)
        aw = cm.AsymWhittaker(penalty=penalty)
        X_smoothed = aw.fit_transform(X)
        self.assertTrue(np.all(np.isclose(0, X_smoothed)))

    def test_very_asymmetric_bg(self):
        r"""
        Test that a high asym_factor leads to no negative values.
        """
        shape = (1, 50)
        penalty = 100
        diff_order = 2
        asym_factor = 1 - 1e-12
        X = np.random.normal(size=shape)
        aw = cm.AsymWhittaker(penalty=penalty, constraint_order=diff_order,
                              asym_factor=asym_factor)
        X_smoothed = aw.fit_transform(X)
        close_to_zero = np.isclose(0, X_smoothed, atol=1e-5)
        greater_than_zero = X_smoothed > 0
        self.assertTrue(np.all(close_to_zero | greater_than_zero))
