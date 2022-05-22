# Copyright 2022 Matthias Rüdt
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

import matplotlib
from scipy.stats import binom
import numpy as np


class TestLVmixin():
    """
    A mixin for running LVmixin tests
    """
    def test_dmodx_shape(self):
        """
        Test that dmodx provides a vector of the correct shape
        """
        dmodx = self.model.dmodx(self.X)
        self.assertTrue(dmodx.shape == (self.n_samples, ))

        dmodx = self.model.dmodx(self.X, normalize=False)
        self.assertTrue(dmodx.shape == (self.n_samples, ))

        dmodx = self.model.dmodx(self.X, absolute=True)
        self.assertTrue(dmodx.shape == (self.n_samples, ))

    def test_dmodx_length(self):
        """
        Test that dmodx = 0 if data from hyperplan is taken
        """
        X_hat = self.model.inverse_transform(self.model.transform(self.X))
        dmodx = self.model.dmodx(X_hat, normalize=False)
        self.assertTrue(np.allclose(dmodx, 0))

    def test_score_plus_dmodx_length(self):
        """
        Test that the variation on the model plane and the variation
        orthogonal to the model plane yield the total variation.
        """
        dmodx = self.model.dmodx(self.X, normalize=False)
        absolut_dmodx = dmodx**2 * (self.X.shape[1] - self.model.n_components)
        X_hat_bar = self.model.x_scores_ @ self.model.x_loadings_.T
        ss_X_hat = np.sum(X_hat_bar**2, axis=1)
        X_bar = self.X - np.mean(self.X, axis=0)
        ss_X = np.sum(X_bar**2, axis=1)
        self.assertTrue(np.allclose(ss_X_hat + absolut_dmodx, ss_X,
                                    atol=1e-2))

    def test_distance_plot_return_arg(self):
        """
        Test that distance_plot returns an axis cv_object
        """
        ax = self.model.distance_plot(self.X)
        self.assertIsInstance(ax, list)
        for inst in ax:
            self.assertIsInstance(inst, matplotlib.axes.Subplot)

    def test_crit_dmodx(self):
        """
        Test that number of outliers corresponds

        Performs a binomial statistical test. This test fails more often then
        the confidence level would predict.

        Potential explanation:
        Since dmodx
        is only approximately f2 distributed [1], crit_dmodx is not completely
        accurate. Especially when considering outliers at high confidence, the
        test seems to be too conservative. As a work-around, the binomial test
        is performed at a confidence level of 0.5 for the crit_dmodx (50%
        outliers).

        References
        ----------
        .. [1] L. Eriksson, E. Johansson, N. Kettaneh-Wold, J. Trygg, C.
        Wikström, and S. Wold. Multi- and Megavariate Data Analysis, Part I
        Basic Principles and Applications. Second Edition.
        """
        f_confidence = 0.50
        crit_dmodx = self.model.crit_dmodx(confidence=f_confidence)
        dmodx = self.model.dmodx(self.X_test)
        count = np.sum(dmodx > crit_dmodx)

        self._test_binom(1-f_confidence, self.n_tests, count)

    def test_crit_dhypx(self):
        """
        Test that number of outliers corresponds for crit_dhypx

        Performs a biomial statistical test. This test fails more often
        then the confidence level would predict. Potential explanation:
        crit_dhypx is not corrected for the bias in estimating the variance of
        the scores based on the training set [1]. This leads to a biased
        normalized dhypx calculations.

        References
        ----------
        .. [1] L. Eriksson, E. Johansson, N. Kettaneh-Wold, J. Trygg, C.
        Wikström, and S. Wold. Multi- and Megavariate Data Analysis, Part I
        Basic Principles and Applications. Second Edition.
        """
        f_confidence = 0.95
        crit_dhypx = self.model.crit_dhypx(confidence=f_confidence)
        dhypx = self.model.dhypx(self.X_test)
        count = np.sum(dhypx > crit_dhypx)
        self._test_binom(1-f_confidence, self.n_tests, count)

    def _test_binom(self, p, n_samples, n_positiv, false_positiv=0.005):
        """
        Perform a binomial test
        """
        pdist = binom(n_samples, p)
        limit_low = pdist.ppf(false_positiv/2)
        limit_high = pdist.ppf(1 - false_positiv/2)
        self.assertTrue(n_positiv >= limit_low)
        self.assertTrue(n_positiv <= limit_high)
