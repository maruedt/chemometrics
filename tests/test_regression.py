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

from .context import chemometrics as cm  # accounts for relativ path
import numpy as np
import unittest
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
import matplotlib


class TestPLSRegression(unittest.TestCase):
    """
    Test adjustments done to PLSRegression child class
    """

    def setUp(self):
        self.n_wl = 100
        self.n_samples = 100
        self.n_conc = 2

        self.Y = np.random.uniform(size=[self.n_samples, self.n_conc])
        noise = 0.1
        spectra = np.zeros(shape=[self.n_wl, self.n_conc])

        for i in range(self.n_conc):
            spectra[:, i] = cm.generate_spectra(self.n_wl, self.n_wl//20, 1)

        self.X = self.Y @ spectra.T + np.random.normal(
            scale=noise,
            size=[self.n_samples, self.n_wl]
        )
        self.pls = cm.PLSRegression()
        self.pls = self.pls.fit(self.X, self.Y)

    def test_vip_shape(self):
        """
        Test the shape of the vip variable
        """
        self.assertTrue(self.pls.vip_.shape == (self.n_wl, ))

    def test_vip_value(self):
        """
        Test that sum of squared VIPs == number of X variables (definition!)
        """
        self.assertTrue(np.isclose(np.sum(self.pls.vip_**2), self.n_wl))

    def test_hat_shape(self):
        """
        Test the shape of the vip variable
        """
        hat = self.pls.hat(self.X)
        self.assertTrue(hat.shape == (self.n_samples, self.n_samples))

    def test_hat_works(self):
        """
        Test that hat works according to its definition i.e. H * Y = Ŷ
        """
        hat = self.pls.hat(self.X)
        # mean centered prediction
        Y_hat = self.pls.predict(self.X) - np.mean(self.Y, axis=0)
        self.assertTrue(np.allclose(hat @ self.Y, Y_hat))

    def test_hat_symmetric(self):
        """
        Test that hat matrix is symmetric
        """
        hat = self.pls.hat(self.X)
        self.assertTrue(np.allclose(hat, hat.T))

    def test_hat_indempotent(self):
        """
        Test that hat matrix is indempotent (hat = hat squared)
        """
        hat = self.pls.hat(self.X)
        self.assertTrue(np.allclose(hat, hat @ hat))

    def test_leverage_shape(self):
        """
        Test that leverage provides correct matrix shape
        """
        leverage = self.pls.leverage(self.X)
        self.assertTrue(leverage.shape == (self.n_samples, ))

    def test_leverage_definition(self):
        """
        Test that leverage is diag of the hat matrix
        """
        leverage = self.pls.leverage(self.X)
        hat = self.pls.hat(self.X)
        self.assertTrue(np.allclose(leverage, np.diag(hat)))

    def test_plot(self):
        """
        Test that plot generates 4 subplot in figure
        """
        axes = self.pls.plot(self.X, self.Y)
        for ax in axes:
            self.assertIsInstance(ax, matplotlib.axes.Axes)


class TestFit_pls(unittest.TestCase):
    """
    Test fit_pls function.
    """

    def test_raise_TypeError_pipeline(self):
        """
        Test if function raises a TypeError if wrong pipeline is provided
        """
        X, Y = cm.generate_data()
        with self.assertRaises(TypeError):
            cm.fit_pls(X, Y, pipeline="string as an example wrong object")

    def test_accept_cv_object(self):
        """
        Test if function raises a TypeError of wrong CV object is provided
        """
        X, Y = cm.generate_data()

        n_splits = 7
        model, analysis = cm.fit_pls(X, Y, cv_object=KFold(n_splits=n_splits))

        self.assertTrue(analysis['q2'].shape[0] == n_splits)

    def test_return_type(self):
        """
        Test if functions returns an object of type Pipeline
        """
        max_lv = 11
        X, Y = cm.generate_data()
        # basic example
        self._assert_return_args(cm.fit_pls(X, Y, max_lv=max_lv))

        # with pipeline provided
        pipe = make_pipeline(cm.PLSRegression())
        self._assert_return_args(cm.fit_pls(X, Y, pipeline=pipe))

        # with cross validation
        cv = KFold()
        self._assert_return_args(cm.fit_pls(X, Y, cv_object=cv))

    def _assert_return_args(self, returned_args):
        """
        Assert that the returned arguments of fit_pls are correct
        """
        # first return argument must be a pipeline
        pipeline = returned_args[0]
        self.assertIsInstance(pipeline, Pipeline)
        self.assertIsInstance(pipeline[-1], PLSRegression)

        # second return argument must capture info on calibration/cv in dict
        calibration_info = returned_args[1]
        self.assertIsInstance(calibration_info, dict)
        keys = ['q2', 'r2', 'figure_cv', 'figure_model']
        for key in keys:
            self.assertIn(key, calibration_info)
