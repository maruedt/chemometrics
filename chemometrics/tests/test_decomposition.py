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

import chemometrics as cm
import numpy as np
import unittest
from chemometrics.tests.test_base import TestLVmixin
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import matplotlib


class TestPCA(unittest.TestCase, TestLVmixin):
    """
    Test PCA class
    """
    def setUp(self):
        self.n_wl = 50
        self.n_samples = 500
        self.n_tests = 100
        self.n_conc = 2

        self.Y = np.random.normal(size=[self.n_samples, self.n_conc])
        noise = 0.1
        spectra = np.zeros(shape=[self.n_wl, self.n_conc])

        for i in range(self.n_conc):
            spectra[:, i] = cm.generate_spectra(self.n_wl, self.n_wl//20, 1)

        self.X = self.Y @ spectra.T + np.random.normal(
            scale=noise,
            size=[self.n_samples, self.n_wl]
        )
        self.model = cm.PCA()
        self.model = self.model.fit(self.X)

        self.Y_test = np.random.normal(size=[self.n_tests, self.n_conc])
        self.X_test = self.Y_test @ spectra.T + np.random.normal(
            scale=noise,
            size=[self.n_tests, self.n_wl]
        )

    def tearDown(self):
        """
        Clean plots etc
        """
        matplotlib.pyplot.close()


class TestFit_pca(unittest.TestCase):
    """
    Test fit_pca function.
    """

    def test_raise_TypeError_pipeline(self):
        """
        Test if function raises a TypeError if wrong pipeline is provided
        """
        X, Y = cm.generate_data()
        with self.assertRaises(TypeError):
            cm.fit_pca(X, pipeline="string as an example wrong object")

    def test_pipeline(self):
        """
        Test if pipeline is accepted as input argument
        """
        X, Y = cm.generate_data()
        pipeline = make_pipeline(StandardScaler(), cm.PCA())
        model, analysis = cm.fit_pca(X, pipeline=pipeline)

    def test_accept_cv_object(self):
        """
        Test if function accepts a cv object
        """
        X, Y = cm.generate_data()
        n_splits = 7
        model, analysis = cm.fit_pca(X, cv_object=KFold(n_splits=n_splits))
        self.assertTrue(analysis['q2'].shape[0] == n_splits)

    def test_return_type(self):
        """
        Test if functions returns an object of type Pipeline
        """
        max_lv = 11
        X, Y = cm.generate_data()
        # basic example
        self._assert_return_args(cm.fit_pca(X, max_lv=max_lv))

        # with pipeline provided
        pipe = make_pipeline(cm.PCA())
        self._assert_return_args(cm.fit_pca(X, pipeline=pipe))

        # with cross validation
        cv = KFold()
        self._assert_return_args(cm.fit_pca(X, cv_object=cv))

    def _assert_return_args(self, returned_args):
        """
        Assert that the returned arguments of fit_pca are correct
        """
        # first return argument must be a pipeline
        pipeline = returned_args[0]
        self.assertIsInstance(pipeline, Pipeline)
        self.assertIsInstance(pipeline[-1], cm.PCA)

        # second return argument must capture info on calibration/cv in dict
        calibration_info = returned_args[1]
        self.assertIsInstance(calibration_info, dict)
        keys = ['q2', 'r2', 'figure_cv', 'figure_model']
        for key in keys:
            self.assertIn(key, calibration_info)

    def tearDown(self):
        """
        Clean plots etc
        """
        matplotlib.pyplot.close()


if __name__ == "__main__":
    unittest.main()
